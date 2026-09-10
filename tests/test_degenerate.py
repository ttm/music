"""What every routine does when a parameter goes to zero.

A rendering routine has a lot of numeric parameters and each of them has
an edge. A duration of zero, a frequency of zero, a width of zero: none is
a sensible thing to ask for, and none is a reason to hand back a
traceback from three frames down, or an array of NaN behind a warning.

So the contract is one of two answers, and the sweep below holds every
export to it:

* it works, returning samples that are finite, or
* it refuses with a `ValueError` that says what the caller did.

Never a `ZeroDivisionError`, an `IndexError`, an `UnboundLocalError` or a
message about broadcasting -- those name the line that gave up rather than
the argument that was wrong. And never a warning, because a warning here
means the routine carried on and returned something built out of NaN.

Seven routines failed this when it was first written. `fade` returned an
envelope 88,200 samples longer than the one asked for; two of the three
glissandos cast a NaN frequency contour to an integer table index and read
a sound out of it; `mix` counted a stereo array's two channels as its
length; `noise`, `gaussian_noise`, `note_with_doppler`, `trill` and
`rhythm_to_durations` each stopped somewhere unhelpful. They are fixed,
and the sweep is what keeps them fixed.
"""

import inspect
import warnings

import numpy as np
import pytest

import music
from test_public_api import ZERO_ARG_EXPORTS, _callable_with_defaults

#: Parameters the sweep leaves alone.  `sample_rate` at zero is a question
#: about arithmetic rather than about sound; `number_of_samples` and
#: `sonic_vector` at zero mean "not supplied" throughout this package, so
#: setting them to zero is asking for the default rather than for an edge.
NOT_AN_EDGE = {"sample_rate", "number_of_samples", "sonic_vector"}


def _numeric_parameters(function):
    """The numeric parameters of `function`, excluding the flags.

    `isinstance(True, int)` is True, so a boolean default would otherwise
    read as a number to set to zero -- which is just the other flag, not
    an edge.
    """
    for name, parameter in inspect.signature(function).parameters.items():
        default = parameter.default
        if name in NOT_AN_EDGE or isinstance(default, bool):
            continue
        if isinstance(default, (int, float)):
            yield name


EDGES = sorted(
    (name, parameter)
    for name in ZERO_ARG_EXPORTS
    for parameter in _numeric_parameters(_callable_with_defaults(name))
)


def test_the_sweep_finds_the_edges():
    """Guard the sweep itself: it must be exercising something."""
    assert len(EDGES) >= 120
    assert ("note", "freq") in EDGES
    assert ("fade", "perc") in EDGES


@pytest.mark.parametrize("name,parameter", EDGES,
                         ids=lambda value: str(value))
def test_a_parameter_at_zero_works_or_is_refused_clearly(name, parameter):
    """One of two answers, and nothing else."""
    function = _callable_with_defaults(name)
    try:
        with warnings.catch_warnings():
            # A RuntimeWarning here is numpy saying it produced NaN and
            # carried on, which is the failure that looks like success.
            warnings.simplefilter("error")
            result = function(**{parameter: 0})
    except ValueError:
        return                      # a deliberate refusal, which is fine
    except Exception as unexpected:  # pragma: no cover - the failure path
        pytest.fail(
            f"{name}({parameter}=0) raised {type(unexpected).__name__}: "
            f"{unexpected}. A degenerate argument should be refused with a "
            "ValueError naming it, not reported from wherever the "
            "arithmetic gave up.")

    if callable(result):
        return                      # `proportional` and friends build bonds

    samples = np.asarray(result, dtype=float)
    assert samples.size == 0 or np.isfinite(samples).all(), (
        f"{name}({parameter}=0) returned samples that are not all finite. "
        "Refuse the argument rather than rendering NaN.")


# ---------------------------------------------------------------------
# The seven, one regression test each
# ---------------------------------------------------------------------

@pytest.mark.parametrize("duration", [0.01, 0.5, 2.0])
def test_a_wholly_linear_fade_is_as_long_as_it_was_asked_for(duration):
    """`perc` is how much of the fade is linear, and 100 meant all of it.

    `loud` reads `number_of_samples=0` as "use the default duration", so
    asking it for the zero-sample exponential part returned two seconds of
    envelope instead: a half-second fade came back 110,250 samples long
    rather than 22,050, which is the default length added to the one
    asked for. The `n0` side of the same split was already guarded.
    """
    expected = int(duration * 44100)
    assert len(music.fade(duration=duration, perc=100)) == expected
    assert len(music.fade(duration=duration, perc=100,
                          fade_out=False)) == expected


def test_a_wholly_linear_fade_is_the_linear_fade():
    """Which is the property that says the fix is the right shape."""
    assert np.allclose(music.fade(duration=0.5, perc=100),
                       music.fade(duration=0.5, method="linear"))
    rising = music.fade(duration=0.5, perc=100, fade_out=False)
    assert rising[0] == pytest.approx(0.0, abs=1e-9)
    assert rising[-1] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("perc", [-1, 101, 150])
def test_a_fade_refuses_a_percentage_that_is_not_one(perc):
    """It used to reach an IndexError from inside `np.hstack`."""
    with pytest.raises(ValueError, match="perc"):
        music.fade(duration=0.5, perc=perc)


def test_a_fade_refuses_a_method_it_does_not_have():
    """It used to raise UnboundLocalError on a name it never bound."""
    with pytest.raises(ValueError, match="lin"):
        music.fade(duration=0.5, method="nonsense")


@pytest.mark.parametrize("glissando", [
    music.note_with_glissando,
    music.note_with_glissando_vibrato,
    music.note_with_two_vibratos_glissando,
])
@pytest.mark.parametrize("start,end", [(0, 220), (220, 0), (220, -220)])
def test_an_exponential_glissando_refuses_what_has_no_ratio(glissando, start,
                                                            end):
    """All three sweep by `start * (end / start) ** t`.

    A start of zero divided by zero. A negative frequency raised to a
    fractional power is NaN -- and the contour is cast to an integer table
    index, so the NaN became some index and the render came back finite,
    plausible and meaningless. Only one of the three refused anything
    before; now one helper refuses for all three.
    """
    with pytest.raises(ValueError, match="positive"):
        glissando(start_freq=start, end_freq=end, duration=0.05)


def test_a_linear_glissando_may_start_at_zero():
    """Since it sweeps by a difference, and zero is a difference away."""
    swept = music.note_with_glissando(start_freq=0, end_freq=220,
                                      duration=0.05, method="lin")
    assert np.isfinite(swept).all()


@pytest.mark.parametrize("first,second", [("stereo", "mono"),
                                          ("mono", "stereo"),
                                          ("stereo", "stereo")])
def test_mix_refuses_a_stereo_sound(first, second):
    """`len()` of a (2, n) array is 2, so mix padded to two samples.

    The same defect `iir` had: a routine that measures a sound with
    `len()` reads a stereo sound as two of something. It surfaced as a
    numpy message about broadcasting shapes, which names neither the
    argument nor the mistake.
    """
    mono = music.note(440, 0.1)
    stereo = np.array([mono, mono])
    sounds = {"mono": mono, "stereo": stereo}
    with pytest.raises(ValueError, match="mix_stereo"):
        music.mix(sounds[first], sounds[second])


def test_mix_still_sums_two_mono_sounds():
    """The fix must not have cost the thing the routine is for."""
    summed = music.mix(music.note(440, 0.1), music.note(660, 0.1))
    assert len(summed) == len(music.note(440, 0.1))


def test_a_noise_refuses_to_be_no_samples_long():
    """It reached `IndexError: index 0 is out of bounds` instead."""
    with pytest.raises(ValueError, match="at least one sample"):
        music.noise(duration=0)
    with pytest.raises(ValueError, match="at least one sample"):
        music.gaussian_noise(duration=0)


def test_a_gaussian_noise_refuses_a_band_with_nothing_in_it():
    """A width of zero zeroed every coefficient, and normalizing an
    all-zero spectrum divides by its own zero range: the caller got an
    array of NaN behind a RuntimeWarning."""
    with pytest.raises(ValueError, match="no frequency"):
        music.gaussian_noise(std=0)


def test_a_doppler_note_refuses_to_be_no_samples_long():
    """It divided by the length it did not have."""
    with pytest.raises(ValueError, match="at least one sample"):
        music.note_with_doppler(duration=0)


def test_a_trill_refuses_a_rate_of_no_notes_per_second():
    """Each note lasts `sample_rate / notes_per_second`, and that divided."""
    with pytest.raises(ValueError, match="notes_per_second"):
        music.trill(notes_per_second=0)


def test_rhythm_to_durations_refuses_when_nothing_gives_a_duration():
    """`duration=0` reads as "not supplied", and so does no bpm and no
    total: the routine then divided None by a sum."""
    with pytest.raises(ValueError, match="gives a duration"):
        music.rhythm_to_durations(duration=0)
