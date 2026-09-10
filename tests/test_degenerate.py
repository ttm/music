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

Nine routines failed this when it was first written. `fade` returned an
envelope 88,200 samples longer than the one asked for; two of the three
glissandos cast a NaN frequency contour to an integer table index and read
a sound out of it; `mix` counted a stereo array's two channels as its
length; `noise`, `gaussian_noise`, `note_with_doppler`, `trill` and
`rhythm_to_durations` each stopped somewhere unhelpful. They are fixed,
and the sweep is what keeps them fixed.

The second half of the file is about one particular edge, the zero
duration, where "it works" and "it refuses" were both being answered and
neither was written down. It renders nothing. The reasoning is in
`test_a_zero_duration_renders_nothing`, and the refusal that used to sit
in the routines now sits at the sinks, where an empty sound stops being
something anyone can act on.
"""

import inspect
import pathlib
import re
import traceback
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
        if not isinstance(default, (int, float)):
            continue
        # Ten pairs defaulted to zero already, so setting them to zero
        # re-ran the default call -- which `test_public_api.py` sweeps
        # anyway. They inflated the count without testing an edge.
        if default == 0:
            continue
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


def test_assessment_md_says_how_many_edges_there_are():
    """The count in that file is a figure like any other in it.

    `tools/assessment_figures.py` measures the figures it knows how to
    run; this one is a property of a test module, so the test module is
    where it is checked -- the same arrangement
    `test_public_api.py` uses for the API reference. Without this the row
    would drift the moment an export gained a numeric parameter, which is
    the failure the whole file-checking arrangement exists to stop, and
    the row went in at 155 while the sweep covered 145.
    """
    assessment = (pathlib.Path(__file__).parent.parent
                  / "ASSESSMENT.md")
    if not assessment.exists():      # pragma: no cover - an sdist only
        pytest.skip("ASSESSMENT.md is missing")

    stated = re.search(r"\*\*(\d+)\*\* \(routine, parameter\) pairs",
                       assessment.read_text())
    assert stated, (
        "ASSESSMENT.md no longer states the pair count where this test "
        "looks for it; fix the pattern rather than deleting the check")
    assert int(stated.group(1)) == len(EDGES), (
        f"ASSESSMENT.md says {stated.group(1)} (routine, parameter) pairs "
        f"and the sweep covers {len(EDGES)}")


def _was_raised_deliberately(error):
    """Whether the package refused, or numpy gave up where it stood.

    Every failure this file was written to forbid is a `ValueError`:
    numpy raises one for a broadcasting mismatch and another for
    concatenating nothing. So catching `ValueError` and calling it a
    refusal accepted exactly the failures the contract forbids -- and it
    did, for as long as this test existed. `amplitude_modulation`,
    `isochronic_tones` and `modulated_noise` failed on broadcasting at a
    zero duration and passed this sweep the whole time; the zero-duration
    sweep further down caught them, which is the only reason they were
    ever found.

    What separates the two is where the exception was raised. A refusal
    this package means is a `raise ValueError` on the frame that raised;
    numpy's comes from an arithmetic line that was trying to do something
    else.
    """
    frame = traceback.extract_tb(error.__traceback__)[-1]
    return "raise ValueError" in (frame.line or "")


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
    except ValueError as refusal:
        assert _was_raised_deliberately(refusal), (
            f"{name}({parameter}=0) raised ValueError from "
            f"{traceback.extract_tb(refusal.__traceback__)[-1].filename}"
            f":{traceback.extract_tb(refusal.__traceback__)[-1].lineno}, "
            f"which is arithmetic giving up rather than a refusal: "
            f"{refusal}")
        return
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

    # Applied rather than returned, which failed differently: the
    # envelope came out longer than the sound it was cut for, so the
    # multiplication raised a broadcast error instead of returning
    # something wrong.
    sound = music.note(440, duration)
    assert len(music.fade(sonic_vector=sound, perc=100)) == len(sound)


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


def test_a_gaussian_noise_refuses_a_band_with_nothing_in_it():
    """A width of zero zeroed every coefficient, and normalizing an
    all-zero spectrum divides by its own zero range: the caller got an
    array of NaN behind a RuntimeWarning."""
    with pytest.raises(ValueError, match="no frequency"):
        music.gaussian_noise(std=0)


def test_a_trill_refuses_a_rate_of_no_notes_per_second():
    """Each note lasts `sample_rate / notes_per_second`, and that divided."""
    with pytest.raises(ValueError, match="notes_per_second"):
        music.trill(notes_per_second=0)


def test_rhythm_to_durations_refuses_when_nothing_gives_a_duration():
    """`duration=0` reads as "not supplied", and so does no bpm and no
    total: the routine then divided None by a sum."""
    with pytest.raises(ValueError, match="gives a duration"):
        music.rhythm_to_durations(duration=0)


# ---------------------------------------------------------------------
# What a zero duration answers with
# ---------------------------------------------------------------------

#: The routines that take a duration and do not answer a zero one with an
#: empty array, and why. Both are deliberate and neither is a render.
NOT_A_ZERO_LENGTH_RENDER = {
    "reverb": (
        "its first_phase_duration is a share of the total, so a total of "
        "zero is a conflict between two parameters rather than a request "
        "for nothing, and it says which two"),
    "rhythm_to_durations": (
        "it returns durations rather than samples, and a zero duration "
        "with no bpm and no total leaves nothing to divide"),
}


def _takes_a_duration(name):
    function = _callable_with_defaults(name)
    return "duration" in inspect.signature(function).parameters


DURATIONS = sorted(name for name in ZERO_ARG_EXPORTS
                   if _takes_a_duration(name))


@pytest.mark.parametrize("name", DURATIONS)
def test_a_zero_duration_renders_nothing(name):
    """Zero samples, keeping the shape the routine would have returned.

    This is the package's convention and it was never written down, so it
    drifted. Of the twenty-seven routines that take a duration, sixteen
    answered a zero one with an empty array, five refused, four failed
    with a numpy message about broadcasting or about concatenating
    nothing, and two -- `binaural_beats` and `monaural_beats` -- returned
    two seconds of audio. Only one of the five refusals was about a zero
    duration at all: `reverb` conflicts with its own
    `first_phase_duration`, and the other four had been added the commit
    before this one, by me, guessing at a convention rather than looking
    for the one already here.

    The two-second answer is the interesting one. Every routine in
    `music.stimulation` sizes itself with `_sample_count`, which returns
    zero honestly, and then renders with `note(number_of_samples=count)`,
    where zero means "not supplied" and gives back the default. The count
    was right; passing it on is what laundered it. So asking for no sound
    got two seconds of it, which is the one answer nobody could want.
    """
    function = _callable_with_defaults(name)
    if name in NOT_A_ZERO_LENGTH_RENDER:
        with pytest.raises(ValueError):
            function(duration=0)
        return

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rendered = np.asarray(function(duration=0), dtype=float)

    assert rendered.size == 0, (
        f"{name}(duration=0) rendered {rendered.shape}. A zero duration is "
        "zero samples; if this routine cannot answer that, register it in "
        "NOT_A_ZERO_LENGTH_RENDER with the reason.")

    # And in the shape it would otherwise have had, so an empty stereo
    # render still stacks and mixes with stereo ones.
    full = np.asarray(function(duration=0.1), dtype=float)
    assert rendered.ndim == full.ndim, (
        f"{name}(duration=0) has {rendered.ndim} dimensions where a real "
        f"render has {full.ndim}; an empty stereo sound is (2, 0)")


@pytest.mark.parametrize("name", sorted(NOT_A_ZERO_LENGTH_RENDER))
def test_the_register_names_something_that_still_refuses(name):
    """So it cannot rot into an excuse for something long since fixed."""
    with pytest.raises(ValueError):
        _callable_with_defaults(name)(duration=0)


def test_an_empty_render_carries_through_the_sequence_operations():
    """Which is the argument for empty over a refusal.

    Zero is the identity and these treat it as one, so a caller building
    notes from computed durations does not have to filter the zeros out
    before stacking them -- and the filtering is the part that goes wrong.
    """
    sound = music.note(440, 0.1)
    nothing = music.note(440, 0)

    assert len(music.horizontal_stack(sound, nothing, sound)) == 2 * len(sound)
    assert len(music.mix(sound, nothing)) == len(sound)
    assert len(music.adsr(sonic_vector=nothing)) == 0


@pytest.mark.parametrize("write,empty", [
    ("write_wav_mono", np.array([])),
    ("write_wav_stereo", np.zeros((2, 0))),
])
def test_writing_nothing_says_there_is_nothing(tmp_path, write, empty):
    """Where an empty sound stops being meaningful, and so where it is
    refused.

    The refusal belongs here rather than in the twenty-four routines that
    can legitimately render nothing: an empty array reaching a file means
    a duration computed as zero somewhere upstream, and this is one call
    from wherever that was. numpy used to report it as a zero-size
    reduction, which names the line rather than the mistake.
    """
    with pytest.raises(ValueError, match="nothing here to normalize"):
        getattr(music, write)(empty, str(tmp_path / "nothing.wav"))


#: Every (routine, string parameter, value) the zero-duration sweep above
#: cannot reach, discovered from the quoted values in each docstring.
#:
#: The sweep varies numbers, so a branch chosen by a string is a branch it
#: never enters -- and `fade` had one. Its linear path handed the sample
#: count straight to `loud`, where zero means "not supplied", so
#: `fade(duration=0, method="linear")` returned two seconds of envelope
#: while `method="exp"`, the default and the only one swept, returned
#: nothing.
#:
#: Discovery rather than a list, so this cannot go stale as routines gain
#: methods; a value the docstring quotes that is not really a method is
#: refused with a ValueError and skipped, which costs nothing.
def _string_branches():
    quoted = re.compile(r"[\"']([a-z_]{2,12})[\"']")
    for name in DURATIONS:
        function = _callable_with_defaults(name)
        documentation = inspect.getdoc(function) or ""
        for parameter, spec in inspect.signature(
                function).parameters.items():
            if not isinstance(spec.default, str):
                continue
            for value in sorted(set(quoted.findall(documentation))
                                | {spec.default}):
                yield name, parameter, value


STRING_BRANCHES = sorted(_string_branches())


def test_the_string_sweep_finds_branches():
    """Guard the discovery, so it cannot quietly find nothing."""
    assert len(STRING_BRANCHES) >= 20
    assert ("fade", "method", "linear") in STRING_BRANCHES


@pytest.mark.parametrize("name,parameter,value", STRING_BRANCHES,
                         ids=lambda value: str(value))
def test_a_zero_duration_renders_nothing_down_every_branch(name, parameter,
                                                           value):
    """The same contract, along the paths a string chooses."""
    function = _callable_with_defaults(name)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rendered = np.asarray(function(duration=0, **{parameter: value}),
                                  dtype=float)
    except ValueError:
        return                      # not a real value for this parameter
    assert rendered.size == 0, (
        f"{name}({parameter}={value!r}, duration=0) rendered "
        f"{rendered.shape}")
