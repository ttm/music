"""Glissando endpoints, trills at any rate, refusal advice and defaults.

The last pass of the oscillator mutation audit. The glissando survivors
were edits to the denominator that makes a sweep land on its end
frequency, which a second-long render cannot measure; the trill ones
found an envelope timed at the wrong sample rate.
"""

import re
import warnings

import numpy as np
import pytest

import music

#: A lookup ramp reads the integrated phase itself, to within one cell.
RAMP = np.arange(65536, dtype=float) / 65536


def _ramp_reading(freqs, sample_rate):
    """What the ramp table renders for a path of frequencies."""
    return np.cumsum(freqs) / sample_rate % 1


def _measured_freq(samples, sample_rate):
    """The frequency of a steady tone, from its upward zero crossings."""
    negative = np.signbit(samples)
    crossings = np.where(~negative[1:] & negative[:-1])[0]
    assert len(crossings) > 2, 'too few crossings to measure a frequency'
    return sample_rate / np.mean(np.diff(crossings))


#: The three single sweeps, with every vibrato silenced so that the sweep
#: alone sets the frequency.
GLISSANDI = {
    "note_with_glissando": lambda **settings: music.note_with_glissando(
        waveform_table=RAMP, **settings),
    "note_with_glissando_vibrato": lambda **settings:
        music.note_with_glissando_vibrato(
            max_pitch_dev=0, waveform_table=RAMP, **settings),
    "note_with_two_vibratos_glissando": lambda **settings:
        music.note_with_two_vibratos_glissando(
            max_pitch_dev=0, secondary_max_pitch_dev=0,
            waveform_table=RAMP, **settings),
}


@pytest.mark.parametrize("routine", GLISSANDI)
@pytest.mark.parametrize("alpha", [1, 2, .5])
def test_a_glissando_lands_on_its_end_frequency_at_its_last_sample(routine,
                                                                   alpha):
    """Five samples from 3 to 48 Hz at 64 Hz: 3, 6, 12, 24, 48 when straight.

    The sweep's exponent is ``samples / (n - 1)``, which reaches one on
    the last sample. Edits to that denominator move the end of a
    second-long sweep by a few thousandths of a cent, which no measured
    pitch resolves, so four of them survived on the curved branch. Over
    five samples the same edits move the end by a quarter of the sweep.
    """
    rendered = GLISSANDI[routine](start_freq=3, end_freq=48, alpha=alpha,
                                  number_of_samples=5, sample_rate=64)
    freqs = 3 * 16 ** ((np.arange(5) / 4) ** alpha)
    np.testing.assert_allclose(rendered, _ramp_reading(freqs, 64),
                               rtol=0, atol=1 / len(RAMP))


@pytest.mark.parametrize("routine", GLISSANDI)
@pytest.mark.parametrize("alpha, first", [(1, 3), (2, 3), (0, 48)])
def test_a_one_sample_glissando_sounds_its_starting_frequency(routine, alpha,
                                                              first):
    """A single sample has no interval to divide by.

    ``0 / 0`` made its frequency NaN, the integer cast turned that into
    ``INT64_MIN``, and the sample read whichever table entry that indexed,
    behind a RuntimeWarning. The sequences already answer as here: the
    start, unless an index of zero jumps straight to the end.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        one, two = (GLISSANDI[routine](start_freq=3, end_freq=48,
                                       alpha=alpha, number_of_samples=count,
                                       sample_rate=64)
                    for count in (1, 2))
    np.testing.assert_allclose(one, _ramp_reading([first], 64),
                               rtol=0, atol=1 / len(RAMP))
    np.testing.assert_allclose(two, _ramp_reading([first, 48], 64),
                               rtol=0, atol=1 / len(RAMP))


def test_a_one_sample_linear_glissando_sounds_its_starting_frequency():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        one, two = (music.note_with_glissando(
            start_freq=3, end_freq=48, method="lin", waveform_table=RAMP,
            number_of_samples=count, sample_rate=64) for count in (1, 2))
    np.testing.assert_allclose(one, _ramp_reading([3], 64),
                               rtol=0, atol=1 / len(RAMP))
    np.testing.assert_allclose(two, _ramp_reading([3, 48], 64),
                               rtol=0, atol=1 / len(RAMP))


def test_only_the_glissando_that_can_sweep_linearly_suggests_it():
    """Advice to use ``method="lin"`` is wrong where there is no method.

    Only `note_with_glissando` offers a linear sweep. Nothing checked
    that its refusal says so, or that the others' refusals do not.
    """
    with pytest.raises(ValueError, match=re.escape('Use method="lin"')):
        music.note_with_glissando(start_freq=0, end_freq=440)
    for routine in (music.note_with_glissando_vibrato,
                    music.note_with_two_vibratos_glissando):
        with pytest.raises(ValueError) as refused:
            routine(start_freq=0, end_freq=440)
        assert str(refused.value).endswith("got start_freq=0, end_freq=440")


def _trill_notes(sample_rate):
    """A two-note trill at a rate other than the 44,100 default."""
    rendered = music.trill(freqs=(400, 600), notes_per_second=2, duration=1,
                           sample_rate=sample_rate)
    note_length = sample_rate // 2
    assert len(rendered) == 2 * note_length
    return [(freq, rendered[k * note_length:(k + 1) * note_length])
            for k, freq in enumerate((400, 600))]


def test_a_trill_at_another_rate_sounds_the_frequencies_it_was_given():
    """No test ran a trill other than at 44,100 and measured its pitch."""
    for freq, sounded in _trill_notes(8000):
        # Between the end of the decay at 40 ms and the release.
        assert _measured_freq(sounded[400:-400], 8000) == pytest.approx(
            freq, rel=5e-3)


def test_a_trill_times_its_envelope_in_milliseconds_at_its_own_rate():
    """At 8 kHz the 20 ms attack and decay are 160 samples each, and the
    10 ms release is 80.

    The trill passed its rate to `note` but not to `adsr`, which timed
    the envelope at 44,100 Hz: 882 samples for each ramp and 441 for the
    release, so each 110 ms instead of 20 and 55 instead of 10.
    """
    sustain = 10 ** (-5 / 20)
    for freq, sounded in _trill_notes(8000):
        carrier = music.note(freq, number_of_samples=len(sounded),
                             sample_rate=8000)
        audible = np.abs(carrier) > 0.25
        gain = np.full(len(sounded), np.nan)
        gain[audible] = sounded[audible] / carrier[audible]
        assert np.nanmax(gain[:320]) == pytest.approx(1)
        held = gain[320:-80]
        np.testing.assert_allclose(held[~np.isnan(held)], sustain,
                                   rtol=1e-12)
        assert np.nanmax(gain[-79:]) < sustain


@pytest.mark.parametrize("notes_per_second", [1, .5])
def test_a_trill_may_hold_each_note_for_a_second_or_longer(notes_per_second):
    """The rate must be positive, not faster than one note a second.

    Only zero was refused in a test, so the guard could have read
    ``<= 1`` and refused every slow trill without anything noticing.
    """
    note_length = int(8000 / notes_per_second)
    rendered = music.trill(freqs=(400, 600),
                           notes_per_second=notes_per_second,
                           duration=2 / notes_per_second, sample_rate=8000)
    assert len(rendered) == 2 * note_length
    for k, freq in enumerate((400, 600)):
        sounded = rendered[k * note_length:(k + 1) * note_length]
        assert _measured_freq(sounded[400:-400], 8000) == pytest.approx(
            freq, rel=5e-3)


@pytest.mark.parametrize("routine, declared", [
    ("note", dict(freq=220, duration=2)),
    ("note_with_phase", dict(freq=220, duration=2, phase=0)),
    ("note_with_fm", dict(freq=220, duration=2, fm=100,
                          max_fm_deviation=2)),
    ("note_with_glissando", dict(start_freq=220, end_freq=440, duration=2)),
    ("trill", dict(freqs=(440, 440 * 2 ** (2 / 12)), notes_per_second=17,
                   duration=5)),
])
def test_a_bare_call_renders_the_defaults_it_declares(routine, declared):
    """The same pinning the vibrato routines have: a bare call is the call
    with its declared defaults. Nothing compared the two for these, so
    every default could change without a test noticing."""
    bare = getattr(music, routine)()
    assert np.array_equal(bare, getattr(music, routine)(**declared))
    assert len(bare) == declared["duration"] * 44100
