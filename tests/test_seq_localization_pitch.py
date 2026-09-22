"""Rendered pitch and segment boundaries of the localized note sequence."""

import numpy as np
import pytest

import music


def _stationary(stereo, sample_rate=64):
    """Unit distance at both receivers makes the rendered pitch observable."""
    return dict(x=(0, 0), y=(1, 1), method=("lin",), zeta=0,
                stereo=stereo, sample_rate=sample_rate)


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("index, frequencies", [
    (1, [1, 2, 4, 8, 16]),
    (2, [1, 2 ** .25, 2, 2 ** 2.25, 16]),
    (.5, [1, 4, 2 ** (2 * np.sqrt(2)), 2 ** (2 * np.sqrt(3)), 16]),
])
def test_five_sample_glide_follows_its_pitch_curve(stereo, index, frequencies):
    """Five points on a four-octave glide, measured through a phase ramp.

    The lookup ramp encodes the integrated frequency directly. These
    hand-calculated pitches distinguish a bent glide from an ordinary
    exponential glide even when the vibrato has its own, different index.
    """
    table = np.arange(65536, dtype=float) / 65536
    out = music.note_with_vibrato_seq_localization(
        freqs=(3, 48), durations=((5 / 64,), (5 / 64,), (5 / 64,)),
        vibratos_freqs=((0,),), max_pitch_devs=((0,),),
        alpha=((index,), (1,), (1,)),
        waveform_tables=((table,), (np.zeros(2),)),
        **_stationary(stereo))

    expected = (3 * np.cumsum(frequencies) / 64) % 1
    np.testing.assert_allclose(out, np.broadcast_to(expected, out.shape),
                               rtol=0, atol=1 / len(table))


@pytest.mark.parametrize("stereo", [False, True])
def test_each_carrier_table_starts_at_its_pitch_segment(stereo):
    """Unequal segments use different timbres, then hold the final timbre."""
    lengths = (3, 5, 2)
    levels = (.25, -.5, .75)
    tables = tuple(np.full(16, level) for level in levels)
    out = music.note_with_vibrato_seq_localization(
        freqs=(2, 3, 4, 5),
        durations=(tuple(n / 64 for n in lengths), (13 / 64,), (11 / 64,)),
        vibratos_freqs=((0,),), max_pitch_devs=((0,),),
        alpha=((1, 1, 1), (1,), (1,)),
        waveform_tables=(tables, (np.zeros(2),)),
        **_stationary(stereo))

    expected = np.repeat(levels, (3, 5, 5))
    np.testing.assert_array_equal(out, np.broadcast_to(expected, out.shape))


def _frequency(samples, sample_rate):
    """Measure an audible steady tone, allowing for LUT quantization."""
    crossings = np.flatnonzero((samples[:-1] < 0) & (samples[1:] >= 0))
    assert len(crossings) > 5
    return sample_rate / np.mean(np.diff(crossings))


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("index", [0, 1, .5, 2])
def test_two_vibrato_lines_reach_the_rendered_pitch(stereo, index):
    """A square modulation produces four independently measurable tones.

    One pattern contributes +/- six semitones and the other contributes
    +/- three. Zero is the sequence API's undistorted-index sentinel.
    """
    rate = 8000
    square = np.array([1., -1.])
    out = music.note_with_vibrato_seq_localization(
        freqs=(220, 220), durations=((1,), (1,), (1,), (1,)),
        vibratos_freqs=((1,), (2,)), max_pitch_devs=((6,), (3,)),
        alpha=((1,), (index,), (1,), (1,)),
        waveform_tables=((music.WAVEFORM_SINE,), (square,), (square,)),
        **_stationary(stereo, rate))

    primary = .5 ** (index or 1)
    pitches = 220 * 2 ** np.array([primary + .25, primary - .25,
                                   -primary + .25, -primary - .25])
    for channel in np.atleast_2d(out):
        for segment, pitch in zip(np.split(channel, 4), pitches):
            assert _frequency(segment, rate) == pytest.approx(pitch, rel=.002)


@pytest.mark.parametrize("stereo", [False, True])
def test_finished_pitch_and_vibrato_lines_hold_frequency_and_unity(stereo):
    """A completed vibrato vanishes while a completed glide holds its end."""
    rate = 8000
    out = music.note_with_vibrato_seq_localization(
        freqs=(220, 330, 440), durations=((.125, .125), (.5,), (.75,)),
        vibratos_freqs=((0,),), max_pitch_devs=((12,),),
        alpha=((1, 1), (1,), (1,)),
        waveform_tables=((music.WAVEFORM_SINE,) * 2, (np.ones(2),)),
        **_stationary(stereo, rate))

    for channel in np.atleast_2d(out):
        assert _frequency(channel[2000:4000], rate) == pytest.approx(
            880, rel=.002)
        assert _frequency(channel[4000:], rate) == pytest.approx(440, rel=.002)


def test_default_sample_rate_is_44100():
    kwargs = _stationary(False)
    kwargs.pop("sample_rate")
    out = music.note_with_vibrato_seq_localization(
        freqs=(220, 220), durations=((1,), (1,), (1,)),
        vibratos_freqs=((0,),), max_pitch_devs=((0,),),
        alpha=((1,), (1,), (1,)),
        waveform_tables=((music.WAVEFORM_SINE,), (np.zeros(2),)), **kwargs)
    assert len(out) == 44100
    assert _frequency(out, 44100) == pytest.approx(220, rel=.0001)
