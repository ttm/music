"""Rendered pitch, modulation clocks and timbres of the note sequence."""

import warnings

import numpy as np
import pytest

import music


def _settings():
    return dict(
        freqs=(125, 250), durations=((.008,), (.008,)),
        vibratos_freqs=((0,),), vibratos_max_pitch_devs=((0,),),
        alpha=((1,), (1,)), sample_rate=1000,
        waveform_tables=((np.arange(16, dtype=float) / 16,), (np.zeros(2),)),
    )


@pytest.mark.parametrize("index, ratios", [
    (1, [1, 2, 4, 8, 16]),
    (2, [1, 2 ** .25, 2, 2 ** 2.25, 16]),
    (.5, [1, 4, 2 ** (2 * np.sqrt(2)), 2 ** (2 * np.sqrt(3)), 16]),
])
def test_curved_glide_renders_the_expected_five_pitches(index, ratios):
    """freqExponencial and indiceExponencial, bent by the documented index.

    A lookup ramp exposes integrated phase within one table cell. Starting
    at 3 Hz distinguishes f2 / f1 from f2 * f1; the vibrato index differs
    from the pitch index so the two rows cannot be interchanged.
    """
    table = np.arange(65536, dtype=float) / 65536
    rendered = music.note_with_vibratos_glissandos(
        freqs=(3, 48), durations=((5 / 64,), (5 / 64,)),
        vibratos_freqs=((0,),), vibratos_max_pitch_devs=((0,),),
        alpha=((index,), (1,)), waveform_tables=((table,), (np.zeros(2),)),
        sample_rate=64,
    )
    expected = (3 * np.cumsum(ratios) / 64) % 1
    np.testing.assert_allclose(rendered, expected, rtol=0, atol=1 / len(table))


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("samples", [0, 1, 2])
def test_short_glide_renders_its_boundary_then_holds_the_endpoint(index,
                                                                 samples):
    """A one-sample glide must not corrupt all subsequent held samples."""
    settings = _settings()
    settings.update(durations=((samples / 1000,), (.008,)),
                    alpha=((index,), (1,)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rendered = music.note_with_vibratos_glissandos(**settings)
    expected = (np.array([4, 8, 12, 0, 4, 8, 12, 0])
                if samples == 0 or index == 0 else
                np.array([2, 6, 10, 14, 2, 6, 10, 14])) / 16
    np.testing.assert_array_equal(rendered, expected)


@pytest.mark.parametrize("freqs", [
    (0, 250), (-125, 250), (125, 0), (125, -250),
    (125, 250, 0), (125, -250, 125),
])
def test_each_glide_requires_positive_endpoints(freqs):
    settings = _settings()
    count = len(freqs) - 1
    settings.update(
        freqs=freqs, durations=((.004,) * count, (.008,)),
        alpha=((1,) * count, (1,)),
        waveform_tables=((settings["waveform_tables"][0][0],) * count,
                         (np.zeros(2),)),
    )
    with pytest.raises(ValueError, match="frequencies must be positive"):
        music.note_with_vibratos_glissandos(**settings)


def test_fractional_vibrato_durations_use_the_same_clock_as_pitch():
    """2.5 and 5.5 ms at 1 kHz occupy two and five samples, not nine total."""
    settings = _settings()
    settings.update(
        freqs=(125, 125), durations=((.008,), (.0025, .0055)),
        vibratos_freqs=((0, 0),), vibratos_max_pitch_devs=((12, 12),),
        alpha=((1,), (1, 1)),
        waveform_tables=(settings["waveform_tables"][0],
                         (np.ones(2), -np.ones(2))),
    )
    rendered = music.note_with_vibratos_glissandos(**settings)
    # Two samples at 250 Hz, five at 62.5 Hz, then unmodulated 125 Hz.
    expected = np.array([4, 8, 9, 10, 11, 12, 13, 15]) / 16
    np.testing.assert_array_equal(rendered, expected)


@pytest.mark.parametrize("container", [list, tuple, np.array])
def test_nested_array_like_tables_render_without_modifying_inputs(container):
    carrier = container([0, 1, 0, -1])
    vibrato = container([1, -1])
    settings = _settings()
    settings.update(freqs=(125, 125), vibratos_max_pitch_devs=((12,),),
                    waveform_tables=((carrier,), (vibrato,)))
    rendered = music.note_with_vibratos_glissandos(**settings)
    np.testing.assert_array_equal(rendered, [1, 0, -1, 0, 1, 0, -1, 0])
    np.testing.assert_array_equal(carrier, [0, 1, 0, -1])
    np.testing.assert_array_equal(vibrato, [1, -1])


def test_timbres_switch_at_pitch_boundaries_and_hold_the_last_table():
    """Unequal pitch segments select distinct tables on their own clock."""
    levels = (.25, -.5, .75)
    rendered = music.note_with_vibratos_glissandos(
        freqs=(2, 3, 4, 5), durations=((3 / 64, 5 / 64, 2 / 64), (13 / 64,)),
        vibratos_freqs=((0,),), vibratos_max_pitch_devs=((0,),),
        alpha=((1, 1, 1), (1,)), sample_rate=64,
        waveform_tables=(tuple(np.full(16, level) for level in levels),
                         (np.zeros(2),)),
    )
    np.testing.assert_array_equal(rendered, np.repeat(levels, (3, 5, 5)))


def _frequency(samples, sample_rate):
    crossings = np.flatnonzero((samples[:-1] < 0) & (samples[1:] >= 0))
    assert len(crossings) > 5
    return sample_rate / np.mean(np.diff(crossings))


@pytest.mark.parametrize("index", [0, 1, .5, 2])
def test_two_vibrato_lines_produce_four_measurable_pitches(index):
    """The zero vibrato index retains its undistorted-oscillation sentinel."""
    square = np.array([1., -1.])
    rendered = music.note_with_vibratos_glissandos(
        freqs=(220, 220), durations=((1,), (1,), (1,)),
        vibratos_freqs=((1,), (2,)), vibratos_max_pitch_devs=((6,), (3,)),
        alpha=((1,), (index,), (1,)), sample_rate=8000,
        waveform_tables=((music.WAVEFORM_SINE,), (square,), (square,)),
    )
    primary = .5 ** (index or 1)
    pitches = 220 * 2 ** np.array([primary + .25, primary - .25,
                                   -primary + .25, -primary - .25])
    for segment, pitch in zip(np.split(rendered, 4), pitches):
        assert _frequency(segment, 8000) == pytest.approx(pitch, rel=.002)


def test_finished_vibrato_disappears_and_pitch_holds_its_last_endpoint():
    rendered = music.note_with_vibratos_glissandos(
        freqs=(220, 330, 440), durations=((.125, .125), (.5,), (.75,)),
        vibratos_freqs=((0,), (0,)), vibratos_max_pitch_devs=((12,), (0,)),
        alpha=((1, 1), (1,), (1,)), sample_rate=8000,
        waveform_tables=((music.WAVEFORM_SINE,) * 2, (np.ones(2),),
                         (np.zeros(2),)),
    )
    assert _frequency(rendered[2000:4000], 8000) == pytest.approx(
        880, rel=.002)
    assert _frequency(rendered[4000:], 8000) == pytest.approx(440, rel=.002)
