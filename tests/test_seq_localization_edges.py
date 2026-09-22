"""Boundary regressions for the independent clocks in a localized note."""
import warnings

import numpy as np
import pytest

import music


def _settings(stereo=False):
    """A stationary source a metre away, with readable table indices."""
    return dict(
        freqs=(125, 250), durations=((.008,), (.008,)),
        vibratos_freqs=(), max_pitch_devs=(), alpha=((1,), (1,)),
        x=(0, 0), y=(1, 1), method=("lin",),
        waveform_tables=((np.arange(16, dtype=float) / 16,),),
        stereo=stereo, zeta=0, sample_rate=1000,
    )


@pytest.mark.parametrize("freqs", [
    (0, 250), (-125, 250), (125, 0), (125, -250),
    (125, 250, 0), (125, -250, 125),
])
def test_every_pitch_transition_requires_positive_endpoints(freqs):
    """An invalid later segment must not poison the accumulated phase.

    A negative ratio used to become NaN then a constant table index;
    a zero starting frequency raised ZeroDivisionError instead.
    """
    settings = _settings()
    segments = len(freqs) - 1
    settings.update(
        freqs=freqs, durations=((.004,) * segments, (.008,)),
        alpha=((1,) * segments, (1,)),
        waveform_tables=((settings["waveform_tables"][0][0],) * segments,),
    )
    with pytest.raises(ValueError, match="frequencies must be positive"):
        music.note_with_vibrato_seq_localization(**settings)


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("samples", [1, 2])
def test_short_pitch_transition_plays_its_boundary_then_holds_its_end(
        stereo, index, samples):
    """One or two samples have no interior points on the pitch curve.

    At 125 Hz the first sample advances two cells of a sixteen-cell
    table. The following held 250 Hz advances four cells each time.
    Previously the zero denominator silenced the entire remaining note.
    """
    settings = _settings(stereo)
    settings.update(durations=((samples / 1000,), (.008,)),
                    alpha=((index,), (1,)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rendered = music.note_with_vibrato_seq_localization(**settings)
    # The zero index jumps straight to the endpoint, even at time zero.
    expected = (np.array([4, 8, 12, 0, 4, 8, 12, 0]) if index == 0 else
                np.array([2, 6, 10, 14, 2, 6, 10, 14])) / 16
    if stereo:
        expected = np.vstack((expected, expected))
    np.testing.assert_array_equal(rendered, expected)


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("duration", [0, .0005])
def test_pitch_transition_without_samples_leaves_the_final_frequency(
        stereo, duration):
    """A sub-sample pitch segment takes no time on the sampled clock."""
    settings = _settings(stereo)
    settings["durations"] = ((duration,), (.008,))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rendered = music.note_with_vibrato_seq_localization(**settings)
    expected = np.array([4, 8, 12, 0, 4, 8, 12, 0]) / 16
    if stereo:
        expected = np.vstack((expected, expected))
    np.testing.assert_array_equal(rendered, expected)


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("duration", [0, .0005, -.001])
def test_a_spatial_segment_needs_time_to_define_its_velocity(stereo, duration):
    """An instantaneous move has no finite Doppler velocity.

    Previously these durations reached division by zero or an empty
    gain array, failing with warnings and unrelated NumPy exceptions.
    """
    settings = _settings(stereo)
    settings.update(
        durations=((.008,), (.008,), (duration,)), y=(1, 2),
        vibratos_freqs=((0,),), max_pitch_devs=((0,),),
        alpha=((1,), (1,), (1,)),
        waveform_tables=(settings["waveform_tables"][0], (np.zeros(2),)),
    )
    with pytest.raises(ValueError, match="duration"):
        music.note_with_vibrato_seq_localization(**settings)


@pytest.mark.parametrize("stereo", [False, True])
def test_one_sample_movement_reaches_its_endpoint(stereo):
    """One interval defines velocity and leaves the source at its endpoint."""
    settings = _settings(stereo)
    settings.update(
        durations=((.008,), (.001,)), y=(2, 2.1),
        waveform_tables=((np.full(4, .5),),),
    )
    rendered = music.note_with_vibrato_seq_localization(**settings)
    expected = np.array([.25] + [.5 / 2.1] * 7)
    if stereo:
        expected = np.vstack((expected, expected))
    np.testing.assert_allclose(rendered, expected, rtol=1e-14)


@pytest.mark.parametrize("stereo", [False, True])
def test_vibrato_segments_use_whole_samples_before_switching_tables(stereo):
    """A fractional remainder cannot add a sample to every segment.

    At 1 kHz the first vibrato occupies two samples at 250 Hz, and the
    second five at 62.5 Hz. The eighth sample has the unmodulated 125 Hz
    carrier. Float arange formerly rounded both segments upwards,
    delaying their boundary and extending the whole note to nine samples.
    """
    settings = _settings(stereo)
    settings.update(
        freqs=(125, 125), durations=((.008,), (.0025, .0055), (.008,)),
        vibratos_freqs=((0, 0),), max_pitch_devs=((12, 12),),
        alpha=((1,), (1, 1), (1,)),
        waveform_tables=(settings["waveform_tables"][0],
                         (np.ones(4), -np.ones(4))),
    )
    rendered = music.note_with_vibrato_seq_localization(**settings)
    expected = np.array([4, 8, 9, 10, 11, 12, 13, 15]) / 16
    if stereo:
        expected = np.vstack((expected, expected))
    np.testing.assert_array_equal(rendered, expected)


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("container", [list, tuple, np.array])
def test_waveform_tables_accept_array_likes_and_integer_samples(stereo,
                                                               container):
    """Documented array-like tables support fractional distance gain."""
    settings = _settings(stereo)
    carrier = container([0, 1, 0, -1])
    vibrato = container([1, -1])
    settings.update(
        freqs=(125, 125), durations=((.008,), (.008,), (.008,)),
        vibratos_freqs=((0,),), max_pitch_devs=((12,),),
        alpha=((1,), (1,), (1,)), y=(2, 2),
        waveform_tables=((carrier,), (vibrato,)),
    )
    rendered = music.note_with_vibrato_seq_localization(**settings)
    expected = np.array([.5, 0, -.5, 0, .5, 0, -.5, 0])
    if stereo:
        expected = np.vstack((expected, expected))
    np.testing.assert_array_equal(rendered, expected)
    np.testing.assert_array_equal(carrier, [0, 1, 0, -1])
    np.testing.assert_array_equal(vibrato, [1, -1])
