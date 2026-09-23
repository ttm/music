"""Moving-source audio measured against the article's spatial equations.

eq:distOuvidos, eq:dii and eq:dti give ear distances, inverse-distance
amplitudes and arrival differences. eq:fDoppler gives pitch from radial
velocity; the rendered phase probe below measures that pitch independently
of the oscillator's phase integration.
"""

import numpy as np
import pytest

import music


def test_stationary_default_note_has_two_seconds_of_220_hz_at_44100_hz():
    """No radial velocity leaves only the default pitch and 1/5 gain."""
    table_length = 65536
    table = np.sin(2 * np.pi * np.arange(table_length) / table_length)
    actual = music.note_with_doppler(
        x=(3, 3), y=(4, 4), stereo=False, waveform_table=table,
    )
    assert actual.shape == (88200,)
    times = np.arange(1, 88201) / 44100
    expected = np.sin(2 * np.pi * 220 * times) / 5
    # One wavetable bin bounds the sine error, including phase wrapping.
    np.testing.assert_allclose(actual, expected, rtol=0,
                               atol=2 * np.pi / table_length / 5 + 1e-12)


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("explicit_count", [False, True])
def test_one_sample_keeps_its_doppler_pitch_and_starting_gain(stereo,
                                                            explicit_count):
    """A single interval still travels from (3, 4) to (6, 8)."""
    sample_rate, freq = 16, 3
    table_length = 65536
    table = np.sin(2 * np.pi * np.arange(table_length) / table_length)
    actual = music.note_with_doppler(
        freq=freq, duration=99 if explicit_count else 1 / sample_rate,
        number_of_samples=int(explicit_count), sample_rate=sample_rate,
        x=(3, 6), y=(4, 8), stereo=stereo, waveform_table=table,
    )
    ears = np.array([-0.215 / 2, 0.215 / 2] if stereo else [0])
    initial_distance = np.hypot(3 - ears, 4)
    final_distance = np.hypot(6 - ears, 8)
    velocity = (final_distance - initial_distance) * sample_rate
    received_frequency = freq * 343.42 / (343.42 + velocity)
    expected = np.sin(2 * np.pi * received_frequency / sample_rate)
    expected /= initial_distance
    assert actual.shape == ((2, 1) if stereo else (1,))
    np.testing.assert_allclose(
        actual.reshape(-1), expected, rtol=0,
        atol=2 * np.pi / table_length / initial_distance.min(),
    )


@pytest.mark.parametrize("stereo", [False, True])
def test_zero_duration_contains_no_samples_or_extra_channels(stereo):
    actual = music.note_with_doppler(duration=0, stereo=stereo)
    assert actual.shape == ((2, 0) if stereo else (0,))
    assert actual.dtype == np.float64


@pytest.mark.parametrize("stereo", [False, True])
def test_diagonal_path_obeys_inverse_distance_at_both_ears(stereo):
    """Changing y affects absolute gain as well as left/right balance."""
    count, sample_rate, carrier = 240, 8000, -0.375
    actual = music.note_with_doppler(
        x=(0.5, 0.8), y=(2, 2.6), stereo=stereo,
        waveform_table=[carrier] * 7, number_of_samples=count,
        sample_rate=sample_rate,
    )
    times = np.arange(count) / sample_rate
    x, y = 0.5 + 10 * times, 2 + 20 * times
    if not stereo:
        np.testing.assert_allclose(actual, carrier / np.hypot(x, y),
                                   rtol=1e-13, atol=1e-14)
        return

    initial_distances = np.hypot([0.5 + 0.215 / 2, 0.5 - 0.215 / 2], 2)
    delay = int(np.diff(initial_distances)[0] * -sample_rate / 343.42)
    assert delay > 0
    assert actual.shape == (2, count + delay)
    np.testing.assert_array_equal(actual[0, :delay], 0)
    np.testing.assert_array_equal(actual[1, count:], 0)
    np.testing.assert_allclose(actual[0, delay:],
                               carrier / np.hypot(x + 0.215 / 2, y),
                               rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(actual[1, :count],
                               carrier / np.hypot(x - 0.215 / 2, y),
                               rtol=1e-13, atol=1e-14)


def test_initial_delay_uses_the_side_before_the_first_sample_crossing():
    """A subsonic path can cross the centre during its first interval."""
    count, sample_rate, carrier = 20, 44100, 0.75
    actual = music.note_with_doppler(
        x=(-0.005, 0.115), y=(0.02, 0.04), waveform_table=[carrier] * 4,
        number_of_samples=count, sample_rate=sample_rate,
    )
    speed = np.hypot(0.12, 0.02) / (count / sample_rate)
    assert speed < 343.42
    initial_distances = np.hypot([-0.005 + 0.1075, -0.005 - 0.1075], 0.02)
    delay = int(np.diff(initial_distances)[0] * sample_rate / 343.42)
    assert delay == 1
    assert actual.shape == (2, count + delay)
    assert actual[0, 0] == pytest.approx(carrier / initial_distances[0])
    assert actual[1, 0] == 0
    assert actual[1, delay] == pytest.approx(carrier / initial_distances[1])
    assert actual[0, -1] == 0


@pytest.mark.parametrize("air_temp", [-10, 35])
def test_non_axis_motion_produces_each_ears_own_radial_doppler(air_temp):
    """Recover pitch from rendered phase and compare with v dot r/|r|."""
    sample_rate, count, freq = 16000, 800, 440
    sound_speed = 331.3 + 0.606 * air_temp
    table_length = 2 ** 20
    phase_table = np.arange(table_length) / table_length
    actual = music.note_with_doppler(
        freq=freq, x=(-0.25, 0.25), y=(0.09, 0.21),
        number_of_samples=count, sample_rate=sample_rate,
        waveform_table=phase_table, air_temp=air_temp,
    )
    ears = (-0.215 / 2, 0.215 / 2)
    initial_distances = np.hypot(-0.25 - np.array(ears), 0.09)
    delay = int(np.diff(initial_distances)[0] * sample_rate / sound_speed)
    times = np.arange(count) / sample_rate
    measured_frequencies = []
    for channel, ear in enumerate(ears):
        offset = delay * channel
        samples = actual[channel, offset:offset + count]
        distance = np.hypot(-0.25 + 10 * times - ear, 0.09 + 2.4 * times)
        phase = np.unwrap(samples * distance * 2 * np.pi) / (2 * np.pi)
        measured = np.diff(phase) * sample_rate
        measured_frequencies.append(measured)

        # Midpoint radial velocity approximates the finite sample interval
        # independently of the implementation's distance differencing.
        midpoints = (np.arange(1, count) + 0.5) / sample_rate
        dx = -0.25 + 10 * midpoints - ear
        dy = 0.09 + 2.4 * midpoints
        radial_speed = (10 * dx + 2.4 * dy) / np.hypot(dx, dy)
        expected = freq * sound_speed / (sound_speed + radial_speed)
        # A difference of two table bins contributes at most sr / length;
        # the remaining allowance covers the midpoint approximation.
        np.testing.assert_allclose(measured, expected, rtol=0, atol=0.02)
    assert np.max(np.abs(np.diff(measured_frequencies, axis=0))) > 10


@pytest.mark.parametrize("stereo", [False, True])
def test_explicit_sample_count_sets_movement_time_independently_of_duration(
        stereo):
    params = dict(x=(-0.8, 0.4), y=(1.4, 2.2), stereo=stereo,
                  number_of_samples=1600, sample_rate=8000)
    short = music.note_with_doppler(duration=0.0001, **params)
    long = music.note_with_doppler(duration=99, **params)
    duration_driven = music.note_with_doppler(
        duration=0.2, **{**params, "number_of_samples": 0},
    )
    np.testing.assert_array_equal(short, long)
    np.testing.assert_array_equal(short, duration_driven)


def test_coincident_ears_equal_mono_for_a_non_axis_moving_source():
    params = dict(freq=377, x=(-1.3, 0.7), y=(2.1, -0.4),
                  number_of_samples=2000, sample_rate=8000)
    mono = music.note_with_doppler(stereo=False, **params)
    stereo = music.note_with_doppler(zeta=0, **params)
    np.testing.assert_array_equal(stereo[0], mono)
    np.testing.assert_array_equal(stereo[1], mono)


@pytest.mark.parametrize("stereo", [False, True])
def test_reflecting_y_keeps_distance_velocity_and_the_whole_sound(stereo):
    params = dict(freq=377, x=(-1.3, 0.7), stereo=stereo,
                  number_of_samples=2000, sample_rate=8000)
    in_front = music.note_with_doppler(y=(2.1, 0.4), **params)
    behind = music.note_with_doppler(y=(-2.1, -0.4), **params)
    np.testing.assert_array_equal(in_front, behind)


def test_reflecting_x_exchanges_the_two_complete_ear_signals():
    params = dict(freq=377, y=(2.1, 0.4), number_of_samples=2000,
                  sample_rate=8000)
    first = music.note_with_doppler(x=(-1.3, 0.7), **params)
    reflected = music.note_with_doppler(x=(1.3, -0.7), **params)
    np.testing.assert_array_equal(first, reflected[::-1])
