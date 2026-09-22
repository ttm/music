"""Geometry and rendered Doppler pitch in the localized sequence oscillator.

The article's eq:distOuvidos, eq:dii and eq:dti determine ear distances,
relative amplitudes and arrival times; eq:fDoppler determines moving pitch.
A constant, nonunit carrier isolates the absolute 1 / distance gain used by
the moving-source routines without depending on their phase integration.
"""

import numpy as np
import pytest

import music


def _render(*, stereo, x, y, movement_durations, method, curves,
            duration=0.02, sample_rate=4000, carrier=None, air_temp=20):
    if carrier is None:
        carrier = np.full(16, -0.375)
    return music.note_with_vibrato_seq_localization(
        freqs=(440, 440),
        durations=((duration,), (0.007,), movement_durations),
        vibratos_freqs=((0,),), max_pitch_devs=((0,),),
        alpha=((1,), (3,), curves), x=x, y=y, method=method,
        waveform_tables=((carrier,), (np.zeros(16),)),
        stereo=stereo, air_temp=air_temp, sample_rate=sample_rate,
    )


def _geometric_signal(x, y, *, stereo, sample_rate=4000, air_temp=20):
    """Apply inverse distance and the initial arrival delay to a DC probe."""
    if not stereo:
        return -0.375 / np.hypot(x, y)
    ear_positions = (-0.215 / 2, 0.215 / 2)
    distances = np.array([np.hypot(x - ear, y) for ear in ear_positions])
    delay = int((distances[0, 0] - distances[1, 0]) * sample_rate
                / (331.3 + 0.606 * air_temp))
    left, right = -0.375 / distances
    return np.array([
        np.pad(left, (max(delay, 0), max(-delay, 0))),
        np.pad(right, (max(-delay, 0), max(delay, 0))),
    ])


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("curve", [1, 2, 0.5])
def test_exponential_position_uses_its_own_curve_and_next_endpoint(stereo,
                                                                curve):
    """The path's index bends both coordinates independently of vibrato."""
    x, y = (0.6, 2.4, 1.2), (0.8, 3.2, 2.4)
    actual = _render(
        stereo=stereo, x=x, y=y, movement_durations=(0.008, 0.012),
        method=("exp", "exp"), curves=(curve, curve),
    )
    positions = []
    for coordinates in (x, y):
        positions.append(np.concatenate([
            coordinates[i] * (coordinates[i + 1] / coordinates[i])
            ** ((np.arange(count) / count) ** curve)
            for i, count in enumerate((32, 48))
        ]))
    expected = _geometric_signal(*positions, stereo=stereo)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("stereo", [False, True])
def test_linear_position_moves_both_coordinates_through_each_segment(stereo):
    """Each segment uses its own adjacent x/y points and duration."""
    x, y = (0.5, 1.2, 2.0), (2.0, 0.75, 1.5)
    actual = _render(
        stereo=stereo, x=x, y=y, movement_durations=(0.008, 0.012),
        method=("lin", "lin"), curves=(1, 1),
    )
    positions = [np.concatenate([
        np.linspace(coordinates[i], coordinates[i + 1], count,
                    endpoint=False)
        for i, count in enumerate((32, 48))
    ]) for coordinates in (x, y)]
    expected = _geometric_signal(*positions, stereo=stereo)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("stereo", [False, True])
def test_finished_path_holds_the_gain_at_its_final_coordinates(stereo):
    """Once movement ends, a continuing note remains at its destination."""
    x, y = (0.3, 0.6), (0.4, 0.8)
    actual = _render(
        stereo=stereo, x=x, y=y, movement_durations=(0.01,),
        method=("lin",), curves=(1,), duration=0.04, sample_rate=1000,
    )
    positions = [np.concatenate([
        np.linspace(start, end, 10, endpoint=False), np.full(30, end),
    ]) for start, end in (x, y)]
    expected = _geometric_signal(*positions, stereo=stereo, sample_rate=1000)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("x", [-0.35, 0.35])
def test_initial_arrival_delay_favours_the_ear_on_the_sources_side(x):
    """eq:dti applies even when positive x is smaller than one metre."""
    sample_rate, count = 44100, 441
    actual = _render(
        stereo=True, x=(x, x), y=(0.8, 0.8),
        movement_durations=(0.01,), method=("lin",), curves=(1,),
        duration=0.01, sample_rate=sample_rate,
    )
    distances = np.hypot([x + 0.215 / 2, x - 0.215 / 2], 0.8)
    delay = int(abs(distances[0] - distances[1]) * sample_rate / 343.42)
    nearer, farther = (1, 0) if x > 0 else (0, 1)
    assert delay > 0
    assert actual.shape == (2, count + delay)
    np.testing.assert_array_equal(actual[farther, :delay], 0)
    np.testing.assert_array_equal(actual[nearer, count:], 0)
    np.testing.assert_allclose(actual[nearer, :count],
                               -0.375 / distances[nearer])
    np.testing.assert_allclose(actual[farther, delay:],
                               -0.375 / distances[farther])


def _measured_frequency(samples, sample_rate):
    """Measure pitch from interpolated upward zero crossings in the audio."""
    crossings = np.flatnonzero((samples[:-1] < 0) & (samples[1:] >= 0))
    assert len(crossings) > 5, "too few cycles to measure the rendered pitch"
    times = crossings - samples[crossings] / (
        samples[crossings + 1] - samples[crossings])
    return sample_rate / np.mean(np.diff(times))


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("air_temp", [-10, 20])
def test_doppler_tracks_receding_approaching_and_finished_path(stereo,
                                                             air_temp):
    """eq:fDoppler uses radial velocity, then returns to the emitted pitch."""
    sample_rate = 16000
    actual = _render(
        stereo=stereo, x=(20, 30, 25), y=(0, 0, 0),
        movement_durations=(0.2, 0.15), method=("lin", "lin"),
        curves=(1, 1), duration=0.6, sample_rate=sample_rate,
        carrier=music.WAVEFORM_SINE, air_temp=air_temp,
    )
    speed_of_sound = 331.3 + 0.606 * air_temp
    # Along the ear axis, each ear has exactly the same radial velocity.
    windows = [(0.03, 0.17, 50), (0.23, 0.32, -5 / 0.15), (0.4, 0.55, 0)]
    for channel in np.atleast_2d(actual):
        for start, stop, velocity in windows:
            window = channel[int(start * sample_rate):int(stop * sample_rate)]
            measured = _measured_frequency(window, sample_rate)
            expected = 440 * speed_of_sound / (speed_of_sound + velocity)
            assert measured == pytest.approx(expected, rel=2e-4)
