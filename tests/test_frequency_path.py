"""Validate harmonic-limited synthesis against analytic and FFT references."""

import numpy as np
import pytest

from music.core.synths.frequency_path import bandlimited_frequency_path


@pytest.mark.parametrize("sample_rate", [44100, 48000, 96000])
def test_sine_phase_is_integrated_at_the_requested_rate(sample_rate):
    count = sample_rate // 10
    f = 1000 + 100 * np.sin(2 * np.pi * np.arange(count) / count)
    result = bandlimited_frequency_path(
        f, sample_rate=sample_rate, waveform="sine")
    phase = np.cumsum(np.r_[0., f[:-1]]) * 2 * np.pi / sample_rate
    np.testing.assert_allclose(result, np.sin(phase), atol=1e-12)


@pytest.mark.parametrize("waveform", ["sawtooth", "square", "triangle"])
def test_static_high_pitch_contains_no_unwanted_fft_bins(waveform):
    f = np.full(48000, 10000.)
    audio = bandlimited_frequency_path(
        f, sample_rate=48000, waveform=waveform, transition=0)
    power = np.abs(np.fft.rfft(audio))**2
    bins = np.arange(len(power))
    allowed = (bins == 10000) | (bins == 20000)
    assert power[~allowed].sum() / power.sum() < 1e-9
    if waveform != "sawtooth":
        assert power[20000] / power.sum() < 1e-9


def test_crossing_a_harmonic_threshold_is_smoothly_weighted():
    f = np.linspace(4500, 5500, 48000)
    a = bandlimited_frequency_path(f, sample_rate=48000,
                                   waveform="square", transition=.2)
    assert a.shape == (48000,)
    assert np.isfinite(a).all()
    # A harmonic crosses the low-pass transition at 4.5 kHz.
    assert np.max(np.abs(np.diff(a))) < 2


@pytest.mark.parametrize("kwargs,match", [
    ({"sample_rate": 0}, "sample_rate"),
    ({"sample_rate": True}, "sample_rate"),
    ({"waveform": "white"}, "waveform"),
    ({"transition": -0.1}, "transition"),
    ({"transition": 1.1}, "transition"),
    ({"transition": float("nan")}, "transition"),
    ({"max_harmonics": 0}, "max_harmonics"),
    ({"max_harmonics": True}, "max_harmonics"),
    ({"max_harmonics": 3000}, "max_harmonics"),
])
def test_reject_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        bandlimited_frequency_path([100.0], **kwargs)


@pytest.mark.parametrize("values", [
    [[100., 200.]], [0], [-100], [float("nan")],
    [float("inf")], [22050.],
])
def test_invalid_frequency_path(values):
    with pytest.raises(ValueError, match="frequencies|Nyquist"):
        bandlimited_frequency_path(values)


def test_empty_path_and_work_bound():
    assert bandlimited_frequency_path([]).shape == (0,)
    with pytest.raises(ValueError, match="50 million"):
        bandlimited_frequency_path(
            np.full(200000, 30.), max_harmonics=256)


def test_frequency_path_excludes_rich_waveform_even_harmonics():
    f = np.full(48000, 1000.)
    square = bandlimited_frequency_path(
        f, sample_rate=48000, waveform="square")
    triangle = bandlimited_frequency_path(
        f, sample_rate=48000, waveform="triangle")
    assert np.max(np.abs(square - triangle)) > .05
    for x in (square, triangle):
        power = np.abs(np.fft.rfft(x))
        assert power[2000] < power[1000] * 1e-7
