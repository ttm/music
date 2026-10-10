"""Independent spectral checks for the opt-in band-limited oscillator."""

import numpy as np
import pytest

import music


def _off_harmonic_fraction(signal, fundamental, rate=44100):
    """Spectral energy outside integer multiples of the intended pitch."""
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    good = np.zeros(len(spectrum), dtype=bool)
    for harmonic in range(1, int((rate / 2) // fundamental) + 1):
        bin_index = int(round(harmonic * fundamental * len(signal) / rate))
        good[bin_index] = True
    return float(spectrum[~good].sum() / spectrum.sum())


@pytest.mark.parametrize("waveform", ["sawtooth", "square", "triangle"])
def test_high_pitch_filter_removes_measured_aliases(waveform):
    raw = music.note(10000, duration=1, waveform_table=music.waveform_table(
        waveform))
    filtered = music.bandlimited_note(10000, duration=1, waveform=waveform)
    assert raw.shape == filtered.shape == (44100,)
    assert _off_harmonic_fraction(filtered, 10000) < 1e-6
    assert _off_harmonic_fraction(filtered, 10000) < (
        _off_harmonic_fraction(raw, 10000) / 1000
    )


def test_sine_matches_an_independent_analytic_oscillator():
    duration = .13
    samples = music.bandlimited_note(
        1000, duration=duration, waveform="sine")
    expected = np.sin(2 * np.pi * 1000 * np.arange(len(samples)) / 44100)
    assert np.max(np.abs(samples - expected)) < 1e-5


def test_rolloff_reduces_harmonics_near_nyquist():
    sharp = music.bandlimited_note(
        10000, duration=1, waveform="sawtooth", rolloff_hz=0)
    soft = music.bandlimited_note(
        10000, duration=1, waveform="sawtooth", rolloff_hz=5000)
    a = np.abs(np.fft.rfft(sharp))
    b = np.abs(np.fft.rfft(soft))
    assert b[10000] / a[10000] == pytest.approx(1.0, abs=.002)
    assert b[20000] / a[20000] == pytest.approx(.41, abs=.002)


def test_count_override_and_empty_render():
    samples = music.bandlimited_note(220, duration=99,
                                      number_of_samples=37, table_size=64)
    assert samples.shape == (37,)
    assert np.isfinite(samples).all()
    assert music.bandlimited_note(220, duration=0).shape == (0,)


@pytest.mark.parametrize("args, message", [
    ({"sample_rate": 0}, "sample_rate"),
    ({"freq": 0}, "freq"),
    ({"freq": 22050}, "Nyquist"),
    ({"freq": float("nan")}, "freq"),
    ({"number_of_samples": -1}, "number_of_samples"),
    ({"duration": -1}, "duration"),
    ({"duration": float("inf")}, "duration"),
    ({"rolloff_hz": -1}, "rolloff_hz"),
    ({"rolloff_hz": float("inf")}, "rolloff_hz"),
    ({"table_size": 3}, "table_size"),
    ({"waveform": "unknown"}, "unknown waveform"),
])
def test_invalid_arguments_refused(args, message):
    with pytest.raises(ValueError, match=message):
        music.bandlimited_note(**args)


def test_gibbs_overshoot_is_not_hidden_by_normalization():
    square = music.bandlimited_note(10000, duration=.1, waveform="square")
    assert np.max(np.abs(square)) > 1.0
