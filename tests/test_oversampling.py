"""Check opt-in oversampling against an independent high-rate reference."""

import builtins
import numpy as np
import pytest

import music

scipy = pytest.importorskip("scipy")
from scipy.signal import resample_poly  # noqa: E402


def gated_carrier(*, number_of_samples, sample_rate,
                  freq=15000.0, pulse_rate=500.0, duty=.30):
    """Analytic carrier with a steep amplitude gate."""
    t = np.arange(number_of_samples) / sample_rate
    return np.sin(2 * np.pi * freq * t) * (
        (t * pulse_rate) % 1 < duty)


def test_oversampling_improves_against_high_rate_reference():
    rate, count = 48000, 4800
    reference = resample_poly(
        gated_carrier(number_of_samples=count * 16,
                      sample_rate=rate * 16), 1, 16,
        window=("kaiser", 8.6))
    direct = gated_carrier(number_of_samples=count, sample_rate=rate)
    filtered = music.render_oversampled(
        gated_carrier, sample_rate=rate, number_of_samples=count,
        factor=4)
    direct_error = np.sqrt(np.mean((direct - reference) ** 2))
    filtered_error = np.sqrt(np.mean((filtered - reference) ** 2))
    assert filtered.shape == direct.shape == (count,)
    assert filtered_error < direct_error / 8


def test_generator_works_with_existing_gated_and_fm_producers():
    mono = music.render_oversampled(
        music.isochronic_tones, sample_rate=48000,
        number_of_samples=2000, factor=4, carrier_freq=15000,
        pulse_rate=300)
    fm = music.render_oversampled(
        music.frequency_modulation, sample_rate=48000,
        number_of_samples=2000, factor=4, carrier_freq=15000,
        modulation_freq=500, frequency_deviation=2000)
    binaural = music.render_oversampled(
        music.binaural_beats, sample_rate=48000,
        number_of_samples=2000, carrier_freq=200, beat_freq=10)
    assert mono.shape == fm.shape == (2000,)
    assert binaural.shape == (2, 2000)
    assert all(np.isfinite(x).all() for x in (mono, fm, binaural))


def test_zero_output_skips_generator():
    def not_called(**kwargs):
        raise AssertionError("zero duration should not execute generator")
    assert music.render_oversampled(not_called, duration=0).size == 0


@pytest.mark.parametrize("kwargs,match", [
    ({"sample_rate": 0}, "sample_rate"),
    ({"sample_rate": True}, "sample_rate"),
    ({"factor": 1}, "factor"),
    ({"factor": 17}, "factor"),
    ({"factor": 2.5}, "factor"),
    ({"factor": True}, "factor"),
    ({"number_of_samples": -1}, "number_of_samples"),
    ({"number_of_samples": True}, "number_of_samples"),
    ({"number_of_samples": 50_000_001}, "50 million"),
    ({"duration": -1}, "duration"),
    ({"duration": float("nan")}, "duration"),
    ({"duration": float("inf")}, "duration"),
])
def test_reject_invalid_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        music.render_oversampled(gated_carrier, **kwargs)


def test_reject_wrong_shapes_and_nans():
    with pytest.raises(ValueError, match="renderer must return"):
        music.render_oversampled(
            lambda **kw: np.ones(7), number_of_samples=10)
    with pytest.raises(ValueError, match="non-finite"):
        music.render_oversampled(
            lambda number_of_samples, **kw: np.full(
                number_of_samples, np.nan),
            number_of_samples=10)


def test_scipy_absence_reports_install_instruction(monkeypatch):
    original = builtins.__import__

    def deny_scipy(name, *args, **kwargs):
        if name == "scipy.signal":
            raise ImportError("SciPy intentionally unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_scipy)
    with pytest.raises(ImportError, match="antialias"):
        music.render_oversampled(gated_carrier, number_of_samples=10)


def test_reject_three_dimensional_audio():
    with pytest.raises(ValueError, match="renderer must return"):
        music.render_oversampled(
            lambda number_of_samples, **kw:
            np.ones((1, 1, number_of_samples)), number_of_samples=10)
