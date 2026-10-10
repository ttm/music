"""FFT and chunk-continuity evidence for opt-in PolyBLEP oscillators."""

import numpy as np
import pytest

from music.core.synths.polyblep import (
    PolyBLEPOscillator, polyblep_frequency_path,
)


@pytest.mark.parametrize("rate", [44100, 48000, 96000])
@pytest.mark.parametrize("waveform", ["sawtooth", "square"])
def test_static_polyblep_reduces_folded_discontinuity_energy(rate, waveform):
    """Coherent 1-s FFT; harmonics above target Nyquist are not permitted."""
    frequency = round(rate * .20835)
    t = np.arange(rate, dtype=float) / rate
    phase = (frequency * t) % 1.0
    if waveform == "sawtooth":
        naive = 2 * phase - 1
        harmonics = range(1, int((rate / 2) // frequency) + 1)
    else:
        naive = np.where(phase < .5, 1., -1.)
        harmonics = range(1, int((rate / 2) // frequency) + 1, 2)
    corrected = polyblep_frequency_path(
        np.full(rate, frequency), sample_rate=rate,
        waveform=waveform)

    def folded_fraction(x):
        power = abs(np.fft.rfft(x)) ** 2
        allowed = np.zeros(len(power), dtype=bool)
        allowed[0] = True
        for harmonic in harmonics:
            allowed[harmonic * frequency] = True
        return float(power[~allowed].sum() / power.sum())

    baseline = folded_fraction(naive)
    polyblep = folded_fraction(corrected)
    assert baseline > .08
    assert polyblep < baseline / 8


@pytest.mark.parametrize("waveform", ["sawtooth", "square"])
def test_chunked_frequency_path_matches_single_block(waveform):
    rate = 48000
    n = 20000
    f = 1000 + 3500 * (
        1 + np.sin(2 * np.pi * 7 * np.arange(n) / rate))
    whole = polyblep_frequency_path(
        f, sample_rate=rate, waveform=waveform,
        phase_cycles=.37)
    stream = PolyBLEPOscillator(
        sample_rate=rate, waveform=waveform,
        phase_cycles=.37)
    blocks = [
        stream.render(f[:1]),
        stream.render([]),
        stream.render(f[1:20]),
        stream.render(f[20:431]),
        stream.render(f[431:]),
    ]
    joined = np.concatenate(blocks)
    np.testing.assert_allclose(joined, whole, atol=1e-8, rtol=0)


def test_noninteger_sample_rate_disallowed_and_phase_config():
    for kw in (
            {"sample_rate": True},
            {"sample_rate": 4000},
            {"sample_rate": 200000},
            {"waveform": "sine"},
            {"phase_cycles": -1},
            {"phase_cycles": 1.0},
            {"phase_cycles": float("nan")},
            {"phase_cycles": "bad"},
    ):
        with pytest.raises(ValueError):
            PolyBLEPOscillator(**kw)


@pytest.mark.parametrize("path", [
    np.zeros((2, 3)), [-1.], [0], [24000], [float("nan")],
    [float("inf")], np.ones(5_000_001),
])
def test_invalid_frequency_paths(path):
    with pytest.raises(ValueError):
        PolyBLEPOscillator(sample_rate=48000).render(path)


def test_chunk_validation_does_not_mutate_phase_or_frequency():
    osc = PolyBLEPOscillator(sample_rate=48000, waveform="square")
    first = osc.render([440, 441])
    with pytest.raises(ValueError):
        osc.render([float("inf")])
    second = osc.render([442])
    fresh = PolyBLEPOscillator(sample_rate=48000, waveform="square")
    expected = fresh.render([440, 441, 442])
    np.testing.assert_allclose(
        np.concatenate((first, second)), expected, atol=1e-10)


def test_polyblep_empty_and_one_sample_are_finite():
    osc = PolyBLEPOscillator()
    assert osc.render([]).shape == (0,)
    assert np.isfinite(osc.render([220])).all()
    assert np.isfinite(osc.render([220])).all()
