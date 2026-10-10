"""Opt-in PolyBLEP sawtooth/square oscillators for frequency trajectories.

PolyBLEP smooths *discontinuities* in sawtooth and square oscillators.
It reduces high-frequency foldover without oversampling or the expensive
per-harmonic summation of a truncated Fourier series. It is not a proof
of alias-free arbitrary FM, wavetable processing or nonlinear effects.

Unlike the original MASS oscillators, this is a separate opt-in engine.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

_MAX_BLOCK_SAMPLES = 5_000_000


def _correction(phase: NDArray[np.float64],
                increment: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standard polynomial band-limited step, active near a phase wrap."""
    correction = np.zeros_like(phase)
    left = phase < increment
    if np.any(left):
        x = phase[left] / increment[left]
        correction[left] = 2 * x - x * x - 1
    right = phase > (1 - increment)
    if np.any(right):
        x = (phase[right] - 1) / increment[right]
        correction[right] = x * x + 2 * x + 1
    return correction


class PolyBLEPOscillator:
    """Incremental, phase-continuous sawtooth or square generator.

    The input describes *instantaneous* frequency at each sample. Phase
    is integrated by the trapezoidal rule, corresponding to linear
    interpolation between consecutive frequency samples. Each chunk
    retains the previous frequency and last phase; arbitrary partitions
    of the same trajectory yield approximately identical samples.
    Allocation is proportional to the chunk, not session length.
    This is offline chunked generation, with no hard real-time guarantee.

    Positive finite fundamentals must stay strictly below Nyquist.
    """

    def __init__(self, *, sample_rate: int = 44100,
                 waveform: str = "sawtooth", phase_cycles: float = 0.0):
        if (not isinstance(sample_rate, int) or isinstance(sample_rate, bool)
                or not 8000 <= sample_rate <= 192000):
            raise ValueError("sample_rate must be 8000 to 192000 Hz")
        if waveform not in ("sawtooth", "square"):
            raise ValueError("waveform must be sawtooth or square")
        if (not isinstance(phase_cycles, (int, float))
                or not math.isfinite(phase_cycles)
                or not 0 <= phase_cycles < 1):
            raise ValueError("phase_cycles must be finite in [0, 1)")
        self.sample_rate = sample_rate
        self.waveform = waveform
        self._phase = float(phase_cycles)
        self._previous: float | None = None

    def render(self, frequencies: ArrayLike) -> NDArray[np.float64]:
        """Return a mono block, keeping oscillator state for the next one."""
        f = np.asarray(frequencies, dtype=np.float64)
        if f.ndim != 1:
            raise ValueError("frequencies must be one-dimensional")
        if len(f) > _MAX_BLOCK_SAMPLES:
            raise ValueError("frequency block exceeds 5 million frames")
        if (not np.isfinite(f).all() or np.any(f <= 0)
                or np.any(f >= self.sample_rate / 2)):
            raise ValueError("frequencies must be finite, positive, < Nyquist")
        if not len(f):
            return np.empty(0, dtype=np.float64)
        fs = self.sample_rate
        # Use the current frequency for the PolyBLEP edge width, while
        # integrating between adjacent instantaneous-frequency samples.
        steps = np.empty(len(f), dtype=np.float64)
        if self._previous is None:
            steps[0] = 0.0
        else:
            steps[0] = (self._previous + f[0]) / (2 * fs)
        steps[1:] = (f[:-1] + f[1:]) / (2 * fs)
        phase = (self._phase + np.cumsum(steps)) % 1.0
        increment = f / fs
        if self.waveform == "sawtooth":
            result = 2 * phase - 1 - _correction(phase, increment)
        else:
            result = (np.where(phase < .5, 1.0, -1.0)
                      + _correction(phase, increment)
                      - _correction((phase + .5) % 1.0, increment))
        self._phase = float(phase[-1])
        self._previous = float(f[-1])
        return result


def polyblep_frequency_path(
        frequencies: ArrayLike, *, sample_rate: int = 44100,
        waveform: str = "sawtooth",
        phase_cycles: float = 0.0) -> NDArray[np.float64]:
    """Render a single block of anti-aliased discontinuous-waveform audio.

    Uses phase interpolation and a polynomial step correction. For long
    trajectories, use :class:`PolyBLEPOscillator` across blocks instead
    of allocating the entire path. Arbitrary nonlinear processing may
    reintroduce aliases; inspect the rendered signal independently.
    """
    return PolyBLEPOscillator(
        sample_rate=sample_rate, waveform=waveform,
        phase_cycles=phase_cycles).render(frequencies)
