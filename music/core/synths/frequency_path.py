"""Frequency-aware additive synthesis for arbitrary positive pitch paths.

Unlike a static prefiltered wavetable, this method recalculates each partial's
Nyquist gain at *every sample*. An integrated fundamental phase keeps the
partials coherent through continuous pitch changes. This is opt-in, does not
alter the MASS reference implementation, and does not claim to eliminate
sidebands caused by rapid modulation or abrupt changes in the pitch path.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def bandlimited_frequency_path(
        frequencies: ArrayLike, *, sample_rate: int = 44100,
        waveform: str = "sawtooth", transition: float = 0.15,
        max_harmonics: int = 256) -> NDArray[np.float64]:
    """Synthesize a variable-pitch sine, sawtooth, square or triangle.

    Parameters
    ----------
    frequencies : array_like
        One finite, positive instantaneous fundamental frequency in Hz
        per output sample. Every value must be strictly below Nyquist.
    sample_rate : int
        Output sample rate in Hz.
    waveform : {'sine', 'sawtooth', 'square', 'triangle'}
        The periodic harmonic series used for synthesis.
    transition : float
        Fraction of Nyquist (0 to 1) used to fade each harmonic to zero
        at Nyquist. A nonzero fade avoids clicks when a partial disappears
        during a continuous glissando.
    max_harmonics : int
        Computation limit, between 1 and 2048; partials above this value
        are intentionally omitted, including at low fundamentals.

    Returns
    -------
    ndarray
        Mono float64 audio, preserving its fundamental phase across
        changes in frequency; no peak normalization is performed.

    Raises
    ------
    ValueError
        For unsupported waveforms, invalid frequency paths, or an
        excessive sample-by-harmonic computation.

    Notes
    -----
    The algorithm integrates phase on the output time grid and changes
    harmonic gains continuously near Nyquist, reducing high-pitch
    wavetable folding. Fast FM still produces spectral sidebands that
    are not bounded by instantaneous harmonic frequencies. For steep
    modulations, hard gates and nonlinear effects also use
    :func:`music.render_oversampled` around the complete generator.

    Examples
    --------
    >>> hz = np.linspace(1000, 10000, 480, dtype=float)
    >>> bandlimited_frequency_path(hz, sample_rate=48000).shape
    (480,)
    """
    fs = sample_rate
    if not isinstance(fs, int) or isinstance(fs, bool) or fs <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if waveform not in ("sine", "sawtooth", "square", "triangle"):
        raise ValueError("unsupported waveform")
    if not np.isfinite(transition) or not 0 <= transition <= 1:
        raise ValueError("transition must be between 0 and 1")
    if (not isinstance(max_harmonics, int)
            or isinstance(max_harmonics, bool)
            or not 1 <= max_harmonics <= 2048):
        raise ValueError("max_harmonics must be in [1, 2048]")
    f = np.asarray(frequencies, dtype=np.float64)
    if f.ndim != 1:
        raise ValueError("frequencies must be a 1D array")
    if not np.isfinite(f).all() or np.any((f <= 0) | (f >= fs / 2)):
        raise ValueError("all frequencies must be finite and inside Nyquist")
    if not len(f):
        return np.empty(0, dtype=np.float64)
    highest = min(max_harmonics, int(np.floor((fs / 2) / f.min())))
    if len(f) * highest > 50_000_000:
        raise ValueError("pitch path exceeds 50 million harmonic-samples")

    phase = (2 * np.pi / fs) * np.cumsum(
        np.r_[0.0, f[:-1]], dtype=np.float64)
    output = np.zeros_like(f)
    nyquist = fs / 2
    floor = nyquist * (1 - transition)

    for harmonic in range(1, highest + 1):
        if waveform == "sine" and harmonic != 1:
            break
        if waveform in ("square", "triangle") and harmonic % 2 == 0:
            continue
        if waveform == "sine":
            coefficient = 1.0
        elif waveform == "sawtooth":
            coefficient = 2 * (-1)**(harmonic + 1) / (np.pi * harmonic)
        elif waveform == "square":
            coefficient = 4 / (np.pi * harmonic)
        else:
            coefficient = (8 * (-1)**((harmonic - 1) // 2)
                           / (np.pi**2 * harmonic**2))
        partial_hz = harmonic * f
        if transition:
            weights = np.clip((nyquist - partial_hz)
                              / (nyquist - floor), 0.0, 1.0)
        else:
            weights = (partial_hz < nyquist).astype(np.float64)
        if np.any(weights):
            output += coefficient * weights * np.sin(harmonic * phase)
    return output
