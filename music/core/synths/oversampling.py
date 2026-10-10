"""Oversample existing synthesis routines, then filter before decimation.

This is opt-in: none of the MASS-compatible synthesis functions change.
Oversampling reduces spectral folding caused by fast modulations, hard gates
and nonlinear *renderers*, but does not undo aliasing in an input waveform
that was already sampled at the target rate.

See `music.bandlimited_note` for static periodic waveforms, which avoids
the oversampling cost. For time-varying FM and isochronic gates, use
`render_oversampled` on the *whole generator* instead of oversampling a
WAV that has already aliased.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def render_oversampled(
        renderer: Callable[..., ArrayLike], *,
        duration: float = 2.0,
        sample_rate: int = 44100,
        number_of_samples: int = 0,
        factor: int = 4,
        **parameters: Any) -> NDArray[np.float64]:
    """Render at a higher sampling rate, low-pass, then downsample.

    The renderer must accept `number_of_samples` and `sample_rate`, as
    `music.note_with_fm`, `music.note_with_glissando`, and
    `music.isochronic_tones` do. It must return samples with time on
    the last axis, either mono `(N,)` or channels-first `(C, N)`.

    Parameters
    ----------
    renderer : callable
        A sound generator. It is called once at the oversampled rate.
        Supply nonlinear operations *inside* it, before decimation.
    duration : float
        Output duration in seconds, used when `number_of_samples=0`.
    sample_rate : int
        Requested output sampling rate in Hz.
    number_of_samples : int
        Exact output sample count, overriding duration when nonzero.
    factor : int
        Oversampling multiplier, an integer from 2 to 16.
    **parameters
        Passed to renderer unchanged, except the sample count and rate.

    Returns
    -------
    ndarray
        Mono or channels-first float64 samples, at exactly the requested
        count. No peak normalization is applied: preserve output levels.

    Raises
    ------
    ValueError
        On invalid sizes, non-finite samples, or wrong renderer shape.
    ImportError
        If SciPy is missing. Install with `pip install 'music[antialias]'`.

    Notes
    -----
    Uses `scipy.signal.resample_poly` and a Kaiser-window low-pass filter.
    Filtering smooths intentional hard edges; alias suppression and an
    infinitely sharp pulse are incompatible at finite sample rate.
    Always benchmark audio quality and CPU/memory costs for your use case.

    Examples
    --------
    >>> from music import isochronic_tones
    >>> signal = render_oversampled(
    ...     isochronic_tones, sample_rate=48000, duration=.02,
    ...     carrier_freq=12000, pulse_rate=100)
    >>> signal.shape
    (960,)
    """
    if (not isinstance(sample_rate, int) or isinstance(sample_rate, bool)
            or sample_rate <= 0):
        raise ValueError("sample_rate must be a positive integer")
    if (not isinstance(factor, int) or isinstance(factor, bool)
            or not 2 <= factor <= 16):
        raise ValueError("factor must be an integer between 2 and 16")
    if (not isinstance(number_of_samples, int)
            or isinstance(number_of_samples, bool)
            or number_of_samples < 0):
        raise ValueError("number_of_samples must be nonnegative integer")
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration must be finite and nonnegative")
    count = (number_of_samples if number_of_samples
             else int(duration * sample_rate))
    if count == 0:
        return np.empty(0, dtype=np.float64)
    if count * factor > 50_000_000:
        raise ValueError("oversampled render exceeds 50 million frames")

    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise ImportError(
            "render_oversampled needs SciPy; "
            "pip install 'music[antialias]'") from exc

    raw = np.asarray(
        renderer(number_of_samples=count * factor,
                 sample_rate=sample_rate * factor, **parameters),
        dtype=np.float64)
    if raw.ndim not in (1, 2) or raw.shape[-1] != count * factor:
        raise ValueError(
            "renderer must return (N,) or (channels, N), with "
            "exactly number_of_samples on the last axis")
    if not np.isfinite(raw).all():
        raise ValueError("renderer returned non-finite audio samples")
    # A narrow transition and strong stopband attenuation reduce folding.
    result = resample_poly(
        raw, up=1, down=factor, axis=-1, window=("kaiser", 8.6))
    return np.asarray(result[..., :count], dtype=np.float64)
