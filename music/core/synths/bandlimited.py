"""Opt-in band-limited wavetable oscillator.

The MASS-compatible :func:`music.note` intentionally reads an unfiltered
table. This module leaves its samples unchanged: filtering a table according
to the *fundamental* would alter that reference behavior. The static-pitch
oscillator here removes harmonics above Nyquist before wavetable lookup,
then linearly interpolates rather than truncating lookup positions.

It is not a cure for aliasing caused by later nonlinear processing,
modulation, clipping, or abrupt pulse gates. Dynamic pitch sweeps need
frequency-dependent filtering, not this constant-frequency oscillator.
"""

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from ...utils import waveform_table


@lru_cache(maxsize=128)
def _filtered_table(kind: str, freq: float, sample_rate: int,
                    table_size: int, rolloff_hz: float) -> NDArray[np.float64]:
    """FFT-low-pass one periodic primary waveform, cached by pitch."""
    spectrum = np.fft.rfft(waveform_table(kind, table_size))
    harmonic_hz = np.arange(len(spectrum)) * freq
    nyquist = sample_rate / 2
    if rolloff_hz:
        gain = np.clip((nyquist - harmonic_hz) / rolloff_hz, 0, 1)
    else:
        gain = (harmonic_hz < nyquist).astype(float)
    spectrum *= gain
    spectrum[0] = 0  # remove the sampled sawtooth's tiny DC offset
    table = np.fft.irfft(spectrum, n=table_size)
    table.flags.writeable = False
    return table


def bandlimited_note(
        freq: float = 220.0, duration: float = 2.0,
        waveform: str = "sawtooth", number_of_samples: int = 0,
        sample_rate: int = 44100, table_size: int = 16384,
        rolloff_hz: float = 0.0) -> NDArray[np.float64]:
    """Render a static-pitch periodic note with prefiltered harmonics.

    Parameters
    ----------
    freq : float
        Positive fundamental in Hz, strictly below Nyquist.
    duration : float
        Seconds, ignored when `number_of_samples` is nonzero.
    waveform : {'sine', 'sawtooth', 'square', 'triangle'}
        A primary waveform. Arbitrary arrays are not accepted because
        their spectrum and source-pitch semantics are unknown.
    number_of_samples : int
        Overrides `duration` when positive, matching `music.note`.
    sample_rate : int
        Samples per second.
    table_size : int
        Resolution of the periodic prefiltered table (at least 4).
    rolloff_hz : float
        Optional nonnegative linear transition width below Nyquist.
        The default zero keeps all allowed harmonics. A smooth rolloff
        avoids sudden harmonic changes between adjacent static pitches.

    Returns
    -------
    ndarray
        One-dimensional float64 samples. The method preserves the
        periodic waveform's fundamental, not its original peak level;
        truncated discontinuous waves may exhibit Gibbs overshoot.

    Notes
    -----
    This is **opt-in**: `music.note` remains sample-identical to MASS.
    Band-limiting is frequency-dependent. Do not reuse one filtered
    table for glissandi or FM. Hard-gated isochronic envelopes, clipping,
    nonlinear effects and subsequent resampling can still introduce
    aliasing. Inspect downstream output separately.

    Examples
    --------
    >>> x = bandlimited_note(10000, duration=0.01, waveform="sawtooth")
    >>> x.shape
    (441,)
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not (0 < freq < sample_rate / 2):
        raise ValueError("freq must be positive and below Nyquist")
    if number_of_samples < 0:
        raise ValueError("number_of_samples cannot be negative")
    if not number_of_samples and (not np.isfinite(duration) or duration < 0):
        raise ValueError("duration must be finite and nonnegative")
    if rolloff_hz < 0 or not np.isfinite(rolloff_hz):
        raise ValueError("rolloff_hz must be finite and nonnegative")
    if table_size < 4:
        raise ValueError("table_size must be at least 4")
    count = (int(number_of_samples) if number_of_samples
             else int(duration * sample_rate))
    if not count:
        return np.empty(0, dtype=np.float64)

    table = _filtered_table(waveform, float(freq), sample_rate,
                            table_size, float(rolloff_hz))
    position = (np.remainder(np.arange(count) * freq / sample_rate, 1)
                * table_size)
    index = np.floor(position).astype(np.intp)
    fraction = position - index
    return (table[index] * (1 - fraction)
            + table[(index + 1) % table_size] * fraction)
