"""Opt-in mipmapped wavetables for a rich wave whose pitch moves.

Tables of the waveform's first partials, summed from its Fourier series:
one for every count from 1 to 16, then four to an octave up to 1024. At
each sample the oscillator reads the two tables whose counts bracket what
fits below the Nyquist frequency at that sample's pitch, and fades from
one to the other, so that no partial folds and none appears or vanishes
in a step. Where more than 16 partials would fit, it gives up those above
the larger table's count, at most a sixth of them. The reads are linear
between entries, of which a table has at least sixteen to a cycle of its
top partial: next to the edge of a wave of hundreds of partials that is
off by up to 0.004, and what it adds between the harmonics is 80 dB below
the wave. Each sample costs the same at every pitch, as PolyBLEP's does,
and no partial folds, as none of ``bandlimited_frequency_path``'s does.
The tables take 1.75 MiB a waveform, built the first time it is asked
for.

Unlike the original MASS oscillators, this is a separate opt-in engine.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

_MAX_BLOCK_SAMPLES = 5_000_000
#: How many partials each table holds: every count to 16, then four to an
#: octave, to 1024.
PARTIALS = tuple(range(1, 16)) + tuple(
    int(round(16 * 2 ** (step / 4))) for step in range(25))
#: Entries in each table: 4096, or sixteen for each partial where that is
#: more, so that a linear read never has fewer than sixteen to a cycle.
SIZES = tuple(max(4096, 1 << (16 * count - 1).bit_length())
              for count in PARTIALS)
_COUNTS = np.array(PARTIALS, dtype=np.float64)
# From each count to the next; past the last, the last spacing again.
_SPACING = np.diff(np.r_[_COUNTS, 2 * _COUNTS[-1] - _COUNTS[-2]])
_SIZES = np.array(SIZES, dtype=np.intp)
# Where each table starts in one array of them all, each a size and one.
_OFFSETS = np.r_[0, np.cumsum(_SIZES + 1)[:-1]].astype(np.intp)
_TABLES: dict[str, NDArray[np.float64]] = {}


def _coefficient(waveform: str, n: NDArray[np.int64]) -> NDArray[np.float64]:
    """Each partial's sine coefficient, as bandlimited_frequency_path's."""
    if waveform == "sawtooth":
        return 2 * (-1.0) ** (n + 1) / (np.pi * n)
    odd = n % 2 == 1
    if waveform == "square":
        return np.where(odd, 4 / (np.pi * n), 0.0)
    return np.where(odd, 8 * (-1.0) ** ((n - 1) // 2) / (np.pi * n) ** 2,
                    0.0)


def _all_tables(waveform: str) -> NDArray[np.float64]:
    """A waveform's tables end to end, built once, kept and read-only."""
    if waveform not in ("sawtooth", "square", "triangle"):
        raise ValueError("waveform must be sawtooth, square or triangle")
    if waveform not in _TABLES:
        coefficients = _coefficient(waveform, np.arange(1, PARTIALS[-1] + 1))
        flat = np.empty(int(np.sum(_SIZES + 1)))
        for count, size, offset in zip(PARTIALS, SIZES, _OFFSETS):
            spectrum = np.zeros(size // 2 + 1, dtype=complex)
            # A sine's coefficient c at bin n, as irfft scales it.
            spectrum[1:count + 1] = -0.5j * size * coefficients[:count]
            flat[offset:offset + size] = np.fft.irfft(spectrum, size)
            flat[offset + size] = flat[offset]
        flat.setflags(write=False)
        _TABLES[waveform] = flat
    return _TABLES[waveform]


def tables(waveform: str) -> tuple[NDArray[np.float64], ...]:
    """A waveform's tables, one per count, built once and kept.

    Table j holds the first ``PARTIALS[j]`` partials over ``SIZES[j]``
    entries, and one more entry, the first again, so that a read between
    the last entry and the first needs no wrap. The tables are read-only.

    Parameters
    ----------
    waveform : {'sawtooth', 'square', 'triangle'}
        The waveform whose partials the tables hold.

    Returns
    -------
    tuple of ndarray
        ``len(PARTIALS)`` float64 tables, of ``SIZES[j] + 1`` entries.
    """
    flat = _all_tables(waveform)
    return tuple(flat[offset:offset + size + 1]
                 for offset, size in zip(_OFFSETS, SIZES))


def _read(flat: NDArray[np.float64], level: NDArray[np.intp],
          phase: NDArray[np.float64]) -> NDArray[np.float64]:
    """Each sample's table at its phase, linear between two entries."""
    size = _SIZES[level]
    position = phase * size
    # A phase a hair below one can round up to the table's end.
    entry = np.minimum(position.astype(np.intp), size - 1)
    base = _OFFSETS[level] + entry
    here = flat[base]
    return here + (position - entry) * (flat[base + 1] - here)


class MipmapOscillator:
    """Incremental, phase-continuous mipmapped sawtooth, square or triangle.

    The input describes *instantaneous* frequency at each sample. Phase
    is integrated by the trapezoidal rule, corresponding to linear
    interpolation between consecutive frequency samples, as
    :class:`~music.core.synths.polyblep.PolyBLEPOscillator` integrates
    it. The waveforms are those of
    :func:`~music.core.synths.frequency_path.bandlimited_frequency_path`,
    in the same phase: a sawtooth crosses zero rising at the start of its
    cycle. Each block continues the phase the previous one left.

    Where ``f`` is a sample's frequency, the partials below
    ``A = sample_rate / 2 / f`` fit below the Nyquist frequency. If
    ``PARTIALS[j]`` is the largest count not above ``A``, the sample is
    read from tables ``j - 1`` and ``j``, the second weighted by how far
    ``A`` has gone from ``PARTIALS[j]`` towards ``PARTIALS[j + 1]`` and
    the first by the rest. The partials of table ``j - 1`` sound in full
    and table ``j``'s others at the weight, so that from one pitch to the
    next nothing steps. Past the last table, its weight grows over one
    more step of the counts' last spacing, then stays whole.

    Parameters
    ----------
    sample_rate : int
        Samples per second, from 8,000 to 192,000.
    waveform : {'sawtooth', 'square', 'triangle'}
        The waveform read.
    phase_cycles : float
        Where in its cycle the first sample falls, from 0 up to 1.
    """

    def __init__(self, *, sample_rate: int = 44100,
                 waveform: str = "sawtooth",
                 phase_cycles: float = 0.0) -> None:
        if (not isinstance(sample_rate, int) or isinstance(sample_rate, bool)
                or not 8000 <= sample_rate <= 192000):
            raise ValueError("sample_rate must be 8000 to 192000 Hz")
        self._tables = _all_tables(waveform)
        if (not isinstance(phase_cycles, (int, float))
                or not math.isfinite(phase_cycles)
                or not 0 <= phase_cycles < 1):
            raise ValueError("phase_cycles must be finite in [0, 1)")
        self.sample_rate = sample_rate
        self.waveform = waveform
        self._phase = float(phase_cycles)
        self._previous: float | None = None

    def render(self, frequencies: ArrayLike) -> NDArray[np.float64]:
        """Return a mono block, keeping oscillator state for the next one.

        Parameters
        ----------
        frequencies : array_like
            One instantaneous frequency in Hz per sample, each positive,
            finite and below the Nyquist frequency; fewer than five
            million of them. An empty block leaves the state as it was.

        Returns
        -------
        ndarray
            One float64 sample per frequency, continuing the phase the
            previous block left.

        Raises
        ------
        ValueError
            For a block that is not one-dimensional, too long, or holds a
            frequency out of range. The state is left as it was.
        """
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
        steps = np.empty(len(f), dtype=np.float64)
        if self._previous is None:
            steps[0] = 0.0
        else:
            steps[0] = (self._previous + f[0]) / (2 * fs)
        steps[1:] = (f[:-1] + f[1:]) / (2 * fs)
        phase = (self._phase + np.cumsum(steps)) % 1.0

        fits = fs / 2 / f
        upper = np.searchsorted(_COUNTS, fits, side="right") - 1
        lower = np.maximum(upper - 1, 0)
        weight = np.minimum((fits - _COUNTS[upper]) / _SPACING[upper], 1.0)
        result = ((1 - weight) * _read(self._tables, lower, phase)
                  + weight * _read(self._tables, upper, phase))
        self._phase = float(phase[-1])
        self._previous = float(f[-1])
        return result


def mipmap_frequency_path(
        frequencies: ArrayLike, *, sample_rate: int = 44100,
        waveform: str = "sawtooth",
        phase_cycles: float = 0.0) -> NDArray[np.float64]:
    """Render a single block from mipmapped tables.

    For long trajectories, use :class:`MipmapOscillator` across blocks
    instead of allocating the entire path.

    Parameters
    ----------
    frequencies : array_like
        One instantaneous frequency in Hz per sample, each positive,
        finite and below the Nyquist frequency.
    sample_rate : int
        Samples per second, from 8,000 to 192,000.
    waveform : {'sawtooth', 'square', 'triangle'}
        The waveform read.
    phase_cycles : float
        Where in its cycle the first sample falls, from 0 up to 1.

    Returns
    -------
    ndarray
        One float64 sample per frequency.
    """
    return MipmapOscillator(
        sample_rate=sample_rate, waveform=waveform,
        phase_cycles=phase_cycles).render(frequencies)
