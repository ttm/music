"""Utilities for reading and writing WAV files."""

import logging
from typing import Any, Sequence

import numpy as np
import soundfile as sf
from numpy.typing import ArrayLike, NDArray
from .functions import normalize_mono, normalize_stereo
from .filters.adsr import adsr, adsr_stereo

#: WAV encoding written at each supported bit depth. 8-bit WAV is unsigned
#: with a midpoint offset, as the RIFF spec requires; libsndfile applies
#: that offset itself, so nothing here has to.
BIT_DEPTHS: dict[int, str] = {
    8: "PCM_U8", 16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}

#: Encodings :func:`read_wav` understands: every depth that can be written,
#: plus float files, which declare no full scale of their own.
READABLE_SUBTYPES = frozenset(BIT_DEPTHS.values()) | {"FLOAT", "DOUBLE"}


def _wav_subtype(bit_depth: int) -> str:
    """Return the WAV encoding for ``bit_depth``, or raise ValueError."""
    try:
        return BIT_DEPTHS[bit_depth]
    except KeyError:
        allowed = ", ".join(str(b) for b in BIT_DEPTHS)
        raise ValueError(
            f"bit_depth values allowed are only {allowed}; got {bit_depth}"
        ) from None


def _fade_pair(fades: Sequence[int] | int) -> tuple[int, int]:
    """Resolve ``fades`` to (fade in, fade out) durations in milliseconds.

    A scalar applies the same duration to both ends.  np.ndim is used rather
    than isinstance because numpy integer scalars do not subclass int.
    """
    if np.ndim(fades) == 0:
        return int(fades), int(fades)  # type: ignore[arg-type]
    return int(fades[0]), int(fades[1])  # type: ignore[index]


def _quantize(samples: NDArray[np.float64], bit_depth: int) -> NDArray[Any]:
    """Quantize samples in [-1, 1] to the integer encoding of a WAV file."""
    _wav_subtype(bit_depth)
    samples = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(samples).all():
        raise ValueError(
            "sonic_vector contains NaN or infinity, which cannot be written "
            "as PCM samples"
        )
    # Scale by 2 ** (bit_depth - 1), matching the divisor read_wav uses, so
    # that writing and reading back is unity gain. Only a sample at exactly
    # +1.0 needs the clip, the positive range being one value shorter.
    #
    # This previously scaled by 2 ** (bit_depth - 1) - 1 while read_wav
    # divided by 2 ** (bit_depth - 1), costing a round trip ~1.5 quantisation
    # steps instead of the half step quantising costs. Fixing the mismatch
    # means files written now differ from older ones by one LSB of gain; see
    # "Needs a decision before release" in CHANGELOG.md. The unity-gain
    # property is pinned by tests/test_fidelity.py.
    full_scale = 2 ** (bit_depth - 1)
    scaled = np.clip(np.round(samples * full_scale), -full_scale,
                     full_scale - 1)
    # libsndfile takes samples in the next integer width up and keeps the
    # high bits, so a sample narrower than its container is shifted into
    # place -- 8-bit into an int16, 24-bit into an int32. It adds the
    # unsigned midpoint 8-bit WAV needs on the way out. Exact both ways.
    if bit_depth in (8, 24):
        scaled = scaled * 256
    return scaled.astype(np.int16 if bit_depth <= 16 else np.int32)


def read_wav(filename: str) -> NDArray[np.float64]:
    """Reads a WAV file and return an array of its values.

    Parameters
    ----------
    filename : string
        File name

    Returns
    -------
    NDArray
        Values of the WAV file
    """
    with sf.SoundFile(str(filename)) as handle:
        subtype = handle.subtype
        logging.debug("read_wav subtype %s", subtype)
        if subtype not in READABLE_SUBTYPES:
            raise ValueError(f"unsupported WAV encoding: {subtype}")
        # libsndfile divides an integer sample by its own full scale on the
        # way to float, which is the normalization this function used to do
        # by hand, and it does it for 24-bit too.
        data = handle.read(dtype="float64")

    if subtype in ("FLOAT", "DOUBLE"):
        # A float WAV declares no full scale, so it is scaled by its own
        # peak instead. An all-zero file has no peak to scale by.
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        data = data / peak if peak else data

    if data.ndim == 2:
        return np.ascontiguousarray(data.T)
    return data


def write_wav_mono(
    sonic_vector: ArrayLike | None = None,
    filename: str = "asound.wav",
    sample_rate: int = 44100,
    fades: Sequence[int] | int = 0,
    bit_depth: int = 16,
    remove_bias: bool = True,
) -> None:
    """Writes a mono WAV file for a numpy array.

    One can also use, for example:
        import sounddevice as S
        S.play(__n(array))

    Parameters
    ----------
    sonic_vector : array_like, optional
        The PCM samples to be written as a WAV sound file. The samples are
        always normalized by normalize_mono(sonic_vector) to have samples
        between -1 and 1. Defaults to about two seconds of uniform noise.
    filename : string
        The filename to use for the file to be written.
    sample_rate : scalar
        The sample frequency.
    fades : iterable or scalar
        Milliseconds of fade in and fade out (to avoid clicks), either as a
        pair or as a single value applied to both ends.
    bit_depth : integer
        The number of bits in each sample of the final file: 8, 16, 24 or
        32. Any other value raises ValueError.
    remove_bias : boolean
        Whether to remove or not the bias (or offset)

    See Also
    --------
    normalize_mono : Normalizes an array to [-1,1]
    write_wav_mono : Writes an array with the same arguments and order of
                     them as soundfile.write.
    write_wav_stereo : Write a stereo file.

    """
    subtype = _wav_subtype(bit_depth)  # reject a bad depth before the work
    if sonic_vector is None:
        sonic_vector = np.random.uniform(-1, 1, size=100000)
    result = normalize_mono(sonic_vector, remove_bias)
    if fades:
        f0, f1 = _fade_pair(fades)
        result = adsr(attack_duration=f0, sustain_level=0,
                      release_duration=f1, sonic_vector=result)
    sf.write(str(filename), _quantize(result, bit_depth), sample_rate,
             subtype=subtype)


def write_wav_stereo(
    sonic_vector: ArrayLike | None = None,
    filename: str = "asound.wav",
    sample_rate: int = 44100,
    fades: Sequence[int] | int = 0,
    bit_depth: int = 16,
    remove_bias: bool = True,
    normalize_separately: bool = False,
) -> None:
    """Write a stereo WAV files for a numpy array.

    Parameters
    ----------
    sonic_vector : array_like, optional
        The PCM samples to be written as a WAV sound file. The samples are
        always normalized by normalize_stereo(sonic_vector) to have samples
        between -1 and 1 and remove the offset.
        Use array of shape (nchannels, nsamples).
        Defaults to about two seconds of uniform noise.
    filename : string
        The filename to use for the file to be written.
    sample_rate : scalar
        The sample frequency.
    fades : iterable or scalar
        Milliseconds of fade in and fade out (to avoid clicks), either as a
        pair or as a single value applied to both ends.
    bit_depth : integer
        The number of bits in each sample of the final file: 8, 16, 24 or
        32. Any other value raises ValueError.
    remove_bias : boolean
        Whether to remove or not the bias (or offset)
    normalize_separately : boolean
        Set to True if each channel should be normalized separately.
        If False (default), the arrays will be rescaled in the same proportion.

    See Also
    --------
    normalize_stereo : Normalizes a stereo array to [-1,1]
    write_wav_mono : Write a mono file.

    """
    subtype = _wav_subtype(bit_depth)  # reject a bad depth before the work
    if sonic_vector is None:
        sonic_vector = np.random.uniform(-1, 1, size=(2, 100000))
    result = normalize_stereo(sonic_vector, remove_bias,
                              normalize_separately)
    if fades:
        f0, f1 = _fade_pair(fades)
        result = adsr_stereo(attack_duration=f0, sustain_level=0,
                             release_duration=f1, sonic_vector=result)
    sf.write(str(filename), _quantize(result, bit_depth).T, sample_rate,
             subtype=subtype)


def play_audio(
    sonic_vector: ArrayLike,
    sample_rate: int = 44100,
    normalize: bool = True,
) -> None:
    """Play a sonic vector using the :mod:`sounddevice` library.

    Parameters
    ----------
    sonic_vector : array_like
        Samples to be played.  Mono arrays should have shape ``(n,)`` and
        stereo arrays ``(2, n)``.
    sample_rate : int, optional
        Playback sample rate.  Defaults to ``44100``.
    normalize : bool, optional
        If ``True`` (default), normalize ``sonic_vector`` before playback using
        :func:`normalize_mono` or :func:`normalize_stereo`.

    Notes
    -----
    If the ``sounddevice`` module is not installed, this function logs a
    warning and returns without playing anything.
    """

    try:
        import sounddevice as sd  # type: ignore
    except Exception:  # pragma: no cover - fallback when sounddevice missing
        logging.warning("sounddevice module not available; cannot play audio")
        return

    data = np.array(sonic_vector, dtype=np.float64)

    if normalize:
        if data.ndim == 1:
            data = normalize_mono(data)
        else:
            data = normalize_stereo(data)

    if data.ndim == 2:
        data = data.T

    sd.play(data, samplerate=sample_rate)
    sd.wait()
