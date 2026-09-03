"""Utilities for reading and writing WAV files."""

import logging
import pathlib
from typing import Any, Sequence

import numpy as np
import soundfile as sf
from numpy.typing import ArrayLike, NDArray
from .functions import normalize_mono, normalize_stereo
from .filters.adsr import adsr, adsr_stereo

#: The encoding written for each container and bit depth. The two do not
#: offer the same depths, and they disagree about 8-bit: WAV stores it
#: unsigned with a midpoint offset, as the RIFF spec requires, while FLAC
#: stores it signed. libsndfile applies whichever the subtype names, so
#: nothing here has to. FLAC has no 32-bit form at all.
FORMAT_SUBTYPES: dict[str, dict[int, str]] = {
    "WAV": {8: "PCM_U8", 16: "PCM_16", 24: "PCM_24", 32: "PCM_32"},
    "FLAC": {8: "PCM_S8", 16: "PCM_16", 24: "PCM_24"},
}

#: The file extensions understood, mapped to their container.
FORMATS: dict[str, str] = {".wav": "WAV", ".flac": "FLAC"}

#: WAV encodings by bit depth. The name predates FLAC support and is kept
#: bound so existing callers work; :data:`FORMAT_SUBTYPES` is the general
#: form.
BIT_DEPTHS: dict[int, str] = FORMAT_SUBTYPES["WAV"]

#: Encodings :func:`read_audio` understands: every depth either container
#: can hold, plus float files, which declare no full scale of their own.
READABLE_SUBTYPES = frozenset(
    subtype
    for depths in FORMAT_SUBTYPES.values()
    for subtype in depths.values()
) | {"FLOAT", "DOUBLE"}


def _audio_format(filename: str) -> str:
    """The container to write `filename` in, from its extension."""
    suffix = pathlib.Path(str(filename)).suffix.lower()
    try:
        return FORMATS[suffix]
    except KeyError:
        allowed = ", ".join(sorted(FORMATS))
        raise ValueError(
            f"unsupported audio file extension {suffix!r}; "
            f"this package writes {allowed}"
        ) from None


def _subtype(bit_depth: int, audio_format: str = "WAV") -> str:
    """The encoding for `bit_depth` in `audio_format`, or raise ValueError."""
    depths = FORMAT_SUBTYPES[audio_format]
    try:
        return depths[bit_depth]
    except KeyError:
        allowed = ", ".join(str(b) for b in depths)
        raise ValueError(
            f"bit_depth values allowed for {audio_format} are only "
            f"{allowed}; got {bit_depth}"
        ) from None


def _wav_subtype(bit_depth: int) -> str:
    """The WAV encoding for ``bit_depth``. The name predates FLAC."""
    return _subtype(bit_depth, "WAV")


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


def read_audio(filename: str) -> NDArray[np.float64]:
    """Read an audio file and return an array of its values.

    Parameters
    ----------
    filename : string
        The path to read. WAV and FLAC are understood; the container is
        whatever the file says it is, not what the extension claims.

    Returns
    -------
    NDArray
        The samples in [-1, 1], mono as ``(nsamples,)`` and stereo as
        ``(2, nsamples)``. Integer PCM is divided by its own full scale;
        a float WAV declares none, so it is scaled by its own peak.

    Raises
    ------
    ValueError
        If the file's encoding is one this package cannot normalize --
        anything outside ``READABLE_SUBTYPES``. libsndfile will decode
        ADPCM and companded formats happily, but they have no full scale
        the normalization is defined against, so being decodable is not
        the same as being supported.
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
        32 for WAV, and 8, 16 or 24 for FLAC, which has no 32-bit form.
        Any other value raises ValueError.
    remove_bias : boolean
        Whether to remove or not the bias (or offset)

    See Also
    --------
    normalize_mono : Normalizes an array to [-1,1]
    write_wav_mono : Writes an array with the same arguments and order of
                     them as soundfile.write.
    write_wav_stereo : Write a stereo file.

    """
    # Reject a bad depth, and a container that cannot hold it, before
    # doing the work of rendering anything.
    audio_format = _audio_format(filename)
    subtype = _subtype(bit_depth, audio_format)
    if sonic_vector is None:
        sonic_vector = np.random.uniform(-1, 1, size=100000)
    result = normalize_mono(sonic_vector, remove_bias)
    if fades:
        f0, f1 = _fade_pair(fades)
        result = adsr(attack_duration=f0, sustain_level=0,
                      release_duration=f1, sonic_vector=result)
    sf.write(str(filename), _quantize(result, bit_depth), sample_rate,
             format=audio_format, subtype=subtype)


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
        32 for WAV, and 8, 16 or 24 for FLAC, which has no 32-bit form.
        Any other value raises ValueError.
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
    audio_format = _audio_format(filename)
    subtype = _subtype(bit_depth, audio_format)
    if sonic_vector is None:
        sonic_vector = np.random.uniform(-1, 1, size=(2, 100000))
    result = normalize_stereo(sonic_vector, remove_bias,
                              normalize_separately)
    if fades:
        f0, f1 = _fade_pair(fades)
        result = adsr_stereo(attack_duration=f0, sustain_level=0,
                             release_duration=f1, sonic_vector=result)
    sf.write(str(filename), _quantize(result, bit_depth).T, sample_rate,
             format=audio_format, subtype=subtype)


#: The name :func:`read_audio` had before it read anything but WAV. Kept
#: bound so existing callers work.
read_wav = read_audio


def write_audio(
    sonic_vector: ArrayLike | None = None,
    filename: str = "asound.wav",
    sample_rate: int = 44100,
    fades: Sequence[int] | int = 0,
    bit_depth: int = 16,
    remove_bias: bool = True,
) -> None:
    """Write a sound to a file, mono or stereo, WAV or FLAC.

    The container comes from the extension and the channel count from
    the array, so a caller who has a sound and a path does not have to
    dispatch on either.

    Parameters
    ----------
    sonic_vector : array_like, optional
        The PCM samples to write: ``(nsamples,)`` for mono or
        ``(2, nsamples)`` for stereo. Defaults to about two seconds of
        uniform noise, as the two writers below do.
    filename : string
        The path to write. ``.wav`` and ``.flac`` are understood.
    sample_rate : scalar
        The sample frequency.
    fades : iterable or scalar
        Milliseconds of fade in and fade out, either as a pair or as a
        single value applied to both ends.
    bit_depth : integer
        Bits per sample: 8, 16, 24 or 32 for WAV, and 8, 16 or 24 for
        FLAC, which has no 32-bit form.
    remove_bias : boolean
        Whether to remove the bias, or offset.

    Returns
    -------
    None
        The sound is written to ``filename``.

    Raises
    ------
    ValueError
        If the extension is not one this package writes, or the bit
        depth is not one the chosen container can hold.

    Notes
    -----
    FLAC is lossless, so it is a smaller file and not a different sound:
    a rendered stimulus read back from FLAC is sample for sample the one
    that was written, which the round-trip tests check at every depth.
    That matters here more than it does in most libraries, because the
    claim this package makes is about the samples.

    Lossy containers are deliberately not offered. Discarding what a
    listener is unlikely to notice is the one thing a package whose
    subject is fidelity to a mathematical model should not do quietly.

    See Also
    --------
    write_wav_mono : the mono writer, with the same arguments.
    write_wav_stereo : the stereo writer, which can also normalize the
                       channels separately.
    read_audio : reads back what these write.

    Examples
    --------
    >>> write_audio(note(duration=1), "note.flac")
    >>> write_audio(localize_linear(note()), "moving.wav")

    """
    if sonic_vector is None:
        write = write_wav_mono
    else:
        write = (write_wav_stereo if np.ndim(sonic_vector) == 2
                 else write_wav_mono)
    write(sonic_vector=sonic_vector, filename=filename,
          sample_rate=sample_rate, fades=fades, bit_depth=bit_depth,
          remove_bias=remove_bias)


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
