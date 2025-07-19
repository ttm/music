"""Utilities for reading and writing WAV files."""

import logging
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.io import wavfile
from .functions import normalize_mono, normalize_stereo
from .filters import adsr, adsr_stereo

SONIC_VECTOR_MONO = np.random.uniform(size=100000)
SONIC_VECTOR_STEREO = np.vstack((np.random.uniform(size=100000),
                                 np.random.uniform(size=100000)))


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
    sample_rate, data = wavfile.read(filename)
    logging.debug("read_wav dtype %s", data.dtype)

    if np.issubdtype(data.dtype, np.integer):
        bits = np.iinfo(data.dtype).bits
        if bits not in (8, 16, 32):
            raise ValueError(
                f"unsupported integer WAV bit depth: {bits}"
            )
        norm = float(2 ** (bits - 1))
    elif np.issubdtype(data.dtype, np.floating):
        norm = float(np.max(np.abs(data)))
        if norm == 0:
            norm = 1.0
    else:
        raise ValueError(f"unsupported WAV data type: {data.dtype}")

    if data.ndim == 2:
        return data.astype(np.float64).T / norm
    return data.astype(np.float64) / norm


def write_wav_mono(
    sonic_vector: ArrayLike = SONIC_VECTOR_MONO,
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
    sonic_vector : array_like
        The PCM samples to be written as a WAV sound file. The samples are
        always normalized by normalize_mono(sonic_vector) to have samples
        between -1 and 1.
    filename : string
        The filename to use for the file to be written.
    sample_rate : scalar
        The sample frequency.
    fades : interable
        An iterable with two values for the milliseconds you want for the fade
        in and out (to avoid clicks).
    bit_depth : integer
        The number of bits in each sample of the final file.
    remove_bias : boolean
        Whether to remove or not the bias (or offset)

    See Also
    --------
    normalize_mono : Normalizes an array to [-1,1]
    write_wav_mono : Writes an array with the same arguments and order of them
                     as scipy.io.wavfile.
    write_wav_stereo : Write a stereo file.

    """
    result = normalize_mono(sonic_vector, remove_bias) * \
        (2 ** (bit_depth - 1) - 1)
    if fades:
        f0, f1 = (fades[0], fades[1]) if isinstance(fades, Sequence) else (0, 0)
        result = adsr(attack_duration=f0, sustain_level=0,
                      release_duration=f1, sonic_vector=result)
    if bit_depth not in (8, 16, 32, 64):
        raise ValueError(
            "bit_depth values allowed are only 8, 16, 32 and 64"
        )
    nn = eval("np.int" + str(bit_depth))
    result = nn(result)
    wavfile.write(filename, sample_rate, result)


def write_wav_stereo(
    sonic_vector: ArrayLike = SONIC_VECTOR_STEREO,
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
    sonic_vector : array_like
        The PCM samples to be written as a WAV sound file. The samples are
        always normalized by normalize_stereo(sonic_vector) to have samples
        between -1 and 1 and remove the offset.
        Use array of shape (nchannels, nsamples).
    filename : string
        The filename to use for the file to be written.
    sample_rate : scalar
        The sample frequency.
    fades : interable
        An iterable with two values for the milliseconds you want for the fade
        in and out (to avoid clicks).
    bit_depth : integer
        The number of bits in each sample of the final file.
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
    result = normalize_stereo(sonic_vector, remove_bias,
                              normalize_separately) * (2 **
                                                       (bit_depth - 1) - 1)
    if fades:
        f0, f1 = (fades[0], fades[1]) if isinstance(fades, Sequence) else (0, 0)
        result = adsr_stereo(attack_duration=f0, sustain_level=0,
                             release_duration=f1, sonic_vector=result)
    if bit_depth not in (8, 16, 32, 64):
        raise ValueError(
            "bit_depth values allowed are only 8, 16, 32 and 64"
        )
    nn = eval("np.int" + str(bit_depth))
    result = nn(result)
    wavfile.write(filename, sample_rate, result.T)
