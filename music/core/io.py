"""
This module contains method to write sonic vectors into WAV files.
"""

import numpy as np
from scipy.io import wavfile
from .functions import normalize_mono, normalize_stereo
from .filters import adsr, adsr_stereo

SONIC_VECTOR_MONO = np.random.uniform(size=100000)
SONIC_VECTOR_STEREO = np.vstack((np.random.uniform(size=100000),
                                 np.random.uniform(size=100000)))


def read_wav(filename: str):
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
    s = wavfile.read(filename)
    print(type(s[1] / 2 ** 15))
    if s[1].dtype != 'int16':
        print('implement non 16bit samples!')
        return np.array(None)
    if len(s[1].shape) == 2:
        return np.array(s[1].transpose() / 2 ** 15)
    return s[1] / 2 ** 15


def write_wav_mono(sonic_vector=SONIC_VECTOR_MONO, filename="asound.wav",
                   sample_rate=44100, fades=0, bit_depth=16, remove_bias=True):
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
        result = adsr(attack_duration=fades[0], sustain_level=0,
                      release_duration=fades[1], sonic_vector=result)
    if bit_depth not in (8, 16, 32, 64):
        print("bit_depth values allowed are only 8, 16, 32 and 64")
        print(f"File {filename} not written")
    nn = eval("np.int" + str(bit_depth))
    result = nn(result)
    wavfile.write(filename, sample_rate, result)


def write_wav_stereo(sonic_vector=SONIC_VECTOR_STEREO, filename="asound.wav",
                     sample_rate=44100, fades=0, bit_depth=16,
                     remove_bias=True, normalize_separately=False):
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
        result = adsr_stereo(attack_duration=fades[0], sustain_level=0,
                             release_duration=fades[1], sonic_vector=result)
    if bit_depth not in (8, 16, 32, 64):
        print("bit_depth values allowed are only 8, 16, 32 and 64")
        print(f"File {filename} not written")
    nn = eval("np.int" + str(bit_depth))
    result = nn(result)
    wavfile.write(filename, sample_rate, result.T)
