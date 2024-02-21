import numpy as n
from scipy.io import wavfile as w
from .functions import __n, __ns


monos = n.random.uniform(size=100000) 
def W(sonic_vector=monos, filename="asound.wav", fs=44100,
        fades=0, bit_depth=16, remove_bias=True):
    """
    Write a mono WAV file for a numpy array.
    
    One can also use, for example:
        import sounddevice as S
        S.play(__n(array))
    
    Parameters
    ----------
    sonic_vector : array_like
        The PCM samples to be written as a WAV sound file.
        The samples are always normalized by __n(sonic_vector)
        to have samples between -1 and 1.
    filename : string
        The filename to use for the file to be written.
    fs : scalar
        The sample frequency.
    fades : interable
        An iterable with two values for the milliseconds you
        want for the fade in and out (to avoid clicks).
    bit_depth : integer
        The number of bits in each sample of the final file.
    remove_bias : boolean
        Whether to remove or not the bias (or offset)

    See Also
    --------
    __n : Normalizes an array to [-1,1]
    W_ : Writes an array with the same arguments
    and order of them as scipy.io.wavfile.
    WS ; Write a stereo file.
    
    """
    s = __n(sonic_vector, remove_bias)*(2**(bit_depth-1)-1)
    if fades:
        s = AD(A=fades[0], S=0, R=fades[1], sonic_vector=s)
    if bit_depth not in (8, 16, 32, 64):
        print("bit_depth values allowed are only 8, 16, 32 and 64")
        print("File {} not written".format(filename))
    nn = eval("n.int"+str(bit_depth))
    s = nn(s)
    w.write(filename, fs, s)


stereos = n.vstack((n.random.uniform(size=100000), n.random.uniform(size=100000)))

def WS(sonic_vector=stereos, filename="asound.wav", fs=44100,
        fades=0, bit_depth=16, remove_bias=True, normalize_sep=False):
    """
    Write a stereo WAV files for a numpy array.
    
    Parameters
    ----------
    sonic_vector : array_like
        The PCM samples to be written as a WAV sound file.
        The samples are always normalized by __n(sonic_vector)
        to have samples between -1 and 1 and remove the offset.
        Use array of shape (nchannels, nsamples).
    filename : string
        The filename to use for the file to be written.
    fs : scalar
        The sample frequency.
    fades : interable
        An iterable with two values for the milliseconds you
        want for the fade in and out (to avoid clicks).
    bit_depth : integer
        The number of bits in each sample of the final file.
    remove_bias : boolean
        Whether to remove or not the bias (or offset)
    normalize_sep : boolean
        Set to True if each channel should be normalized
        separately. If False (default), the arrays will be
        rescaled in the same proportion.

    See Also
    --------
    __ns : Normalizes a stereo array to [-1,1]
    W ; Write a mono file.
    
    """
    s = __ns(sonic_vector, remove_bias, normalize_sep)*(2**(bit_depth-1)-1)
    if fades:
        s = ADS(A=fades[0], S=0, R=fades[1], sonic_vector=s)
    if bit_depth not in (8, 16, 32, 64):
        print("bit_depth values allowed are only 8, 16, 32 and 64")
        print("File {} not written".format(filename))
    nn = eval("n.int"+str(bit_depth))
    s = nn(s)
    w.write(filename, fs, s.T)


def W_(fn, fs, sa): 
    """To mimic scipy.io.wavefile input"""
    W(sa, fn, fs=44100)