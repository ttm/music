import numpy as n
from music.utils import convolve

def FIR(samples, sonic_vector, freq=True, max_freq=True):
    """
    Apply a FIR filter to a sonic_array.

    Parameters
    ----------
    samples : array_like
        A sequence of absolute values for the frequencies
        (if freq=True) or samples of an impulse response.
    sonic_vector : array_like
        An one-dimensional array with the PCM samples of
        the signal (e.g. sound) for the FIR filter
        to be applied to.
    freq : boolean
        Set to True if samples are frequency absolute values
        or False if samples is an impulse response.
        If max_freq=True, the separations between the frequencies
        are fs/(2*N-2).
        If max_freq=False, the separation between the frequencies
        are fs/(2*N-1).
        Where N is the length of the provided samples.
    max_freq : boolean
        Set to true if the last item in the samples is related
        to the Nyquist frequency fs/2.
        Ignored if freq=False.

    Notes
    -----
    If freq=True, the samples are the absolute values of
    the frequency components.
    The phases are set to zero to maintain the phases
    of the components of the original signal.

    """
    if not freq:
        return convolve(samples, sonic_vector)
    if max_freq:
        s = n.hstack(( samples, samples[1:-1][::-1] ))
    else:
        s = n.hstack(( samples, samples[1:][::-1] ))
    return convolve(samples, sonic_vector)
