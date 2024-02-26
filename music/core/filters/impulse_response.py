import numpy as np


def fir(samples, sonic_vector, freq=True, max_freq=True):
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
        return np.convolve(samples, sonic_vector)
    if max_freq:
        s = np.hstack((samples, samples[1:-1][::-1]))
    else:
        s = np.hstack((samples, samples[1:][::-1]))
    return np.convolve(samples, sonic_vector)


def iir(sonic_vector, A, B):
    """
    Apply an IIR filter to a signal.

    Parameters
    ----------
    sonic_vector : array_like
        An one dimensional array representing the signal
        (potentially a sound) for the filter to by applied to.
    A : iterable of scalars
        The feedforward coefficients.
    B : iterable of scalars
        The feedback filter coefficients.

    Notes
    -----
    Check [1] to know more about this function.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the
    discrete-time representation of sound."
    arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    signal = sonic_vector
    signal_ = []
    for i in range(len(signal)):
        samples_A = signal[i::-1][:len(A)]
        A_coeffs = A[:i + 1]
        A_contrib = (samples_A * A_coeffs).sum()

        samples_B = signal_[-1:-1 - i:-1][:len(B) - 1]
        B_coeffs = B[1:i + 1]
        B_contrib = (samples_B * B_coeffs).sum()
        t_i = (A_contrib + B_contrib) / B[0]
        signal_.append(t_i)
    return np.array(signal_)
