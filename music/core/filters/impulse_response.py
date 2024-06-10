import numpy as np


def fir(samples, sonic_vector, freq=True, max_freq=True):
    """
    Apply a FIR filter to a sonic_array.

    Parameters
    ----------
    samples : array_like
        A sequence of absolute values for the frequencies (if freq=True) or
        samples of an impulse response.
    sonic_vector : array_like
        An one-dimensional array with the PCM samples of the signal (e.g.
        sound) for the FIR filter to be applied to.
    freq : boolean
        Set to True if samples holds frequency amplitude absolute values or
        False if samples is an impulse response. If max_freq=True, the
        separations between the frequencies are: fs / (2 * N - 2).
        If max_freq=False, the separation between the frequencies are
        fs / (2 * N - 1). Where N is the length of the provided samples.
    max_freq : boolean
        Set to True if the last item in the samples is related to the Nyquist
        frequency fs / 2. Ignored if freq=False.

    Notes
    -----
    If freq=True, the samples are the absolute values of the frequency
    components. The phases are set to zero to maintain the phases of the
    components of the original signal.

    """
    if not freq:
        return np.convolve(samples, sonic_vector)
    if max_freq:
        s = np.hstack((samples, samples[1:-1][::-1]))
    else:
        s = np.hstack((samples, samples[1:][::-1]))
    return np.convolve(s, sonic_vector)


def iir(sonic_vector, a, b):
    """
    Apply an IIR filter to a signal.

    Parameters
    ----------
    sonic_vector : array_like
        An one-dimensional array representing the signal (potentially a sound)
        for the filter to by applied to.
    a : iterable of scalars
        The feedforward coefficients.
    b : iterable of scalars
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
        samples_a = signal[i::-1][:len(a)]
        a_coeffs = a[:i + 1]
        a_contrib = (samples_a * a_coeffs).sum()

        samples_b = signal_[-1:-1 - i:-1][:len(b) - 1]
        b_coeffs = b[1:i + 1]
        b_contrib = (samples_b * b_coeffs).sum()
        t_i = (a_contrib + b_contrib) / b[0]
        signal_.append(t_i)
    return np.array(signal_)
