"""Finite impulse response and related filters."""

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

    Returns
    -------
    ndarray
        The filtered signal, of length len(sonic_vector) + len(kernel) - 1.
        The kernel is symmetric, so the filter is linear phase and the
        output is delayed by half the kernel.

    Notes
    -----
    If freq=True, the samples are the absolute values of the frequency
    components. The phases are set to zero to maintain the phases of the
    components of the original signal.

    A magnitude response is applied by convolving with its inverse
    transform, not with the magnitudes themselves. Convolving with them
    directly -- which this did -- makes a flat response a boxcar average
    rather than the identity it should be.

    Examples
    --------
    >>> signal = np.arange(5.)
    >>> flat = np.ones(5)  # pass every frequency unchanged
    >>> filtered = fir(flat, signal)
    >>> np.allclose(filtered[4:9], signal)
    True

    """
    samples = np.asarray(samples, dtype=np.float64)
    sonic_vector = np.asarray(sonic_vector, dtype=np.float64)
    if not freq:
        return np.convolve(samples, sonic_vector)
    if max_freq:
        spectrum = np.hstack((samples, samples[1:-1][::-1]))
    else:
        spectrum = np.hstack((samples, samples[1:][::-1]))
    # Zero phase, so the transform is real and symmetric; fftshift centres
    # it, which turns the wrap-around into a plain delay.
    kernel = np.fft.fftshift(np.fft.ifft(spectrum).real)
    return np.convolve(kernel, sonic_vector)


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
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    # asarray so the "iterable of scalars" the parameters document really
    # works: two Python lists multiplied together is a TypeError, not an
    # elementwise product.
    signal = np.asarray(sonic_vector, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    signal_: list = []
    for i in range(len(signal)):
        samples_a = signal[i::-1][:len(a)]
        a_coeffs = a[:i + 1]
        a_contrib = (samples_a * a_coeffs).sum()

        samples_b = np.asarray(signal_[-1:-1 - i:-1][:len(b) - 1])
        b_coeffs = b[1:i + 1]
        b_contrib = (samples_b * b_coeffs).sum()
        t_i = (a_contrib + b_contrib) / b[0]
        signal_.append(t_i)
    return np.array(signal_)
