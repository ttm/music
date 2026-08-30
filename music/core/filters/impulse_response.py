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

    Raises
    ------
    ValueError
        If either argument is not one-dimensional, or if either is
        empty. numpy raises for these anyway, but with messages such as
        "object too deep for desired array", which name neither the
        argument nor the problem.

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

    for name, array in (("samples", samples),
                        ("sonic_vector", sonic_vector)):
        if array.ndim != 1:
            raise ValueError(
                f"fir works on one-dimensional arrays; {name} has shape "
                f"{array.shape}. Filter each channel separately.")
        if array.size == 0:
            raise ValueError(f"{name} is empty")

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

    Returns
    -------
    ndarray
        The filtered signal, the same length as the input.

    Raises
    ------
    ValueError
        If ``sonic_vector`` is not one-dimensional, or if ``b`` is empty
        or begins with zero. A stereo array used to come back as a
        two-element result -- ``len()`` of it counts channels, not
        samples -- and a zero divisor produced an array of infinities
        behind a warning. Filter one channel at a time.

    Notes
    -----
    The recurrence implemented is

    .. math::
        b_0 y[n] = \\sum_k a_k x[n-k] + \\sum_{j \\geq 1} b_j y[n-j]

    Note the plus sign on the feedback sum: this is not the convention
    ``scipy.signal.lfilter`` uses, which subtracts it, and the names of
    the two coefficient arrays are also the other way round there.

    Cost is linear in the length of the signal. It reads only the last
    ``len(a)`` inputs and ``len(b) - 1`` outputs at each step, which is
    all the recurrence refers to; taking a longer slice and discarding
    the remainder -- as this did -- made a second of audio at 44.1 kHz
    take about three seconds, and ten seconds of audio take five
    minutes.

    Check [1] to know more about this function.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    Examples
    --------
    >>> impulse = np.array([1., 0., 0., 0.])
    >>> np.allclose(iir(impulse, [1.], [1., .5]), [1., .5, .25, .125])
    True

    """
    # asarray so the "iterable of scalars" the parameters document really
    # works: two Python lists multiplied together is a TypeError, not an
    # elementwise product.
    signal = np.asarray(sonic_vector, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError(
            f"iir filters one channel at a time; got an array of shape "
            f"{signal.shape}. Pass each channel separately.")
    if b.size == 0:
        raise ValueError("b must hold at least the divisor b[0]")
    if b[0] == 0:
        raise ValueError("b[0] is the divisor of the recurrence and "
                         "cannot be zero")

    out = np.zeros(len(signal))
    for i in range(len(signal)):
        # Only the last len(a) inputs and len(b) - 1 outputs are read, so
        # both slices are bounded by the filter order rather than growing
        # with i. Multiplying and summing, rather than taking a dot
        # product, keeps the arithmetic bit-for-bit what it was: BLAS is
        # free to reassociate, and in a recursive filter that difference
        # compounds.
        first = max(0, i - len(a) + 1)
        forward = (signal[first:i + 1][::-1] * a[:i + 1 - first]).sum()

        first = max(0, i - len(b) + 1)
        feedback = (out[first:i][::-1] * b[1:i + 1 - first]).sum()

        out[i] = (forward + feedback) / b[0]
    return out
