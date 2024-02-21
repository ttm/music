import numpy as n

def IIR(sonic_vector, A, B):
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
        A_coeffs = A[:i+1]
        A_contrib = (samples_A*A_coeffs).sum()

        samples_B = signal_[-1:-1-i:-1][:len(B)-1]
        B_coeffs = B[1:i+1]
        B_contrib = (samples_B*B_coeffs).sum()
        t_i = (A_contrib + B_contrib)/B[0]
        signal_.append(t_i)
    return n.array(signal_)
