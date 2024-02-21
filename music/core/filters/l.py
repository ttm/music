import numpy as n

def L(d=2, dev=10, alpha=1, to=True, method="exp",
        nsamples=0, sonic_vector=0, fs=44100):
    """
    An envelope for linear or exponential transition of amplitude.

    An exponential transition of loudness yields a linean
    transition of loudness (theoretically).

    Parameters
    ----------
    d : scalar
        The duration of the envelope in seconds.
    dev : scalar
        The deviation of the transition.
        If method="exp" the deviation is in decibels.
        If method="linear" the deviation is an amplitude proportion.
    alpha : scalar
        An index to make the transition slower or faster [1].
        Ignored it method="linear".
    to : boolean
        If True, the transition ends at the deviation.
        If False, the transition starts at the deviation.
    method : string
        "exp" for exponential transitions of amplitude (linear loudness).
        "linear" for linear transitions of amplitude.
    nsamples : integer
        The number of samples of the envelope.
        If supplied, d is ignored.
    sonic_vector : array_like
        Samples for the envelope to be applied to.
        If supplied, d and nsamples are ignored.
    fs : integer
        The sample rate.
        Only used if nsamples and sonic_vector are not supplied.

    Returns
    -------
    E : ndarray
        A numpy array where each value is a value of the envelope 
        for the PCM samples.
        If sonic_vector is supplied,
        ai is the sonic vector with the envelope applied to it.

    See Also
    --------
    L_ : An envelope with an arbitrary number of transitions.
    F : Fade in and out.
    AD : An ADSR envelope.
    T : An oscillation of loudness.

    Examples
    --------
    >>> W(V()*L())  # writes a WAV file of a loudness transition
    >>> s = H( [V()*L(dev=i, method=j) for i, j in zip([6, -50, 2.3], ["exp", "exp", "linear"])] )  # OR
    >>> s = H( [L(dev=i, method=j, sonic_vector=V()) for i, j in zip([6, -50, 2.3], ["exp", "exp", "linear"])] )
    >>> envelope = L(d=10, dev=-80, to=False, alpha=2)  # a lengthy fade in 

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if type(sonic_vector) in (n.ndarray, list):
        N = len(sonic_vector)
    elif nsamples:
        N = nsamples
    else:
        N = int(fs*d)
    samples = n.arange(N)
    N_ = N-1
    if 'lin' in method:
        if to:
            a0 = 1
            al = dev
        else:
            a0 = dev
            al = 1
        E = a0 + (al - a0)*samples/N_
    if 'exp' in method:
        if to:
            if alpha != 1:
                samples_ = (samples/N_)**alpha
            else:
                samples_ = (samples/N_)
        else:
            if alpha != 1:
                samples_ = ( (N_-samples)/N_)**alpha
            else:
                samples_ = ( (N_-samples)/N_)
        E = 10**(samples_*dev/20)
    if type(sonic_vector) in (n.ndarray, list):
        return E*sonic_vector
    else:
        return E
