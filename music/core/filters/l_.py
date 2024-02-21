import numpy as n
from music.core.filters import L

def L_(d=[2,4,2], dev=[5,-10,20], alpha=[1,.5, 20], method=["exp", "exp", "exp"],
        nsamples=0, sonic_vector=0, fs=44100):
    """
    An envelope with linear or exponential transitions of amplitude.

    See L() for more details.

    Parameters
    ----------
    d : iterable
        The durations of the transitions in seconds.
    dev : iterable
        The deviation of the transitions.
        If method="exp" the deviation is in decibels.
        If method="linear" the deviation is an amplitude proportion.
    alpha : iterable
        Indexes to make the transitions slower or faster [1].
        Ignored it method[1]="linear".
    method : iterable
        Methods for each transition.
        "exp" for exponential transitions of amplitude (linear loudness).
        "linear" for linear transitions of amplitude.
    nsamples : interable
        The number of samples of each transition.
        If supplied, d is ignored.
    sonic_vector : array_like
        Samples for the envelope to be applied to.
        If supplied, d or nsamples is used, the final
        sound has the greatest duration of sonic_array
        and d (or nsamples) and missing samples are
        replaced with silence (if sonic_vector is shorter)
        or with a constant value (if d or nsamples yield shorter
        sequences).
    fs : integer
        The sample rate.
        Only used if nsamples and sonic_vector are not supplied.

    Returns
    -------
    E : ndarray
        A numpy array where each value is a value of the envelope
        for the PCM samples.
        If sonic_vector is supplied,
        E is the sonic vector with the envelope applied to it.

    See Also
    --------
    L : An envelope for a loudness transition.
    F : Fade in and out.
    AD : An ADSR envelope.
    T : An oscillation of loudness.

    Examples
    --------
    >>> W(V(d=8)*L_())  # writes a WAV file with a loudness transitions

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
        N = sum(nsamples)
    else:
        N = int(fs*sum(d))
    samples = n.arange(N)
    s = []
    fact = 1
    if nsamples:
        for i, ns in enumerate(nsamples):
            s_ = L(dev[i], alpha[i], nsamples=ns, 
                    method=method[i])*fact
            s.append(s_)
            fact = s_[-1]
    else:
        for i, dur in enumerate(d):
            s_ = L(dur, dev[i], alpha[i],
                    method=method[i], fs=fs)*fact
            s.append(s_)
            fact = s_[-1]
    E = n.hstack(s)
    if type(sonic_vector) in (n.ndarray, list):
        if len(E) < len(sonic_vector):
            s = n.hstack((E, n.ones(len(sonic_vector)-len(E))*E[-1]))
        if len(E) > len(sonic_vector):
            sonic_vector = n.hstack((sonic_vector, n.ones(len(E)-len(sonic_vector))*E[-1]))
        return sonic_vector*E
    else:
        return E
