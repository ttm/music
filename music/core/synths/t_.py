import numpy as n
from music.utils import S, Tr
from music.core.synths import T

def T_(d=[[3,4,5],[2,3,7,4]], fa=[[2,6,20],[5,6.2,21,5]],
        dB=[[10,20,1],[5,7,9,2]], alpha=[[1,1,1],[1,1,1,9]],
            taba=[[S(),S(),S()],[Tr(),Tr(),Tr(),S()]],
        nsamples=0, sonic_vector=0, fs=44100):
    """
    An envelope with multiple tremolos.

    Parameters
    ----------
    d : iterable of iterable of scalars
        the durations of each tremolo.
    fa : iterable of iterable of scalars
        The frequencies of each tremolo.
    dB : iterable of iterable of scalars
        The maximum loudness variation
        of each tremolo.
    alpha : iterable of iterable of scalars
        Indexes for distortion of each tremolo [1].
    taba : iterable of iterable of array_likes
        Tables for lookup for each tremolo.
    nsamples : iterable of iterable of scalars
        The number of samples or each tremolo.
    sonic_vector : array_like
        The sound to which apply the tremolos.
        If supplied, the tremolo lines are
        applied to the sound and missing samples
        are completed by zeros (if sonic_vector
        is smaller then the lengthiest tremolo)
        or ones (is sonic_vector is larger).
    fs : integer
        The sample rate

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
    L_ : An envelope with an arbitrary number of transitions.
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
    for i in range(len(taba)):
        for j in range(i):
            taba[i][j] = n.array(taba[i][j])
    T_ = []
    if nsamples:
        for i, ns in enumerate(nsamples):
            T_.append([])
            for j, ns_ in enumerate(ns):
                s = T(fa=fa[i][j], dB=dB[i][j], alpha=alpha[i][j],
                    taba=taba[i][j], nsamples=ns_)
                T_[-1].append(s)
    else:
        for i, durs in enumerate(d):
            T_.append([])
            for j, dur in enumerate(durs):
                s = T(dur, fa[i][j], dB[i][j], alpha[i][j],
                    taba=taba[i][j])
                T_[-1].append(s)
    amax = 0
    if type(sonic_vector) in (n.ndarray, list):
        amax = len(sonic_vector)
    for i in range(len(T_)):
        T_[i] = n.hstack(T_[i])
        amax = max(amax, len(T_[i]))
    for i in range(len(T_)):
        if len(T_[i]) < amax:
            T_[i] = n.hstack((T_[i], n.ones(amax-len(T_[i]))*T_[i][-1]))
    if type(sonic_vector) in (n.ndarray, list):
        if len(sonic_vector) < amax:
            sonic_vector = n.hstack(( sonic_vector, n.zeros(amax-len(sonic_vector)) ))
        T_.append(sonic_vector)
    s = n.prod(T_, axis=0)
    return s
