import numpy as n
from .l import L
from ...utils import resolveStereo

def F(d=2, out=True, method="exp", dB=-80, alpha=1, perc=1,
        nsamples=0, sonic_vector=0, fs=44100):
    """
    A fade in or out.

    Implements the loudness transition and asserts that it reaches
    zero amplitude.

    Parameters
    ----------
    d : scalar
        The duration in seconds of the fade.
    out : boolean
        If True, the fade is a fade out, else it is a fade in.
    method : string
        "exp" for an exponential transition of amplitude (linear loudness).
        "linear" for a linear transition of amplitude.
    dB : scalar
        The decibels from which to reach before using
        the linear transition to reach zero.
        Not used if method="linear".
    alpha : scalar
        An index to make the exponential fade slower or faster [1].
        Ignored it transitions="linear". 
    perc : scalar
        The percentage of the fade that is linear to assure it reaches zero.
        Has no effect if method="linear".
    nsamples : integer
        The number of samples of the fade. If supplied, d is ignored.
    sonic_vector : array_like
        Samples for the fade to be applied to.
        If supplied, d and nsamples are ignored.
    fs : integer
        The sample rate. Only used if nsamples and sonic_vector are not supplied.

    Returns
    -------
    T : ndarray
        A numpy array where each value is a value of the envelope for the PCM samples.
        If sonic_vector is input, T is the sonic vector with the fade applied to it.

    See Also
    --------
    AD : An ADSR envelope.
    L : A transition of loudness.
    L_ : An envelope with an arbitrary number or loudness transitions.
    T : An oscillation of loudness.

    Examples
    --------
    >>> W(V()*F())  # writes a WAV file with a fade in
    >>> s = H( [V()*F(out=i, method=j) for i, j in zip([1, 0, 1], ["exp", "exp", "linear"])] )  # OR
    >>> s = H( [F(out=i, method=j, sonic_vector=V()) for i, j in zip([1, 0, 1], ["exp", "exp", "linear"])] )
    >>> envelope = F(d=10, out=0, perc=0.1)  # a lengthy fade in 

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if type(sonic_vector) in (n.ndarray, list):
        if len(sonic_vector.shape) == 2:
            return resolveStereo(F, locals())
        N = len(sonic_vector)
    elif nsamples:
        N = nsamples
    else:
        N = int(fs*d)
    if 'lin' in method:
        if out:
            ai = L(method="linear", dev=0, nsamples=N)
        else:
            ai = L(method="linear", to=0, dev=0, nsamples=N)
    if 'exp' in method:
        N0 = int(N*perc/100)
        N1 = N - N0
        if out:
            ai1 = L(dev=dB, alpha=alpha, nsamples=N1)
            if N0:
                ai0 = L(method="linear", dev=0, nsamples=N0)*ai1[-1]
            else:
                ai0 = []
            ai = n.hstack((ai1, ai0))
        else:
            ai1 = L(dev=dB, to=0, alpha=alpha, nsamples=N1)
            if N0:
                ai0 = L(method="linear", to=0, dev=0, nsamples=N0)*ai1[0]
            else:
                ai0 = []
            ai = n.hstack((ai0, ai1))
    if type(sonic_vector) in (n.ndarray, list):
        return ai*sonic_vector
    else:
        return ai
