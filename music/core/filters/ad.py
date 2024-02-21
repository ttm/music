import numpy as n
from .f import F
from .l import L

def AD(d=2, A=20, D=20, S=-5, R=50, trans="exp", alpha=1,
        dB=-80, to_zero=1, nsamples=0, sonic_vector=0, fs=44100):
    """
    Synthesize an ADSR envelope.
    
    ADSR (Atack, Decay, Sustain, Release) is a very traditional
    loudness envelope in sound synthesis [1].
    
    Parameters
    ----------
    d : scalar
        The duration of the envelope in seconds.
    A : scalar
        The duration of the Attack in milliseconds.
    D : scalar
        The duration of the Decay in milliseconds.
    S : scalar
        The Sustain level after the Decay in decibels.
        Usually negative.
    R : scalar
        The duration of the Release in milliseconds.
    trans : string
        "exp" for exponential transitions of amplitude 
        (linear loudness).
        "linear" for linear transitions of amplitude.
    alpha : scalar or array_like
        An index to make the exponential fade slower or faster [1].
        Ignored it transitions="linear" or alpha=1.
        If it is an array_like, it should hold three values to be used
        in Attack, Decay and Release.
    dB : scalar or array_like
        The decibels deviation to reach before using a linear fade
        to reach zero amplitude.
        If it is an array_like, it should hold two values,
        one for Attack and another for Release.
        Ignored if trans="linear".
    to_zero : scalar or array_like
        The duration in milliseconds for linearly departing from zero
        in the Attack and reaching the value of zero at the end
        of the Release.
        If it is an array_like, it should hold two values,
        one for Attack and another for Release.
        Is ignored if trans="linear".
    nsamples : integer
        The number of samples of the envelope.
        If supplied, d is ignored.
    sonic_vector : array_like
        Samples for the ADSR envelope to be applied to.
        If supplied, d and nsamples are ignored.
    fs : integer
        The sample rate.

    Returns
    -------
    AD : ndarray
        A numpy array where each value is a value of
        the envelope for the PCM samples if sonic_vector is 0.
        If sonic_vector is input,
        AD is the sonic vector with the ADSR envelope applied to it.

    See Also
    --------
    T : An oscillation of loudness.
    L : A loudness transition.
    F : A fade in or fade out.

    Examples
    --------
    >>> W(V()*AD())  # writes a WAV file of a note with ADSR envelope
    >>> s = H( [V()*AD(A=i, R=j) for i, j in zip([6, 50, 300], [100, 10, 200])] )  # OR
    >>> s = H( [AD(A=i, R=j, sonic_vector=V()) for i, j in zip([6, 15, 100], [2, 2, 20])] )
    >>> envelope = AD(d=440, A=10e3, D=0, R=5e3)  # a lengthy envelope

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if type(sonic_vector) in (n.ndarray, list):
        Lambda = len(sonic_vector)
    elif nsamples:
        Lambda = nsamples
    else:
        Lambda = int(d*fs)
    Lambda_A = int(A*fs*0.001)
    Lambda_D = int(D*fs*0.001)
    Lambda_R = int(R*fs*0.001)

    perc = to_zero/A
    A = F(out=0, method=trans, alpha=alpha, dB=dB, perc=perc, nsamples=Lambda_A)

    D = L(dev=S, method=trans, alpha=alpha, nsamples=Lambda_D)

    a_S = 10**(S/20.)
    S = n.ones( Lambda - (Lambda_A+Lambda_R+Lambda_D) )*a_S

    perc = to_zero/R
    R = F(method=trans, alpha=alpha, dB=dB, perc=perc, nsamples=Lambda_R)*a_S

    AD = n.hstack((A,D,S,R))
    if type(sonic_vector) in (n.ndarray, list):
        return sonic_vector*AD
    else:
        return AD
    