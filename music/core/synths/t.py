import numpy as n
from music.utils import S

def T(d=2, fa=2, dB=10, alpha=1, taba=S(), nsamples=0, sonic_vector=0, fs=44100):
    """
    Synthesize a tremolo envelope or apply it to a sound.
    
    Set fa=0 or dB=0 for a constant envelope with value 1.
    A tremolo is an oscillatory pattern of loudness [1].
    
    Parameters
    ----------
    d : scalar
        The duration of the envelope in seconds.
    fa : scalar
        The frequency of the tremolo oscillations in Hertz.
    dB : scalar
        The maximum deviation of loudness in the tremolo in decibels.
    alpha : scalar
        An index to distort the tremolo pattern [1].
    taba : array_like
        The table with the waveform for the tremolo oscillatory pattern.
    nsamples : integer
        The number of samples of the envelope. If supplied, d is ignored.
    sonic_vector : array_like
        Samples for the tremolo to be applied to.
        If supplied, d and nsamples are ignored.
    fs : integer
        The sample rate.

    Returns
    -------
    T : ndarray
        A numpy array where each value is a PCM sample
        of the envelope.
        if sonic_vector is 0.
        If sonic_vector is input,
        T is the sonic vector with the tremolo applied to it.

    See Also
    --------
    V : A musical note with an oscillation of pitch.
    FM : A linear oscillation of fundamental frequency.
    AM : A linear oscillation of amplitude.

    Examples
    --------
    >>> W(V()*T())  # writes a WAV file of a note with tremolo
    >>> s = H( [V()*T(fa=i, dB=j) for i, j in zip([6, 15, 100], [2, 1, 20])] )  # OR
    >>> s = H( [T(fa=i, dB=j, sonic_vector=V()) for i, j in zip([6, 15, 100], [2, 1, 20])] )
    >>> envelope2 = T(440, 1.5, 60)  # a lengthy envelope

    Notes
    -----
    In the MASS framework implementation, for obtaining a sound with a tremolo (or AM),
    the tremolo pattern is considered separately from a synthesis of the sound.

    The vibrato and FM patterns are considering when synthesizing the sound.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """

    taba = n.array(taba)
    if type(sonic_vector) in (n.ndarray, list):
        Lambda = len(sonic_vector)
    elif nsamples:
        Lambda = nsamples
    else:
        Lambda = n.floor(fs*d)
    samples = n.arange(Lambda)

    l = len(taba)
    Gammaa = (samples*fa*l/fs).astype(n.int64)  # indexes for LUT
    # amplitude variation at each sample
    Ta = taba[ Gammaa % l ] 
    if alpha != 1:
        T = 10.**((Ta*dB/20)**alpha)
    else:
        T = 10.**(Ta*dB/20)
    if type(sonic_vector) in (n.ndarray, list):
        return T*sonic_vector
    else:
        return T
    