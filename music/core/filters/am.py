import numpy as n
from music.utils import S

def AM(d=2, fm=50, a=.4, taba=S(), nsamples=0, sonic_vector=0, fs=44100):
    """
    Synthesize an AM envelope or apply it to a sound.
    
    Set fm=0 or a=0 for a constant envelope with value 1.
    An AM is a linear oscillatory pattern of amplitude [1].
    
    Parameters
    ----------
    d : scalar
        The duration of the envelope in seconds.
    fm : scalar
        The frequency of the modultar in Hertz.
    a : scalar in [0,1]
        The maximum deviation of amplitude of the AM.
    tabm : array_like
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
        T is the sonic vector with the AM applied to it.

    See Also
    --------
    V : A musical note with an oscillation of pitch.
    FM : A linear oscillation of fundamental frequency.
    T : A tremolo, an oscillation of loudness.

    Examples
    --------
    >>> W(V()*AM())  # writes a WAV file of a note with tremolo
    >>> s = H( [V()*AM(fm=i, a=j) for i, j in zip([60, 150, 100], [2, 1, 20])] )  # OR
    >>> s = H( [AM(fm=i, a=j, sonic_vector=V()) for i, j in zip([60, 150, 100], [2, 1, 20])] )
    >>> envelope2 = AM(440, 150, 60)  # a lengthy envelope

    Notes
    -----
    In the MASS framework implementation, for obtaining a sound with a tremolo (or AM),
    the tremolo pattern is considered separately from a synthesis of the sound.

    The vibrato and FM patterns are considering when synthesizing the sound.

    One might want to run this function twice to obtain
    a stereo reverberation.

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
    Gammaa = (samples*fs*l/fs).astype(n.int64)  # indexes for LUT
    # amplitude variation at each sample
    Ta = taba[ Gammaa % l ]
    T = 1 + Ta*a
    if type(sonic_vector) in (n.ndarray, list):
        return T*sonic_vector
    else:
        return T