import numpy as np
from music.utils import WAVEFORM_SINE, WAVEFORM_TRIANGULAR


def am(duration=2, fm=50, max_amplitude=.4, waveform_table=WAVEFORM_SINE,
       number_of_samples=0, sonic_vector=0, sample_rate=44100):
    """
    Synthesize an AM envelope or apply it to a sound.

    Set fm=0 or a=0 for a constant envelope with value 1.
    An AM is a linear oscillatory pattern of amplitude [1].

    Parameters
    ----------
    duration : scalar
        The duration of the envelope in seconds.
    fm : scalar
        The frequency of the modular in Hertz.
    a : scalar in [0,1]
        The maximum deviation of amplitude of the AM.
    waveform_table : array_like
        The table with the waveform for the tremolo oscillatory pattern.
    number_of_samples : integer
        The number of samples of the envelope. If supplied, d is ignored.
    sonic_vector : array_like
        Samples for the tremolo to be applied to.
        If supplied, d and nsamples are ignored.
    sample_rate : integer
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
    >>> W(V()*am())  # writes a WAV file of a note with tremolo
    >>> s = H( [V()*am(fm=i, a=j) for i, j in zip([60, 150, 100], [2, 1, 20])] )  # OR
    >>> s = H( [am(fm=i, a=j, sonic_vector=V()) for i, j in zip([60, 150, 100], [2, 1, 20])] )
    >>> envelope2 = am(440, 150, 60)  # a lengthy envelope

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

    waveform_table = np.array(waveform_table)
    if type(sonic_vector) in (np.ndarray, list):
        Lambda = len(sonic_vector)
    elif number_of_samples:
        Lambda = number_of_samples
    else:
        Lambda = np.floor(sample_rate * duration)
    samples = np.arange(Lambda)

    l = len(waveform_table)
    Gammaa = (samples * sample_rate * l / sample_rate).astype(np.int64)  # indexes for LUT
    # amplitude variation at each sample
    Ta = waveform_table[Gammaa % l]
    T = 1 + Ta * a
    if type(sonic_vector) in (np.ndarray, list):
        return T * sonic_vector
    else:
        return T
    
    
def tremolo(duration=2, tremolo_freq=2, max_db_dev=10, alpha=1,
            waveform_table=WAVEFORM_SINE, number_of_samples=0,
            sonic_vector=0, sample_rate=44100):
    """
    Synthesize a tremolo envelope or apply it to a sound.

    Set fa=0 or dB=0 for a constant envelope with value 1.
    A tremolo is an oscillatory pattern of loudness [1].

    Parameters
    ----------
    duration : scalar
        The duration of the envelope in seconds.
    tremolo_freq : scalar
        The frequency of the tremolo oscillations in hertz.
    max_db_dev : scalar
        The maximum deviation of loudness in the tremolo in decibels.
    alpha : scalar
        An index to distort the tremolo pattern [1].
    waveform_table : array_like
        The table with the waveform for the tremolo oscillatory pattern.
    number_of_samples : integer
        The number of samples of the envelope. If supplied, d is ignored.
    sonic_vector : array_like
        Samples for the tremolo to be applied to.
        If supplied, d and nsamples are ignored.
    sample_rate : integer
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

    waveform_table = np.array(waveform_table)
    if type(sonic_vector) in (np.ndarray, list):
        Lambda = len(sonic_vector)
    elif number_of_samples:
        Lambda = number_of_samples
    else:
        Lambda = np.floor(sample_rate * duration)
    samples = np.arange(Lambda)

    l = len(waveform_table)
    Gammaa = (samples * tremolo_freq * l / sample_rate).astype(np.int64)  # indexes for LUT
    # amplitude variation at each sample
    Ta = waveform_table[Gammaa % l]
    if alpha != 1:
        T = 10. ** ((Ta * max_db_dev / 20) ** alpha)
    else:
        T = 10. ** (Ta * max_db_dev / 20)
    if type(sonic_vector) in (np.ndarray, list):
        return T * sonic_vector
    else:
        return T


# FIXME: Unused param (`sample_rate`)
def tremolos(durations=[[3, 4, 5], [2, 3, 7, 4]],
             tremolo_freqs=[[2, 6, 20], [5, 6.2, 21, 5]],
             max_db_devs=[[10, 20, 1], [5, 7, 9, 2]],
             alpha=[[1, 1, 1], [1, 1, 1, 9]],
             waveform_tables=[[WAVEFORM_SINE, WAVEFORM_SINE, WAVEFORM_SINE],
                              [WAVEFORM_TRIANGULAR, WAVEFORM_TRIANGULAR, WAVEFORM_TRIANGULAR, WAVEFORM_SINE]],
             number_of_samples=0, sonic_vector=0, sample_rate=44100):
    """
    An envelope with multiple tremolos.

    Parameters
    ----------
    durations : iterable of iterable of scalars
        the durations of each tremolo.
    tremolo_freqs : iterable of iterable of scalars
        The frequencies of each tremolo.
    max_db_devs : iterable of iterable of scalars
        The maximum loudness variation
        of each tremolo.
    alpha : iterable of iterable of scalars
        Indexes for distortion of each tremolo [1].
    waveform_tables : iterable of iterable of array_likes
        Tables for lookup for each tremolo.
    number_of_samples : iterable of iterable of scalars
        The number of samples or each tremolo.
    sonic_vector : array_like
        The sound to which apply the tremolos.
        If supplied, the tremolo lines are
        applied to the sound and missing samples
        are completed by zeros (if sonic_vector
        is smaller then the lengthiest tremolo)
        or ones (is sonic_vector is larger).
    sample_rate : integer
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
    for i in range(len(waveform_tables)):
        for j in range(i):
            waveform_tables[i][j] = np.array(waveform_tables[i][j])
    T_ = []
    if number_of_samples:
        for i, ns in enumerate(number_of_samples):
            T_.append([])
            for j, ns_ in enumerate(ns):
                s = tremolo(tremolo_freq=tremolo_freqs[i][j], max_db_dev=max_db_devs[i][j], alpha=alpha[i][j],
                            waveform_table=waveform_tables[i][j], number_of_samples=ns_)
                T_[-1].append(s)
    else:
        for i, durs in enumerate(durations):
            T_.append([])
            for j, dur in enumerate(durs):
                s = tremolo(dur, tremolo_freqs[i][j], max_db_devs[i][j], alpha[i][j],
                            waveform_table=waveform_tables[i][j])
                T_[-1].append(s)
    amax = 0
    if type(sonic_vector) in (np.ndarray, list):
        amax = len(sonic_vector)
    for i in range(len(T_)):
        T_[i] = np.hstack(T_[i])
        amax = max(amax, len(T_[i]))
    for i in range(len(T_)):
        if len(T_[i]) < amax:
            T_[i] = np.hstack((T_[i], np.ones(amax - len(T_[i])) * T_[i][-1]))
    if type(sonic_vector) in (np.ndarray, list):
        if len(sonic_vector) < amax:
            sonic_vector = np.hstack((sonic_vector, np.zeros(amax - len(sonic_vector))))
        T_.append(sonic_vector)
    s = np.prod(T_, axis=0)
    return s
