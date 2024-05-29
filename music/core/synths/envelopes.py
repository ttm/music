import numpy as np
from music.utils import WAVEFORM_SINE, WAVEFORM_TRIANGULAR


def am(duration=2, fm=50, max_amplitude=.4, waveform_table=WAVEFORM_SINE,
       number_of_samples=0, sonic_vector=0, sample_rate=44100):
    """
    Synthesize an AM envelope or apply it to a sound.

    Set fm=0 or max_amplitude=0 for a constant envelope with value 1. An AM is
    a linear oscillatory pattern of amplitude [1].

    Parameters
    ----------
    duration : scalar
        The duration of the envelope in seconds.
    fm : scalar
        The frequency of the modulation in Hertz.
    max_amplitude : scalar
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
    t : ndarray
        A numpy array where each value is a PCM sample of the envelope. If
        sonic_vectoris input, T is the sonic vector with the AM applied to it.

    See Also
    --------
    note_with_vibrato : A musical note with an oscillation of pitch.
    fm : A linear oscillation of fundamental frequency.
    tremolo : A tremolo, an oscillation of loudness.

    Examples
    --------
    >>> W(V()*am())  # writes a WAV file of a note with tremolo
    >>> s = H([V()*am(fm=i, a=j) for i, j in zip([60, 150, 100],
                                                 [2, 1, 20])])
    >>> s = H([am(fm=i, a=j, sonic_vector=V()) for i, j in zip([60, 150, 100],
                                                               [2, 1, 20])])
    >>> envelope2 = am(440, 150, 60)  # a lengthy envelope

    Notes
    -----
    In the MASS framework implementation, for obtaining a sound with a tremolo
    (or AM), the tremolo pattern is considered separately from a synthesis of
    the sound.

    The AM is an oscilattory pattern of amplitude while the tremolo is an
    oscilattory pattern of loudness being: loudness ~ log(amplitude)

    The vibrato and FM patterns are considered when synthesizing the sound.

    One might want to run this function twice to obtain a stereo reverberation.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """

    waveform_table = np.array(waveform_table)
    if type(sonic_vector) in (np.ndarray, list):
        lambda_am = len(sonic_vector)
    elif number_of_samples:
        lambda_am = number_of_samples
    else:
        lambda_am = np.floor(sample_rate * duration)
    samples = np.arange(lambda_am)

    length = len(waveform_table)
    # indexes for LUT
    gamma_am = (samples * fm * length / sample_rate).astype(np.int64)
    # amplitude variation at each sample
    t_am = waveform_table[gamma_am % length]
    t = 1 + t_am * max_amplitude
    if type(sonic_vector) in (np.ndarray, list):
        return t * sonic_vector
    else:
        return t


def tremolo(duration=2, tremolo_freq=2, max_db_dev=10, alpha=1,
            waveform_table=WAVEFORM_SINE, number_of_samples=0,
            sonic_vector=0, sample_rate=44100):
    """
    Synthesize a tremolo envelope or apply it to a sound.

    Set fa=0 or dB=0 for a constant envelope with value 1. A tremolo is an
    oscillatory pattern of loudness [1].

    Parameters
    ----------
    duration : scalar
        The duration of the envelope in seconds.
    tremolo_freq : scalar
        The frequency of the tremolo oscillations in Hertz.
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
    t : ndarray
        A numpy array where each value is a PCM sample of the envelope.
        If sonic_vector is input, t is the sonic vector with the tremolo
        applied to it.

    See Also
    --------
    note_with_vibrato : A musical note with an oscillation of pitch.
    fm : A linear oscillation of fundamental frequency.
    am : A linear oscillation of amplitude.

    Examples
    --------
    >>> W(V()*t())  # writes a WAV file of a note with tremolo
    >>> s = H([V()*t(fa=i, dB=j) for i, j in zip([6, 15, 100], [2, 1, 20])])
    >>> s = H([t(fa=i, dB=j, sonic_vector=V()) for i, j in zip([6, 15, 100],
                                                               [2, 1, 20])])
    >>> envelope2 = t(440, 1.5, 60)  # a lengthy envelope

    Notes
    -----
    In the MASS framework implementation, for obtaining a sound with a tremolo
    (or AM), the tremolo pattern is considered separately from a synthesis of
    the sound.

    The vibrato and FM patterns are considering when synthesizing the sound.

    Cite the following article whenever you use this function.

    See the envelopes.am function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """

    waveform_table = np.array(waveform_table)
    if type(sonic_vector) in (np.ndarray, list):
        lambda_tremolo = len(sonic_vector)
    elif number_of_samples:
        lambda_tremolo = number_of_samples
    else:
        lambda_tremolo = np.floor(sample_rate * duration)
    samples = np.arange(lambda_tremolo)

    length = len(waveform_table)
    # indexes for LUT
    gamma_tremolo = (samples * tremolo_freq * length /
                     sample_rate).astype(np.int64)
    # amplitude variation at each sample
    table_amp = waveform_table[gamma_tremolo % length]
    if alpha != 1:
        t = 10. ** ((table_amp * max_db_dev / 20) ** alpha)
    else:
        t = 10. ** (table_amp * max_db_dev / 20)
    if type(sonic_vector) in (np.ndarray, list):
        return t * sonic_vector
    else:
        return t


def tremolos(durations=[[3, 4, 5], [2, 3, 7, 4]],
             tremolo_freqs=[[2, 6, 20], [5, 6.2, 21, 5]],
             max_db_devs=[[10, 20, 1], [5, 7, 9, 2]],
             alpha=[[1, 1, 1], [1, 1, 1, 9]],
             waveform_tables=[[WAVEFORM_SINE, WAVEFORM_SINE, WAVEFORM_SINE],
                              [WAVEFORM_TRIANGULAR, WAVEFORM_TRIANGULAR,
                               WAVEFORM_TRIANGULAR, WAVEFORM_SINE]],
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
        The sound to which apply the tremolos. If supplied, the tremolo lines
        are applied to the sound and missing samples are completed by zeros
        (if sonic_vector is smaller then the lengthiest tremolo) or ones
        (is sonic_vector is larger).
    sample_rate : integer
        The sample rate

    Returns
    -------
    s : ndarray
        A numpy array where each value is a value of the envelope for the PCM
        samples. If sonic_vector is supplied, e is the sonic vector with the
        envelope applied to it.

    See Also
    --------
    loud : An envelope for a loudness transition.
    louds : An envelope with an arbitrary number of transitions.
    fade : Fade in and out.
    adsr : An ADSR envelope.
    tremolo : An oscillation of loudness.

    Examples
    --------
    >>> W(V(d=8)*L_())  # writes a WAV file with a loudness transitions

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)


    """
    for i in range(len(waveform_tables)):
        for j in range(i):
            waveform_tables[i][j] = np.array(waveform_tables[i][j])
    t_ = []
    if number_of_samples:
        for i, ns in enumerate(number_of_samples):
            t_.append([])
            for j, ns_ in enumerate(ns):
                s = tremolo(tremolo_freq=tremolo_freqs[i][j],
                            max_db_dev=max_db_devs[i][j], alpha=alpha[i][j],
                            waveform_table=waveform_tables[i][j],
                            number_of_samples=ns_, sample_rate=sample_rate)
                t_[-1].append(s)
    else:
        for i, durs in enumerate(durations):
            t_.append([])
            for j, dur in enumerate(durs):
                s = tremolo(dur, tremolo_freqs[i][j], max_db_devs[i][j],
                            alpha[i][j], waveform_table=waveform_tables[i][j],
                            sample_rate=sample_rate)
                t_[-1].append(s)
    amax = 0
    if type(sonic_vector) in (np.ndarray, list):
        amax = len(sonic_vector)
    for i in range(len(t_)):
        t_[i] = np.hstack(t_[i])
        amax = max(amax, len(t_[i]))
    for i in range(len(t_)):
        if len(t_[i]) < amax:
            t_[i] = np.hstack((t_[i], np.ones(amax - len(t_[i])) * t_[i][-1]))
    if type(sonic_vector) in (np.ndarray, list):
        if len(sonic_vector) < amax:
            sonic_vector = np.hstack((sonic_vector,
                                      np.zeros(amax - len(sonic_vector))))
        t_.append(sonic_vector)
    s = np.prod(t_, axis=0)
    return s
