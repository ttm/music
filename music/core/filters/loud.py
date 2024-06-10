import numpy as np


def loud(duration=2, trans_dev=10, alpha=1, to=True, method="exp",
         number_of_samples=0, sonic_vector=0, sample_rate=44100):
    """
    An envelope for a linear or exponential transition of amplitude.

    An exponential transition of loudness yields a linear transition of
    loudness (theoretically).

    Parameters
    ----------
    duration : scalar
        The duration of the envelope in seconds.
    trans_dev : scalar
        The deviation of the transition. If method="exp", the deviation is in
        decibels. If method="linear", the deviation is an amplitude proportion.
    alpha : scalar
        An index to make the transition slower or faster [1].
        Ignored if method="linear".
    to : boolean
        If True, the transition ends at the deviation.
        If False, the transition starts at the deviation.
    method : string
        "exp" for an exponential transitions of amplitude (linear loudness).
        "linear" for a linear transition of amplitude.
    number_of_samples : integer
        The number of samples of the envelope.
        If supplied, d is ignored.
    sonic_vector : array_like
        Samples for the envelope to be applied to.
        If supplied, d and nsamples are ignored.
    sample_rate : integer
        The sample rate.
        Only used if nsamples and sonic_vector are not supplied.

    Returns
    -------
    e : ndarray
        A numpy array where each value is a value of the envelope for the PCM
        samples. If sonic_vector is supplied, e is the sonic vector with the
        envelope applied to it.

    See Also
    --------
    louds : An envelope with an arbitrary number of transitions.
    fade : Fade in and out.
    adsr : An ADSR envelope.
    tremolo : An oscillation of loudness.

    Examples
    --------
    >>> write_wav_mono(note_with_vibrato() * loud())
    >>> s = horizontal_stack([note_with_vibrato() *
    ...                       loud(trans_dev=i, method=j)
    ...                       for i, j in zip([6, -50, 2.3],
    ...                                       ["exp", "exp", "linear"])])
    >>> s = horizontal_stack([
    ...     loud(trans_dev=i, method=j, sonic_vector=note_with_vibrato())
    ...     for i, j in zip([6, -50, 2.3], ["exp", "exp", "linear"])])
    >>> envelope = loud(duration=10, trans_dev=-80, to=False, alpha=2)

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if type(sonic_vector) in (np.ndarray, list):
        n = len(sonic_vector)
    elif number_of_samples:
        n = number_of_samples
    else:
        n = int(sample_rate * duration)
    samples = np.arange(n)
    n_ = n - 1
    if 'lin' in method:
        if to:
            a0 = 1
            al = trans_dev
        else:
            a0 = trans_dev
            al = 1
        e = a0 + (al - a0) * samples / n_
    if 'exp' in method:
        if to:
            if alpha != 1:
                samples_ = (samples / n_) ** alpha
            else:
                samples_ = (samples / n_)
        else:
            if alpha != 1:
                samples_ = ((n_ - samples) / n_) ** alpha
            else:
                samples_ = ((n_ - samples) / n_)
        e = 10 ** (samples_ * trans_dev / 20)
    if type(sonic_vector) in (np.ndarray, list):
        return e * sonic_vector
    else:
        return e


def louds(durations=[2, 4, 2], trans_devs=[5, -10, 20], alpha=[1, .5, 20],
          method=["exp", "exp", "exp"], number_of_samples=0, sonic_vector=0,
          sample_rate=44100):
    """
    An envelope with linear or exponential transitions of amplitude.

    See L() for more details.

    Parameters
    ----------
    durations : iterable
        The durations of the transitions in seconds.
    trans_devs : iterable
        The sequence of deviations of the transitions.
        If method="exp" the deviations are in decibels.
        If method="linear" the deviations are amplitude proportions.
    alpha : iterable
        Indexes to make the transitions slower or faster [1].
        Ignored it method="linear".
    method : iterable
        Methods for each transition.
        "exp" for exponential transitions of amplitude (linear loudness).
        "linear" for linear transitions of amplitude.
    number_of_samples : interable
        The number of samples of each transition.
        If supplied, durations is ignored.
    sonic_vector : array_like
        Samples for the envelope to be applied to.
        If supplied, durations or number_of_samples is given, the final sound
        has the greatest duration of sonic_vector and durations
        (or number_of_samples) and missing samples are replaced with silence
        (if sonic_vector is shorter) or with a constant value (if durations or
        number_of_samples yield shorter sequences).
    sample_rate : integer
        The sample rate.
        Only used if nsamples and sonic_vector are not supplied.

    Returns
    -------
    e : ndarray
        A numpy array where each value is a value of the envelope for the PCM
        samples. If sonic_vector is supplied, e is the sonic vector with the
        envelope applied to it.

    See Also
    --------
    loud : An envelope for a loudness transition.
    fade : Fade in and out.
    adsr : An ADSR envelope.
    tremolo : An oscillation of loudness.

    Examples
    --------
    >>> write_wav_mono(note_with_vibrato(duraton=8) * louds())

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if type(sonic_vector) in (np.ndarray, list):
        n = len(sonic_vector)
    elif number_of_samples:
        n = sum(number_of_samples)
    else:
        n = int(sample_rate * sum(durations))
    samples = np.arange(n)  # TODO: eliminate?
    obj = object
    obj.foo = samples
    s = []
    fact = 1
    if number_of_samples:
        for i, ns in enumerate(number_of_samples):
            s_ = loud(trans_devs[i], alpha[i], number_of_samples=ns,
                      method=method[i]) * fact
            s.append(s_)
            fact = s_[-1]
    else:
        for i, dur in enumerate(durations):
            s_ = loud(dur, trans_devs[i], alpha[i],
                      method=method[i], sample_rate=sample_rate) * fact
            s.append(s_)
            fact = s_[-1]
    e = np.hstack(s)

    if type(sonic_vector) in (np.ndarray, list):
        if len(e) < len(sonic_vector):
            s = np.hstack((e, np.ones(len(sonic_vector) - len(e)) * e[-1]))
        if len(e) > len(sonic_vector):
            sonic_vector = np.hstack((sonic_vector, np.ones(
                len(e) - len(sonic_vector)) * e[-1]))
        return sonic_vector * e
    else:
        return e
