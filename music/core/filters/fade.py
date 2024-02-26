import numpy as np
from .loud import loud
from ...utils import resolve_stereo, mix_with_offset


def fade(duration=2, fade_out=True, method="exp", dB=-80, alpha=1, perc=1,
         number_of_samples=0, sonic_vector=0, sample_rate=44100):
    """
    A fade in or out.

    Implements the loudness transition and asserts that it reaches
    zero amplitude.

    Parameters
    ----------
    duration : scalar
        The duration in seconds of the fade.
    fade_out : boolean
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
    number_of_samples : integer
        The number of samples of the fade. If supplied, d is ignored.
    sonic_vector : array_like
        Samples for the fade to be applied to.
        If supplied, d and nsamples are ignored.
    sample_rate : integer
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
    >>> W(V()*fade())  # writes a WAV file with a fade in
    >>> s = H( [V()*fade(fade_out=i, method=j) for i, j in zip([1, 0, 1], ["exp", "exp", "linear"])] )  # OR
    >>> s = H( [fade(fade_out=i, method=j, sonic_vector=V()) for i, j in zip([1, 0, 1], ["exp", "exp", "linear"])] )
    >>> envelope = fade(duration=10, fade_out=0, perc=0.1)  # a lengthy fade in

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if type(sonic_vector) in (np.ndarray, list):
        if len(sonic_vector.shape) == 2:
            return resolve_stereo(fade, locals())
        N = len(sonic_vector)
    elif number_of_samples:
        N = number_of_samples
    else:
        N = int(sample_rate * duration)
    if 'lin' in method:
        if fade_out:
            ai = loud(method="linear", trans_dev=0, number_of_samples=N)
        else:
            ai = loud(method="linear", to=0, trans_dev=0, number_of_samples=N)
    if 'exp' in method:
        N0 = int(N*perc/100)
        N1 = N - N0
        if fade_out:
            ai1 = loud(trans_dev=dB, alpha=alpha, number_of_samples=N1)
            if N0:
                ai0 = loud(method="linear", trans_dev=0, number_of_samples=N0) * ai1[-1]
            else:
                ai0 = []
            ai = np.hstack((ai1, ai0))
        else:
            ai1 = loud(trans_dev=dB, to=0, alpha=alpha, number_of_samples=N1)
            if N0:
                ai0 = loud(method="linear", to=0, trans_dev=0, number_of_samples=N0) * ai1[0]
            else:
                ai0 = []
            ai = np.hstack((ai0, ai1))
    if type(sonic_vector) in (np.ndarray, list):
        return ai*sonic_vector
    else:
        return ai


def cross_fade(sonic_vector_1, sonic_vector_2, duration=500, method='lin', sample_rate=44100):
    """
    Cross fade in dur milisseconds.

    """
    ns = int(duration * sample_rate / 1000)
    if len(sonic_vector_1.shape) != len(sonic_vector_2.shape):
        print('enter s1 and s2 with the same shape')
    if len(sonic_vector_1.shape) == 2:
        s1_ = cross_fade(sonic_vector_1[0], sonic_vector_2[0], duration, method, sample_rate)
        s2_ = cross_fade(sonic_vector_1[1], sonic_vector_2[1], duration, method, sample_rate)
        s = np.array((s1_, s2_))
        return s
    sonic_vector_1[-ns:] *= fade(number_of_samples=ns, method=method, sample_rate=sample_rate)
    sonic_vector_2[:ns] *= fade(number_of_samples=ns, method=method, sample_rate=sample_rate, fade_out=False)
    s = mix_with_offset(sonic_vector_1, sonic_vector_2, duration=-duration / 1000)
    return s
