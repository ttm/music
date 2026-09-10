"""Amplitude fade filters for smooth transitions."""

import numpy as np
from .loud import loud
from ...utils import (resolve_stereo, mix_with_offset,
                      as_sonic_vector)


def fade(duration=2, fade_out=True, method="exp", db=-80, alpha=1, perc=1,
         number_of_samples=0, sonic_vector=0, sample_rate=44100):
    """
    A fade in or out.

    Implements the loudness transition and asserts that it reaches zero
    amplitude.

    Parameters
    ----------
    duration : scalar
        The duration in seconds of the fade.
    fade_out : boolean
        If True, the fade is a fade out, else it is a fade in.
    method : string
        "exp" for an exponential transition of amplitude (linear loudness).
        "linear" for a linear transition of amplitude.
    db : scalar
        The decibels from which to reach before using the linear transition to
        reach zero. Not used if method="linear".
    alpha : scalar
        An index to make the exponential fade slower or faster [1]. Ignored if
        transitions="linear".
    perc : scalar
        The percentage of the fade that is linear to ensure it reaches zero.
        It has no effect if method="linear".
    number_of_samples : integer
        The number of samples of the fade. If supplied, d is ignored.
    sonic_vector : array_like
        Samples for the fade to be applied to. If supplied, d and nsamples are
        ignored.
    sample_rate : integer
        The sample rate. Only used if number_of_samples and sonic_vector are
        not supplied.

    Returns
    -------
    ai : ndarray
        Each value is a value of the envelope for the PCM samples. If
        sonic_vector is input, ai is the sonic vector with the fade applied to
        it.

    Raises
    ------
    ValueError
        If ``perc`` is outside [0, 100], or if ``method`` names neither
        "lin" nor "exp". A percentage over 100 used to reach an
        ``IndexError`` from inside the routine, and an unknown method an
        ``UnboundLocalError``, neither of which says what the caller did.

    See Also
    --------
    adsr : An ADSR envelope.
    loud : A transition of loudness.
    louds : An envelope with an arbitrary number or loudness transitions.
    tremolo : An oscillation of loudness.

    Examples
    --------
    >>> write_wav_mono(note_with_vibrato() * fade())
    >>> s = horizontal_stack(*[note_with_vibrato() * fade(fade_out=i, method=j)
    ...                       for i, j in zip([1, 0, 1],
    ...                                       ["exp", "exp", "linear"])])
    >>> s = horizontal_stack(
    ...     *[fade(fade_out=i, method=j, sonic_vector=note_with_vibrato())
    ...                       for i, j in zip([1, 0, 1],
    ...                                       ["exp", "exp", "linear"])])
    >>> envelope = fade(duration=10, fade_out=0, perc=0.1)

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if not 0 <= perc <= 100:
        raise ValueError(
            "perc is the percentage of the fade that is linear and must lie "
            f"in [0, 100]; got {perc}")
    if "lin" not in method and "exp" not in method:
        raise ValueError(
            f'method must name "lin" or "exp"; got {method!r}')
    sonic_vector = as_sonic_vector(sonic_vector)
    if sonic_vector is not None:
        if len(sonic_vector.shape) == 2:
            return resolve_stereo(fade, locals())
        n = len(sonic_vector)
    elif number_of_samples:
        n = number_of_samples
    else:
        n = int(sample_rate * duration)
    if n < 1:
        # Both branches below hand `n` to `loud`, where a
        # number_of_samples of zero means "not supplied" and gives back
        # two seconds. The exponential branch was guarded when its split
        # was fixed; this catches the linear one, which the zero-duration
        # sweep never reached because it only ever calls the default
        # method.
        return np.array([])
    if 'lin' in method:
        if fade_out:
            ai = loud(method="linear", trans_dev=0, number_of_samples=n)
        else:
            ai = loud(method="linear", to=0, trans_dev=0, number_of_samples=n)
    if 'exp' in method:
        n0 = int(n*perc/100)
        n1 = n - n0
        # `loud` reads number_of_samples=0 as "use the default duration",
        # so an empty part has to be built here rather than asked for. At
        # perc=100 the whole fade is the linear part, and asking for zero
        # exponential samples returned two seconds of them: a fade over
        # half a second came back 110,250 samples long instead of 22,050,
        # which is the default length added to the one requested. The n0
        # side was already guarded; this is the other one.
        if fade_out:
            ai1 = (loud(trans_dev=db, alpha=alpha, number_of_samples=n1)
                   if n1 else np.array([]))
            # Where the linear part picks up. With no exponential part it
            # starts at full amplitude, which is what perc=100 means.
            joins_at = ai1[-1] if n1 else 1.0
            ai0 = (loud(method="linear", trans_dev=0,
                        number_of_samples=n0) * joins_at if n0 else [])
            ai = np.hstack((ai1, ai0))
        else:
            ai1 = (loud(trans_dev=db, to=0, alpha=alpha,
                        number_of_samples=n1) if n1 else np.array([]))
            joins_at = ai1[0] if n1 else 1.0
            ai0 = (loud(method="linear", to=0, trans_dev=0,
                        number_of_samples=n0) * joins_at if n0 else [])
            ai = np.hstack((ai0, ai1))
    if sonic_vector is not None:
        return ai*sonic_vector
    return ai


def cross_fade(sonic_vector_1, sonic_vector_2, duration=500, method='lin',
               sample_rate=44100):
    """
    Cross fade two sounds over `duration` milliseconds.

    The tail of the first sound is faded out, the head of the second is
    faded in, and the two are overlapped by that duration, so the result
    is shorter than the two inputs laid end to end.

    Parameters
    ----------
    sonic_vector_1 : ndarray
        The sound that fades out. Mono, or ``(2, nsamples)`` stereo.
    sonic_vector_2 : ndarray
        The sound that fades in, of the same number of dimensions.
    duration : scalar
        The length of the crossfade in milliseconds.
    method : string
        The fade shape, as :func:`fade` takes it: "lin" or "exp".
    sample_rate : integer
        The sample rate in Hertz.

    Returns
    -------
    ndarray
        The two sounds overlapped, ``duration`` milliseconds shorter
        than their concatenation.

    Raises
    ------
    ValueError
        If the two sounds do not have the same number of dimensions.
        Crossfading a mono sound with a stereo one has no single sensible
        answer, so it is refused rather than guessed at.

    Notes
    -----
    **Both inputs are modified in place.** The fades are applied to the
    caller's arrays rather than to copies, so a sound that is crossfaded
    is no longer the sound it was. Pass a copy to keep the original.

    The overlap makes the result shorter, which is why
    :class:`music.StimulationSession` does not use this function: a
    protocol's phase durations have to survive its transitions.

    See Also
    --------
    fade : the fade in or out on its own.
    music.StimulationSession : crossfades that preserve total duration.


    Examples
    --------
    >>> joined = cross_fade(note(220, 1), note(330, 1), duration=200)
    >>> len(joined) / 44100    # the overlap is 200 ms of the two seconds
    1.8
    """
    ns = int(duration * sample_rate / 1000)
    if len(sonic_vector_1.shape) != len(sonic_vector_2.shape):
        raise ValueError('sonic_vector_1 and sonic_vector_2 must have '
                         'the same shape')
    if len(sonic_vector_1.shape) == 2:
        s1_ = cross_fade(sonic_vector_1[0], sonic_vector_2[0], duration,
                         method, sample_rate)
        s2_ = cross_fade(sonic_vector_1[1], sonic_vector_2[1], duration,
                         method, sample_rate)
        s = np.array((s1_, s2_))
        return s
    sonic_vector_1[-ns:] *= fade(number_of_samples=ns, method=method,
                                 sample_rate=sample_rate)
    sonic_vector_2[:ns] *= fade(number_of_samples=ns, method=method,
                                sample_rate=sample_rate, fade_out=False)
    s = mix_with_offset(sonic_vector_1, sonic_vector_2,
                        duration=-duration / 1000)
    return s
