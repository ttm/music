"""Amplitude fade filters for smooth transitions."""

import numpy as np
from .loud import loud
from ..._deprecation import renamed_arguments
from ..._rate import resolve_rate
from ...utils import (resolve_stereo, mix_with_offset,
                      as_sonic_vector, _transition_is_linear)


def fade(duration=2, fade_out=True, method="exp", db=-80, alpha=1, perc=1,
         number_of_samples=None, sonic_vector=0, sample_rate=None):
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
        "exp" or "exponential" for an exponential transition of amplitude
        (linear loudness); "lin" or "linear" for a linear transition of
        amplitude.
    db : scalar
        The decibels from which to reach before using the linear transition to
        reach zero. Not used if method="linear".
    alpha : scalar
        An index to make the exponential fade slower or faster [1]. Ignored if
        transitions="linear".
    perc : scalar
        The percentage of the fade that is linear to ensure it reaches zero.
        It has no effect if method="linear".
    number_of_samples : integer or None
        The number of samples of the fade. If given, duration is ignored,
        and zero is a fade of no samples.
    sonic_vector : array_like
        Samples for the fade to be applied to. If supplied, duration and
        number_of_samples are ignored.
    sample_rate : integer or None
        The sample rate. Only used if number_of_samples and sonic_vector are
        not supplied. None, the default, is the rate
        :func:`music.get_sample_rate` returns.

    Returns
    -------
    ai : ndarray
        Each value is a value of the envelope for the PCM samples. If
        sonic_vector is input, ai is the sonic vector with the fade applied to
        it.

    Raises
    ------
    ValueError
        If ``perc`` is outside [0, 100], or if ``method`` is not one of
        the four names above. A percentage over 100 used to reach an
        ``IndexError`` from inside the routine, and an unknown method an
        ``UnboundLocalError``, neither of which says what the caller did.
        Names were once matched by substring, so ``"explicit"`` was
        exponential; they are now the ones :func:`loud` takes.

        If `alpha` is negative and the transition exponential: it bends
        a transition from zero, where a negative power has no finite
        value, and the envelope came back holding infinity.

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
    sample_rate = resolve_rate(sample_rate)
    if not 0 <= perc <= 100:
        raise ValueError(
            "perc is the percentage of the fade that is linear and must lie "
            f"in [0, 100]; got {perc}")
    sonic_vector = as_sonic_vector(sonic_vector)
    if sonic_vector is not None:
        if len(sonic_vector.shape) == 2:
            return resolve_stereo(fade, locals())
        n = len(sonic_vector)
    elif number_of_samples is not None:
        n = number_of_samples
    else:
        n = int(sample_rate * duration)
    # Checked after the stereo branch, which hands this call's locals to
    # each channel's: a name bound before it would arrive as an argument.
    linear = _transition_is_linear(method)
    if n < 1:
        # No samples to fade. A negative count is no samples too, and the
        # split below would find no last sample to join its parts at.
        return np.array([])
    if linear:
        if fade_out:
            ai = loud(method="linear", trans_dev=0, number_of_samples=n)
        else:
            ai = loud(method="linear", to=0, trans_dev=0, number_of_samples=n)
    else:
        n0 = int(n*perc/100)
        n1 = n - n0
        # Either part may be empty: at perc=100 the whole fade is the
        # linear part, and at a small perc on a short fade there is none.
        # Asking `loud` for zero samples of either used to return two
        # seconds of them, since zero meant "not given": a fade over half
        # a second came back 110,250 samples long instead of 22,050.
        if fade_out:
            ai1 = loud(trans_dev=db, alpha=alpha, number_of_samples=n1)
            # Where the linear part picks up. With no exponential part it
            # starts at full amplitude, which is what perc=100 means.
            joins_at = ai1[-1] if n1 else 1.0
            ai0 = loud(method="linear", trans_dev=0,
                       number_of_samples=n0) * joins_at
            ai = np.hstack((ai1, ai0))
        else:
            ai1 = loud(trans_dev=db, to=0, alpha=alpha,
                       number_of_samples=n1)
            joins_at = ai1[0] if n1 else 1.0
            ai0 = loud(method="linear", to=0, trans_dev=0,
                       number_of_samples=n0) * joins_at
            ai = np.hstack((ai0, ai1))
    if sonic_vector is not None:
        return ai*sonic_vector
    return ai


def cross_fade(sonic_vector_1, sonic_vector_2, duration_ms=500,
               method='lin', sample_rate=None, **renamed):
    """
    Cross fade two sounds over `duration_ms` milliseconds.

    The tail of the first sound is faded out, the head of the second is
    faded in, and the two are overlapped by that duration, so the result
    is shorter than the two inputs laid end to end.

    Parameters
    ----------
    sonic_vector_1 : array_like
        The sound that fades out. Mono, or ``(2, nsamples)`` stereo.
    sonic_vector_2 : array_like
        The sound that fades in, of the same number of dimensions.
    duration_ms : scalar
        The length of the crossfade in milliseconds.
    method : string
        The fade shape, as :func:`fade` takes it: "lin" or "linear", "exp"
        or "exponential".
    sample_rate : integer or None
        The sample rate in Hertz. None, the default, is the rate
        :func:`music.get_sample_rate` returns.
    **renamed
        ``duration``, the name ``duration_ms`` had before it said its
        unit beside :func:`fade`'s ``duration``, which is in seconds.
        Still accepted, with a :class:`DeprecationWarning`.

    Returns
    -------
    ndarray
        The two sounds overlapped, ``duration_ms`` milliseconds shorter
        than their concatenation.

    Raises
    ------
    ValueError
        If the two sounds do not have the same number of dimensions.
        Crossfading a mono sound with a stereo one has no single sensible
        answer, so it is refused rather than guessed at.

        If ``duration_ms`` is not a positive number of samples at
        ``sample_rate``, or is longer than the shorter of the two sounds.
        Both used to reach a NumPy broadcast failure naming two sample
        counts, which does not say which duration caused it.
    TypeError
        If a keyword is neither a parameter nor the old name.

    Notes
    -----
    The inputs are left as they were. The fades used to be applied to the
    caller's arrays, the one routine here that did that, so crossfading
    the same two sounds twice faded each of them twice.

    The overlap makes the result shorter, which is why
    :class:`music.StimulationSession` does not use this function: a
    protocol's phase durations have to survive its transitions.

    See Also
    --------
    fade : the fade in or out on its own.
    music.StimulationSession : crossfades that preserve total duration.


    Examples
    --------
    >>> joined = cross_fade(note(220, 1), note(330, 1), duration_ms=200)
    >>> len(joined) / 44100    # the overlap is 200 ms of the two seconds
    1.8
    """
    sample_rate = resolve_rate(sample_rate)
    duration_ms = renamed_arguments('cross_fade', renamed,
                                    duration='duration_ms'
                                    ).get('duration_ms', duration_ms)
    # Copies, which the fades below are applied to: they were applied to
    # the caller's arrays.
    sonic_vector_1 = np.array(sonic_vector_1, dtype=np.float64)
    sonic_vector_2 = np.array(sonic_vector_2, dtype=np.float64)
    ns = int(duration_ms * sample_rate / 1000)
    if len(sonic_vector_1.shape) != len(sonic_vector_2.shape):
        raise ValueError('sonic_vector_1 and sonic_vector_2 must have '
                         'the same shape')
    if len(sonic_vector_1.shape) == 2:
        s1_ = cross_fade(sonic_vector_1[0], sonic_vector_2[0], duration_ms,
                         method, sample_rate)
        s2_ = cross_fade(sonic_vector_1[1], sonic_vector_2[1], duration_ms,
                         method, sample_rate)
        s = np.array((s1_, s2_))
        return s
    shortest = min(len(sonic_vector_1), len(sonic_vector_2))
    if ns < 1:
        # `v[-0:]` is the whole of v rather than none of it, so a duration
        # of no samples multiplied a whole sound by an empty fade: a
        # broadcast failure naming sample counts, which says nothing about
        # the duration.
        raise ValueError(
            f"duration_ms ({duration_ms} ms) must be a positive number of "
            f"samples at {sample_rate} Hz; got {ns}")
    if ns > shortest:
        raise ValueError(
            f"duration_ms ({duration_ms} ms, {ns} samples at {sample_rate} "
            f"Hz) cannot exceed the shorter of the two sounds ({shortest} "
            f"samples): there is nothing for the rest of the fade to "
            f"overlap with")
    sonic_vector_1[-ns:] *= fade(number_of_samples=ns, method=method,
                                 sample_rate=sample_rate)
    sonic_vector_2[:ns] *= fade(number_of_samples=ns, method=method,
                                sample_rate=sample_rate, fade_out=False)
    # The overlap has to be measured at the same rate the fades were cut
    # at. Without this, `mix_with_offset` placed the second sound using its
    # own default of 44.1 kHz while the fades above used `sample_rate`, so
    # at any other rate the two sounds overlapped at full level: a 3 and a
    # 5 crossfaded at 8 kHz summed to 8 where they should have stayed
    # between 3 and 5. Same defect as the export fades fixed in 1.8.1.
    s = mix_with_offset(sonic_vector_1, sonic_vector_2,
                        duration=-duration_ms / 1000, sample_rate=sample_rate)
    return s
