"""Amplitude envelope filters including ADSR-related helpers."""

import numpy as np
from .fade import fade
from .loud import loud
from ...utils import as_sonic_vector


def adsr(envelope_duration=2, attack_duration=20,
         decay_duration=20, sustain_level=-5,
         release_duration=50, transition="exp", alpha=1,
         db_dev=-80, to_zero=1, number_of_samples=0, sonic_vector=0,
         sample_rate=44100):
    """
    Synthesize an ADSR envelope.

    ADSR (Atack, Decay, Sustain, Release) is a very traditional loudness
    envelope in sound synthesis [1].

    Parameters
    ----------
    envelope_duration : scalar
        The duration of the envelope in seconds.
    attack_duration : scalar
        The duration of the Attack in milliseconds.
    decay_duration : scalar
        The duration of the Decay in milliseconds.
    sustain_level : scalar
        The Sustain level after the Decay in decibels.
        Usually negative.
    release_duration : scalar
        The duration of the Release in milliseconds.
    transition : string
        "exp" for exponential transitions of amplitude
        (linear loudness).
        "linear" for linear transitions of amplitude.
    alpha : scalar or array_like
        An index to make the exponential fade slower or faster [1]. Ignored it
        transitions="linear" or alpha=1. If it is an array_like, it should
        hold three values to be used in Attack, Decay and Release.
    db_dev : scalar or array_like
        The decibels deviation to reach before using a linear fade to reach
        zero amplitude. If it is an array_like, it should hold two values, one
        for Attack and another for Release. Ignored if trans="linear".
    to_zero : scalar or array_like
        The duration in milliseconds for linearly departing from zero in the
        Attack and reaching the value of zero at the end of the Release. If it
        is an array_like, it should hold two values, one for Attack and
        another for Release. It's ignored if trans="linear".
    number_of_samples : integer
        The number of samples of the envelope. If supplied, d is ignored.
    sonic_vector : array_like
        Samples for the ADSR envelope to be applied to. If supplied, d and
        nsamples are ignored.
    sample_rate : integer
        The sample rate.

    Returns
    -------
    as : ndarray
        A numpy array where each value is a value of the envelope for the PCM
        samples if sonic_vector is 0. If sonic_vector is input, ad is the
        sonic vector with the ADSR envelope applied to it.

    See Also
    --------
    tremolo : An oscillation of loudness.
    loud : A loudness transition.
    fade : A fade in or fade out.

    Examples
    --------
    >>> write_wav_mono(note_with_vibrato() * adsr())
    >>> s = horizontal_stack([note_with_vibrato() *
    ...                       adsr(attack_duration=i, release_duration=j)
    ...                       for i, j in zip([6, 50, 300], [100, 10, 200])])
    >>> s = horizontal_stack([adsr(A=i, R=j, sonic_vector=note_with_vibrato())
    ...                       for i, j in zip([6, 15, 100], [2, 2, 20])])
    >>> envelope = adsr(d=440, A=10e3, D=0, R=5e3)

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    sonic_vector = as_sonic_vector(sonic_vector)
    if sonic_vector is not None:
        lambda_adsr = len(sonic_vector)
    elif number_of_samples:
        lambda_adsr = number_of_samples
    else:
        lambda_adsr = int(envelope_duration * sample_rate)
    lambda_a = int(attack_duration * sample_rate * 0.001)
    lambda_d = int(decay_duration * sample_rate * 0.001)
    lambda_r = int(release_duration * sample_rate * 0.001)

    # The attack, decay and release stages cannot outlast the envelope they
    # belong to: compress them proportionally when the sound is too short,
    # rather than leaving the sustain stage with a negative length.
    stages = lambda_a + lambda_d + lambda_r
    if stages > lambda_adsr:
        ratio = lambda_adsr / stages
        lambda_a = int(lambda_a * ratio)
        lambda_d = int(lambda_d * ratio)
        lambda_r = int(lambda_r * ratio)

    # A stage of zero length is built explicitly: fade() and loud() treat
    # number_of_samples=0 as "unset" and fall back to their own default
    # duration, which would stretch the envelope past the sound.
    if lambda_a:
        attack = fade(fade_out=0, method=transition, alpha=alpha, db=db_dev,
                      perc=to_zero / attack_duration,
                      number_of_samples=lambda_a)
    else:
        attack = np.array([])

    if lambda_d:
        decay = loud(trans_dev=sustain_level, method=transition,
                     alpha=alpha, number_of_samples=lambda_d)
    else:
        decay = np.array([])

    a_s = 10 ** (sustain_level / 20.)
    sustain = np.ones(lambda_adsr - (lambda_a + lambda_r + lambda_d)) * a_s

    if lambda_r:
        release = fade(method=transition, alpha=alpha, db=db_dev,
                       perc=to_zero / release_duration,
                       number_of_samples=lambda_r) * a_s
    else:
        release = np.array([])

    ad = np.hstack((attack, decay, sustain, release))
    if sonic_vector is not None:
        return sonic_vector * ad
    return ad


def adsr_vibrato(note_dict={}, adsr_dict={}):
    """
    Creates a note with a vibrato and an ADSR envelope.

    A shorthand for calling :func:`music.note_with_vibrato` and passing
    the result to :func:`adsr`, so that the two sets of arguments do not
    have to be interleaved at the call site.

    Parameters
    ----------
    note_dict : dict
        Keyword arguments for :func:`music.note_with_vibrato`, which
        renders the note the envelope is applied to.
    adsr_dict : dict
        Keyword arguments for :func:`adsr`, which shapes it.
        ``sonic_vector`` is supplied by this function and should not be
        given here.

    Returns
    -------
    ndarray
        A mono sequence of PCM samples: the vibrato note under the
        envelope.

    See Also
    --------
    adsr : the envelope, and the meaning of every key of ``adsr_dict``.
    music.note_with_vibrato : the note, and the keys of ``note_dict``.

    Examples
    --------
    >>> sound = adsr_vibrato(note_dict={'freq': 220, 'duration': 1},
    ...                      adsr_dict={'sustain_level': -10})
    >>> sound.ndim
    1

    """
    # imported here rather than at module scope: music.core.synths.notes
    # imports this module back, and a top-level import would make the
    # filters <-> synths cycle bind names during partial initialisation.
    from ..synths.notes import note_with_vibrato
    return adsr(sonic_vector=note_with_vibrato(**note_dict), **adsr_dict)


def adsr_stereo(duration=2, attack_duration=20, decay_duration=20,
                sustain_level=-5, release_duration=50, transition="exp",
                alpha=1, db_dev=-80, to_zero=1, number_of_samples=0,
                sonic_vector=0, sample_rate=44100):
    """
    A shorthand to make an ADSR envelope for a stereo sound.

    :func:`adsr` is applied to each channel with the same arguments, so
    the two channels keep their relative level through the envelope
    rather than being shaped independently.

    Parameters
    ----------
    duration : scalar
        The duration of the envelope in seconds. Passed to :func:`adsr`
        as ``envelope_duration``.
    attack_duration : scalar
        The duration of the Attack in milliseconds.
    decay_duration : scalar
        The duration of the Decay in milliseconds.
    sustain_level : scalar
        The Sustain level after the Decay in decibels. Usually negative.
    release_duration : scalar
        The duration of the Release in milliseconds.
    transition : string
        "exp" for exponential transitions of amplitude (linear
        loudness), "linear" for linear transitions of amplitude.
    alpha : scalar or array_like
        An index to make the exponential fade slower or faster. Ignored
        if ``transition="linear"`` or ``alpha=1``. An array_like should
        hold three values, for Attack, Decay and Release.
    db_dev : scalar or array_like
        The decibels deviation to reach before using a linear fade to
        reach zero amplitude. An array_like should hold two values, for
        Attack and Release. Ignored if ``transition="linear"``.
    to_zero : scalar or array_like
        The duration in milliseconds for linearly departing from zero in
        the Attack and reaching zero at the end of the Release. An
        array_like should hold two values, for Attack and Release.
        Ignored if ``transition="linear"``.
    number_of_samples : integer
        The number of samples of the envelope. If supplied, ``duration``
        is ignored.
    sonic_vector : array_like
        A ``(2, nsamples)`` stereo sound for the envelope to be applied
        to. If supplied, ``duration`` and ``number_of_samples`` are
        ignored.
    sample_rate : integer
        The sample rate.

    Returns
    -------
    ndarray
        A ``(2, nsamples)`` array: the envelope itself if
        ``sonic_vector`` is 0, or the sound with the envelope applied.

    See Also
    --------
    adsr : the mono envelope, and the meaning of every argument here.

    Examples
    --------
    >>> envelope = adsr_stereo(duration=1)
    >>> envelope.shape[0]
    2

    """
    sonic_vector = as_sonic_vector(sonic_vector)
    if sonic_vector is not None:
        sonic_vector1 = sonic_vector[0]
        sonic_vector2 = sonic_vector[1]
    else:
        sonic_vector1 = 0
        sonic_vector2 = 0
    s1 = adsr(envelope_duration=duration, attack_duration=attack_duration,
              decay_duration=decay_duration, sustain_level=sustain_level,
              release_duration=release_duration, transition=transition,
              alpha=alpha, db_dev=db_dev, to_zero=to_zero,
              number_of_samples=number_of_samples, sonic_vector=sonic_vector1,
              sample_rate=sample_rate)
    s2 = adsr(envelope_duration=duration, attack_duration=attack_duration,
              decay_duration=decay_duration, sustain_level=sustain_level,
              release_duration=release_duration, transition=transition,
              alpha=alpha, db_dev=db_dev, to_zero=to_zero,
              number_of_samples=number_of_samples, sonic_vector=sonic_vector2,
              sample_rate=sample_rate)
    s = np.vstack((s1, s2))
    return s
