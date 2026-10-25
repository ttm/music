"""Amplitude envelope filters including ADSR-related helpers."""

import numpy as np
from .fade import fade
from .loud import loud
from ..._deprecation import renamed_arguments
from ..._rate import resolve_rate
from ...utils import _transition_is_linear, as_sonic_vector

#: The parameters :func:`adsr` and :func:`adsr_stereo` renamed so their
#: names say they are milliseconds, among parameters that are seconds.
_RENAMED = {'attack_duration': 'attack_ms', 'decay_duration': 'decay_ms',
            'release_duration': 'release_ms', 'to_zero': 'to_zero_ms'}


def adsr(envelope_duration=2, attack_ms=20,
         decay_ms=20, sustain_level=-5,
         release_ms=50, transition="exp", alpha=1,
         db_dev=-80, to_zero_ms=1, number_of_samples=None, sonic_vector=0,
         sample_rate=None, **renamed):
    """
    Synthesize an ADSR envelope.

    ADSR (Attack, Decay, Sustain, Release) is a very traditional loudness
    envelope in sound synthesis [1].

    Parameters
    ----------
    envelope_duration : scalar
        The duration of the envelope in seconds.
    attack_ms : scalar
        The duration of the Attack in milliseconds.
    decay_ms : scalar
        The duration of the Decay in milliseconds.
    sustain_level : scalar
        The Sustain level after the Decay in decibels.
        Usually negative.
    release_ms : scalar
        The duration of the Release in milliseconds.
    transition : string
        "exp" or "exponential" for exponential transitions of amplitude
        (linear loudness); "lin" or "linear" for linear transitions of
        amplitude.
    alpha : scalar or array_like
        An index to make the exponential fade slower or faster [1]. Ignored it
        transitions="linear" or alpha=1. If it is an array_like, it should
        hold three values to be used in Attack, Decay and Release.
    db_dev : scalar or array_like
        The decibels deviation to reach before using a linear fade to reach
        zero amplitude. If it is an array_like, it should hold two values, one
        for Attack and another for Release. Ignored if trans="linear".
    to_zero_ms : scalar or array_like
        The duration in milliseconds for linearly departing from zero in the
        Attack and reaching the value of zero at the end of the Release. If it
        is an array_like, it should hold two values, one for Attack and
        another for Release. It's ignored if trans="linear".
    number_of_samples : integer or None
        The number of samples of the envelope. If given, envelope_duration
        is ignored, and zero is an envelope of no samples.
    sonic_vector : array_like
        Samples for the ADSR envelope to be applied to: mono, or of shape
        ``(nchannels, nsamples)``, every channel shaped alike. If
        supplied, envelope_duration and number_of_samples are ignored.
    sample_rate : integer or None
        The sample rate. None, the default, is the rate
        :func:`music.get_sample_rate` returns.
    **renamed
        ``attack_duration``, ``decay_duration``, ``release_duration`` and
        ``to_zero``, the names these milliseconds had before they said
        their unit among parameters in seconds. Still accepted, with a
        :class:`DeprecationWarning`.

    Returns
    -------
    as : ndarray
        A numpy array where each value is a value of the envelope for the PCM
        samples if sonic_vector is 0. If sonic_vector is input, ad is the
        sonic vector with the ADSR envelope applied to it.

    Raises
    ------
    ValueError
        If `transition` is not one of the four names above. It is checked
        here because a stage of zero length never reaches :func:`fade` or
        :func:`loud`, so an envelope of no stages accepted any name.
    TypeError
        If a keyword is neither a parameter nor one of the old names.

    See Also
    --------
    tremolo : An oscillation of loudness.
    loud : A loudness transition.
    fade : A fade in or fade out.

    Examples
    --------
    >>> write_wav_mono(note_with_vibrato() * adsr())
    >>> s = horizontal_stack(*[note_with_vibrato() *
    ...                       adsr(attack_ms=i, release_ms=j)
    ...                       for i, j in zip([6, 50, 300], [100, 10, 200])])
    >>> s = horizontal_stack(*[adsr(attack_ms=i, release_ms=j,
    ...                            sonic_vector=note_with_vibrato())
    ...                       for i, j in zip([6, 15, 100], [2, 2, 20])])
    >>> envelope = adsr(envelope_duration=440, attack_ms=10e3,
    ...                 decay_ms=0, release_ms=5e3)
    >>> stereo = localize(note(duration=1))
    >>> adsr(sonic_vector=stereo).shape == stereo.shape   # each channel
    True

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    sample_rate = resolve_rate(sample_rate)
    given = renamed_arguments('adsr', renamed, **_RENAMED)
    attack_ms = given.get('attack_ms', attack_ms)
    decay_ms = given.get('decay_ms', decay_ms)
    release_ms = given.get('release_ms', release_ms)
    to_zero_ms = given.get('to_zero_ms', to_zero_ms)
    _transition_is_linear(transition, 'transition')
    sonic_vector = as_sonic_vector(sonic_vector)
    if sonic_vector is not None:
        # The length of a channel. len() counted a stereo sound's two
        # channels as its length, built a two-sample envelope and left
        # numpy to say it could not broadcast it.
        lambda_adsr = sonic_vector.shape[-1]
    elif number_of_samples is not None:
        lambda_adsr = number_of_samples
    else:
        lambda_adsr = int(envelope_duration * sample_rate)
    lambda_a = int(attack_ms * sample_rate * 0.001)
    lambda_d = int(decay_ms * sample_rate * 0.001)
    lambda_r = int(release_ms * sample_rate * 0.001)

    # The attack, decay and release stages cannot outlast the envelope they
    # belong to: compress them proportionally when the sound is too short,
    # rather than leaving the sustain stage with a negative length.
    stages = lambda_a + lambda_d + lambda_r
    if stages > lambda_adsr:
        ratio = lambda_adsr / stages
        lambda_a = int(lambda_a * ratio)
        lambda_d = int(lambda_d * ratio)
        lambda_r = int(lambda_r * ratio)

    # An attack or release of no samples is built explicitly, because its
    # share of `to_zero_ms` is divided by its duration: an attack of 0 ms
    # divides by zero, and one too short to round to a sample asks `fade`
    # for a linear share past 100 %.
    #
    # `to_zero_ms` and the stage durations are both in milliseconds, so
    # their ratio is a fraction; `fade` reads `perc` as a percentage.
    # Passing the fraction, as the reference does, made `to_zero` a hundred
    # times smaller than it says it is, so it rounded to no samples at all
    # below about 2.3 ms at 44.1 kHz: `adsr(to_zero=1)` was byte-identical
    # to `adsr(to_zero=0)` and the envelope never reached zero at either
    # end. This is why `AD` and `ADS` are no longer sample-exact rows in
    # RECONCILIATION.md.
    if lambda_a:
        attack = fade(fade_out=0, method=transition, alpha=alpha, db=db_dev,
                      perc=100 * to_zero_ms / attack_ms,
                      number_of_samples=lambda_a)
    else:
        attack = np.array([])

    decay = loud(trans_dev=sustain_level, method=transition,
                 alpha=alpha, number_of_samples=lambda_d)

    a_s = 10 ** (sustain_level / 20.)
    sustain = np.ones(lambda_adsr - (lambda_a + lambda_r + lambda_d)) * a_s

    if lambda_r:
        release = fade(method=transition, alpha=alpha, db=db_dev,
                       perc=100 * to_zero_ms / release_ms,
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


def adsr_stereo(duration=2, attack_ms=20, decay_ms=20,
                sustain_level=-5, release_ms=50, transition="exp",
                alpha=1, db_dev=-80, to_zero_ms=1, number_of_samples=None,
                sonic_vector=0, sample_rate=None, **renamed):
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
    attack_ms : scalar
        The duration of the Attack in milliseconds.
    decay_ms : scalar
        The duration of the Decay in milliseconds.
    sustain_level : scalar
        The Sustain level after the Decay in decibels. Usually negative.
    release_ms : scalar
        The duration of the Release in milliseconds.
    transition : string
        "exp" or "exponential" for exponential transitions of amplitude
        (linear loudness), "lin" or "linear" for linear transitions of
        amplitude.
    alpha : scalar or array_like
        An index to make the exponential fade slower or faster. Ignored
        if ``transition="linear"`` or ``alpha=1``. An array_like should
        hold three values, for Attack, Decay and Release.
    db_dev : scalar or array_like
        The decibels deviation to reach before using a linear fade to
        reach zero amplitude. An array_like should hold two values, for
        Attack and Release. Ignored if ``transition="linear"``.
    to_zero_ms : scalar or array_like
        The duration in milliseconds for linearly departing from zero in
        the Attack and reaching zero at the end of the Release. An
        array_like should hold two values, for Attack and Release.
        Ignored if ``transition="linear"``.
    number_of_samples : integer or None
        The number of samples of the envelope. If given, ``duration``
        is ignored, and zero is an envelope of no samples.
    sonic_vector : array_like
        A ``(2, nsamples)`` stereo sound for the envelope to be applied
        to. If supplied, ``duration`` and ``number_of_samples`` are
        ignored.
    sample_rate : integer or None
        The sample rate. None, the default, is the rate
        :func:`music.get_sample_rate` returns.
    **renamed
        ``attack_duration``, ``decay_duration``, ``release_duration`` and
        ``to_zero``, as :func:`adsr` takes them. Still accepted, with a
        :class:`DeprecationWarning`.

    Returns
    -------
    ndarray
        A ``(2, nsamples)`` array: the envelope itself if
        ``sonic_vector`` is 0, or the sound with the envelope applied.

    Raises
    ------
    TypeError
        If a keyword is neither a parameter nor one of the old names.

    See Also
    --------
    adsr : the mono envelope, and the meaning of every argument here.

    Examples
    --------
    >>> envelope = adsr_stereo(duration=1)
    >>> envelope.shape[0]
    2

    """
    sample_rate = resolve_rate(sample_rate)
    given = renamed_arguments('adsr_stereo', renamed, **_RENAMED)
    attack_ms = given.get('attack_ms', attack_ms)
    decay_ms = given.get('decay_ms', decay_ms)
    release_ms = given.get('release_ms', release_ms)
    to_zero_ms = given.get('to_zero_ms', to_zero_ms)
    sonic_vector = as_sonic_vector(sonic_vector)
    if sonic_vector is not None:
        sonic_vector1 = sonic_vector[0]
        sonic_vector2 = sonic_vector[1]
    else:
        sonic_vector1 = 0
        sonic_vector2 = 0
    s1 = adsr(envelope_duration=duration, attack_ms=attack_ms,
              decay_ms=decay_ms, sustain_level=sustain_level,
              release_ms=release_ms, transition=transition,
              alpha=alpha, db_dev=db_dev, to_zero_ms=to_zero_ms,
              number_of_samples=number_of_samples, sonic_vector=sonic_vector1,
              sample_rate=sample_rate)
    s2 = adsr(envelope_duration=duration, attack_ms=attack_ms,
              decay_ms=decay_ms, sustain_level=sustain_level,
              release_ms=release_ms, transition=transition,
              alpha=alpha, db_dev=db_dev, to_zero_ms=to_zero_ms,
              number_of_samples=number_of_samples, sonic_vector=sonic_vector2,
              sample_rate=sample_rate)
    s = np.vstack((s1, s2))
    return s
