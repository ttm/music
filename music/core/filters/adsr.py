import numpy as np
from .fade import fade
from .loud import loud
from ..synths.notes import note_with_vibrato


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
    if type(sonic_vector) in (np.ndarray, list):
        lambda_adsr = len(sonic_vector)
    elif number_of_samples:
        lambda_adsr = number_of_samples
    else:
        lambda_adsr = int(envelope_duration * sample_rate)
    lambda_a = int(attack_duration * sample_rate * 0.001)
    lambda_d = int(decay_duration * sample_rate * 0.001)
    lambda_r = int(release_duration * sample_rate * 0.001)

    perc = to_zero / attack_duration
    attack_duration = fade(fade_out=0, method=transition, alpha=alpha,
                           db=db_dev, perc=perc, number_of_samples=lambda_a)

    decay_duration = loud(trans_dev=sustain_level, method=transition,
                          alpha=alpha, number_of_samples=lambda_d)

    a_s = 10 ** (sustain_level / 20.)
    sustain_level = np.ones(lambda_adsr -
                            (lambda_a + lambda_r + lambda_d)) * a_s

    perc = to_zero / release_duration
    release_duration = fade(method=transition, alpha=alpha, db=db_dev,
                            perc=perc, number_of_samples=lambda_r) * a_s

    ad = np.hstack((attack_duration, decay_duration, sustain_level,
                    release_duration))
    if type(sonic_vector) in (np.ndarray, list):
        return sonic_vector * ad
    else:
        return ad


def adsr_vibrato(note_dict={}, adsr_dict={}):
    """
    Creates a note with a vibrato and an ADSR envelope.

    Check the adsr and the note_with_vibrato functions.

    """
    return adsr(sonic_vector=note_with_vibrato(**note_dict), **adsr_dict)


def adsr_stereo(duration=2, attack_duration=20, decay_duration=20,
                sustain_level=-5, release_duration=50, transition="exp",
                alpha=1, db_dev=-80, to_zero=1, number_of_samples=0,
                sonic_vector=0, sample_rate=44100):
    """
    A shorthand to make an ADSR envelope for a stereo sound.

    See adsr() for more information.

    """
    if type(sonic_vector) in (np.ndarray, list):
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
