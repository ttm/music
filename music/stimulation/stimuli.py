"""The stimulus generators of :mod:`music.stimulation`."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.filters.localization import _localize_positions
from ..core.synths.noises import noise
from ..core.synths.notes import note
from ..utils import WAVEFORM_SINE, _integrate_phase

__all__ = [
    'amplitude_modulation',
    'binaural_beats',
    'frequency_modulation',
    'isochronic_tones',
    'modulated_noise',
    'monaural_beats',
    'spatial_motion',
]


def _sample_count(duration, number_of_samples, sample_rate):
    """The length to render, in samples.

    ``number_of_samples`` wins over ``duration`` when given, which is the
    convention every synthesis routine in this package follows.
    """
    if number_of_samples:
        return int(number_of_samples)
    return int(duration * sample_rate)


def _oscillator(waveform_table, freq, count, sample_rate):
    """One period-indexed lookup of ``waveform_table`` at ``freq``.

    The modulators below are read from a table rather than from
    ``np.sin`` so that a caller can drive a modulation with any waveform
    the package can build, as :func:`music.note_with_fm` already allows
    for its own modulator.
    """
    table = np.asarray(waveform_table)
    samples = np.arange(count)
    length = len(table)
    index = (samples * freq * length / sample_rate).astype(np.int64)
    return table[index % length]


def binaural_beats(carrier_freq: float = 200.0, beat_freq: float = 10.0,
                   duration: float = 2.0,
                   waveform_table: ArrayLike = WAVEFORM_SINE,
                   number_of_samples: int = 0,
                   sample_rate: int = 44100) -> NDArray[np.float64]:
    """Synthesize a binaural beat: ``sstim-v:techBinauralBeats``.

    Two sine tones at slightly different frequencies presented
    dichotically, one per ear, producing a perceived beat at their
    difference frequency that arises centrally rather than in the air
    [2]_. The two carriers are placed symmetrically about
    ``carrier_freq``, at ``carrier_freq -/+ beat_freq / 2``, so the
    perceived pitch stays where the caller put it.

    Parameters
    ----------
    carrier_freq : scalar
        The centre frequency of the two carriers, in Hertz.
    beat_freq : scalar
        The difference between the two carriers, in Hertz, which is the
        rate of the perceived beat.
    duration : scalar
        The duration in seconds.
    waveform_table : array_like
        The table the carriers are looked up in.
    number_of_samples : integer
        The number of samples of the sound, taken instead of
        ``duration`` when it is given.
    sample_rate : integer
        The sampling frequency in Hertz.

    Returns
    -------
    ndarray
        A ``(2, n)`` stereo array: the lower carrier in the left
        channel, the higher in the right.

    Notes
    -----
    **The beat is not in the signal.** SSTIM records this technique as
    producing a perceptually constructed rather than a physically
    present modulation. Each channel here is a steady tone, and a
    spectrum of either one alone does not contain the beat: it is a
    neural construct from dichotic presentation.

    Channel separation is therefore the mechanism, not a packaging
    detail, and the stimulus only works over headphones. Summing the two
    channels does not preserve it -- it produces a real envelope at the
    difference frequency, numerically identical to
    :func:`monaural_beats`, which is a different technique with a
    different mechanism and a different evidence base. Anything that
    downmixes this output has silently substituted one for the other.

    See Also
    --------
    monaural_beats : the same two tones in one channel, where the beat
                     is physically present.

    Examples
    --------
    >>> stimulus = binaural_beats(carrier_freq=200, beat_freq=10)
    >>> stimulus.shape[0]
    2

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853
           (2017)
    .. [2] SSTIM, ``techBinauralBeats``.
           https://w3id.org/sstim/vocab#techBinauralBeats

    """
    count = _sample_count(duration, number_of_samples, sample_rate)
    left = note(freq=carrier_freq - beat_freq / 2,
                waveform_table=waveform_table,
                number_of_samples=count, sample_rate=sample_rate)
    right = note(freq=carrier_freq + beat_freq / 2,
                 waveform_table=waveform_table,
                 number_of_samples=count, sample_rate=sample_rate)
    return np.vstack((left, right))


def monaural_beats(carrier_freq: float = 200.0, beat_freq: float = 10.0,
                   duration: float = 2.0,
                   waveform_table: ArrayLike = WAVEFORM_SINE,
                   number_of_samples: int = 0,
                   sample_rate: int = 44100) -> NDArray[np.float64]:
    """Synthesize a monaural beat: ``sstim-v:techMonauralBeats``.

    Two close-frequency tones summed in a single channel, producing a
    physically present amplitude beat at their difference frequency
    [2]_. Distinct from a binaural beat, where the beat is a neural
    construct from dichotic presentation.

    Parameters
    ----------
    carrier_freq : scalar
        The centre frequency of the two tones, in Hertz.
    beat_freq : scalar
        The difference between them, in Hertz, which is the rate of the
        resulting amplitude beat.
    duration : scalar
        The duration in seconds.
    waveform_table : array_like
        The table the two tones are looked up in.
    number_of_samples : integer
        The number of samples of the sound, taken instead of
        ``duration`` when it is given.
    sample_rate : integer
        The sampling frequency in Hertz.

    Returns
    -------
    ndarray
        A mono sequence of PCM samples, the mean of the two tones.

    Notes
    -----
    **The beat is in the signal.** Unlike :func:`binaural_beats`, the
    modulation is physically present: the sum of two tones a few Hertz
    apart has an envelope at their difference frequency, which a
    spectrum of the rendered audio shows and which survives being played
    through one loudspeaker.

    See Also
    --------
    binaural_beats : the same two tones one per ear, where the beat is
                     perceptual only.
    amplitude_modulation : an envelope imposed on one carrier rather
                           than arising from two.

    Examples
    --------
    >>> stimulus = monaural_beats(carrier_freq=200, beat_freq=10)
    >>> stimulus.ndim
    1

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853
           (2017)
    .. [2] SSTIM, ``techMonauralBeats``.
           https://w3id.org/sstim/vocab#techMonauralBeats

    """
    count = _sample_count(duration, number_of_samples, sample_rate)
    lower = note(freq=carrier_freq - beat_freq / 2,
                 waveform_table=waveform_table,
                 number_of_samples=count, sample_rate=sample_rate)
    upper = note(freq=carrier_freq + beat_freq / 2,
                 waveform_table=waveform_table,
                 number_of_samples=count, sample_rate=sample_rate)
    return (lower + upper) / 2


def isochronic_tones(carrier_freq: float = 200.0, pulse_rate: float = 10.0,
                     duty_cycle: float = 0.5, duration: float = 2.0,
                     ramp_duration: float = 0.0,
                     waveform_table: ArrayLike = WAVEFORM_SINE,
                     number_of_samples: int = 0,
                     sample_rate: int = 44100) -> NDArray[np.float64]:
    """Synthesize an isochronic tone train: ``sstim-v:techIsochronicTones``.

    A single tone switched on and off at evenly spaced intervals -- a
    gated pulse train -- at a target rate; a monaural, single-channel
    entrainment stimulus [2]_.

    Parameters
    ----------
    carrier_freq : scalar
        The frequency of the gated tone, in Hertz.
    pulse_rate : scalar
        How many times per second the tone is switched on, in Hertz.
    duty_cycle : scalar
        The fraction of each pulse period the tone is on, in ``(0, 1]``.
    duration : scalar
        The duration in seconds.
    ramp_duration : scalar
        A linear fade in and out applied at each pulse edge, in seconds.
        Zero, the default, gates abruptly, which is the literal
        technique; see the note below on what that costs. A ramp longer
        than half the sounding part of a pulse leaves the two ramps
        overlapping, and the pulse becomes a triangle that never reaches
        full amplitude rather than an error.
    waveform_table : array_like
        The table the tone is looked up in.
    number_of_samples : integer
        The number of samples of the sound, taken instead of
        ``duration`` when it is given.
    sample_rate : integer
        The sampling frequency in Hertz.

    Returns
    -------
    ndarray
        A mono sequence of PCM samples.

    Raises
    ------
    ValueError
        If ``duty_cycle`` is outside ``(0, 1]``. A duty cycle of zero is
        silence and one above one is not a gate, and both are more
        likely to be a mistake than an intention.
    ValueError
        If ``ramp_duration`` is negative, which would otherwise scale
        every pulse by a clipped negative and return silence.

    Notes
    -----
    **The modulation is physically present**, and at ``pulse_rate``
    rather than at ``carrier_freq``.

    An abrupt gate is what the technique names, but it is also a step
    discontinuity twice per pulse, and each step spreads energy across
    the spectrum -- audible as a click, and visible in a recording as
    broadband splatter that is not part of the intended stimulus. Set
    ``ramp_duration`` to a few milliseconds to taper the edges when the
    stimulus is going to be measured or listened to for any length of
    time.

    Examples
    --------
    >>> stimulus = isochronic_tones(pulse_rate=10, duty_cycle=0.5)
    >>> stimulus.ndim
    1

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853
           (2017)
    .. [2] SSTIM, ``techIsochronicTones``.
           https://w3id.org/sstim/vocab#techIsochronicTones

    """
    if not 0 < duty_cycle <= 1:
        raise ValueError(
            f"duty_cycle must be in (0, 1], got {duty_cycle}")
    if ramp_duration < 0:
        raise ValueError(
            f"ramp_duration cannot be negative, got {ramp_duration}")
    count = _sample_count(duration, number_of_samples, sample_rate)
    tone = note(freq=carrier_freq, waveform_table=waveform_table,
                number_of_samples=count, sample_rate=sample_rate)

    samples = np.arange(count)
    phase = (samples * pulse_rate / sample_rate) % 1.0
    gate = (phase < duty_cycle).astype(np.float64)

    if ramp_duration:
        ramp = int(ramp_duration * sample_rate)
        if ramp:
            # Distance in samples to the nearest edge of the pulse, so
            # one expression tapers both the rise and the fall.
            period = sample_rate / pulse_rate
            into = phase * period
            out_of = (duty_cycle - phase) * period
            edge = np.minimum(into, out_of)
            gate *= np.clip(edge / ramp, 0.0, 1.0)
    return tone * gate


def amplitude_modulation(
        carrier_freq: float = 200.0, modulation_freq: float = 10.0,
        modulation_depth: float = 1.0, duration: float = 2.0,
        waveform_table: ArrayLike = WAVEFORM_SINE,
        modulation_waveform_table: ArrayLike = WAVEFORM_SINE,
        number_of_samples: int = 0,
        sample_rate: int = 44100) -> NDArray[np.float64]:
    """Sinusoidally modulate a carrier: ``sstim-v:techAmplitudeModulation``.

    A carrier tone whose amplitude is sinusoidally modulated at a target
    rate; the canonical stimulus for evoking auditory steady-state
    responses at the modulation frequency [2]_.

    Parameters
    ----------
    carrier_freq : scalar
        The frequency of the carrier, in Hertz.
    modulation_freq : scalar
        The rate at which its amplitude is modulated, in Hertz.
    modulation_depth : scalar
        How deep the modulation goes, in ``[0, 1]``. At 1 the envelope
        reaches zero once per modulation period; at 0 the carrier is
        left alone.
    duration : scalar
        The duration in seconds.
    waveform_table : array_like
        The table the carrier is looked up in.
    modulation_waveform_table : array_like
        The table the modulator is looked up in, so the modulation need
        not be sinusoidal.
    number_of_samples : integer
        The number of samples of the sound, taken instead of
        ``duration`` when it is given.
    sample_rate : integer
        The sampling frequency in Hertz.

    Returns
    -------
    ndarray
        A mono sequence of PCM samples.

    Raises
    ------
    ValueError
        If ``modulation_depth`` is outside ``[0, 1]``, where the
        envelope would go negative and invert the carrier's phase
        rather than deepening the modulation.

    Notes
    -----
    **The modulation is physically present**, and it is the modulation
    rate rather than the carrier that sets the frequency of the evoked
    response, which is the whole point of the stimulus.

    See Also
    --------
    monaural_beats : an equivalent envelope arising from two tones
                     rather than imposed on one.

    Examples
    --------
    >>> stimulus = amplitude_modulation(carrier_freq=200,
    ...                                 modulation_freq=40)
    >>> stimulus.ndim
    1

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853
           (2017)
    .. [2] SSTIM, ``techAmplitudeModulation``.
           https://w3id.org/sstim/vocab#techAmplitudeModulation

    """
    if not 0 <= modulation_depth <= 1:
        raise ValueError(
            f"modulation_depth must be in [0, 1], got {modulation_depth}")
    count = _sample_count(duration, number_of_samples, sample_rate)
    tone = note(freq=carrier_freq, waveform_table=waveform_table,
                number_of_samples=count, sample_rate=sample_rate)
    modulator = _oscillator(modulation_waveform_table, modulation_freq,
                            count, sample_rate)
    envelope = 1 - modulation_depth * (1 - modulator) / 2
    return tone * envelope


def frequency_modulation(
        carrier_freq: float = 200.0, modulation_freq: float = 10.0,
        frequency_deviation: float = 20.0, duration: float = 2.0,
        waveform_table: ArrayLike = WAVEFORM_SINE,
        modulation_waveform_table: ArrayLike = WAVEFORM_SINE,
        number_of_samples: int = 0,
        sample_rate: int = 44100) -> NDArray[np.float64]:
    """Sweep a carrier's pitch: ``sstim-v:techFrequencyModulation``.

    A tone whose pitch sweeps over time. SSTIM types this as a generic
    technique because its status depends on rate: fast modulation can
    evoke frequency-following responses, while slow, breathing-rate
    modulation engages autonomic rather than entrainment pathways [2]_.

    Parameters
    ----------
    carrier_freq : scalar
        The centre frequency of the carrier, in Hertz.
    modulation_freq : scalar
        The rate at which the pitch sweeps, in Hertz.
    frequency_deviation : scalar
        The peak departure from ``carrier_freq``, in Hertz. Stated in
        Hertz rather than in semitones, which is the convention this
        literature uses; :func:`music.note_with_vibrato` takes the
        musical form of the same idea in semitones.
    duration : scalar
        The duration in seconds.
    waveform_table : array_like
        The table the carrier is looked up in.
    modulation_waveform_table : array_like
        The table the modulator is looked up in.
    number_of_samples : integer
        The number of samples of the sound, taken instead of
        ``duration`` when it is given.
    sample_rate : integer
        The sampling frequency in Hertz.

    Returns
    -------
    ndarray
        A mono sequence of PCM samples.

    Notes
    -----
    **The modulation is physically present.**

    The instantaneous frequency is computed per sample and integrated
    into the lookup index, rather than the modulation being applied to
    an already-rendered tone. This is the package's model throughout,
    and it is why the rendered sound matches the mathematics that
    describes it rather than approximating it.

    See Also
    --------
    music.note_with_vibrato : the same modulation expressed musically,
                              in semitones.
    music.note_with_fm : FM synthesis, where the modulator is at audio
                         rate and the point is timbre.

    Examples
    --------
    >>> stimulus = frequency_modulation(carrier_freq=200,
    ...                                 modulation_freq=0.1)
    >>> stimulus.ndim
    1

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853
           (2017)
    .. [2] SSTIM, ``techFrequencyModulation``.
           https://w3id.org/sstim/vocab#techFrequencyModulation

    """
    count = _sample_count(duration, number_of_samples, sample_rate)
    modulator = _oscillator(modulation_waveform_table, modulation_freq,
                            count, sample_rate)
    instantaneous = carrier_freq + frequency_deviation * modulator

    table = np.asarray(waveform_table)
    length = len(table)
    index = _integrate_phase(instantaneous * length / sample_rate,
                             length).astype(np.int64)
    return table[index % length]


def modulated_noise(noise_type: str | float = 'pink',
                    modulation_freq: float = 10.0,
                    modulation_depth: float = 1.0, duration: float = 2.0,
                    min_freq: float = 15.0, max_freq: float = 15000.0,
                    modulation_waveform_table: ArrayLike = WAVEFORM_SINE,
                    number_of_samples: int = 0,
                    sample_rate: int = 44100) -> NDArray[np.float64]:
    """Modulate a band of noise: ``sstim-v:techBroadbandNoise``.

    Broadband noise of a chosen colour, optionally amplitude-modulated at
    a target rate. Unmodulated it is the continuous broadband stimulus
    SSTIM catalogues as ``techBroadbandNoise`` [2]_, the vehicle for
    stochastic resonance and for masking. Modulated it is also
    ``sstim-v:techAmplitudeModulation`` [3]_, whose definition names a
    carrier tone *or noise*; the noise carries no pitch of its own, so
    the rate is all the listener has to lock to.

    Parameters
    ----------
    noise_type : string or scalar
        The colour of the noise, as :func:`music.noise` takes it: one of
        ``"white"``, ``"pink"``, ``"brown"``, ``"blue"``, ``"violet"``,
        ``"black"``, or a number of decibels of gain per octave. The
        default is pink, which has equal energy per octave and is the
        usual choice where a spectrum is meant to be even to the ear
        rather than even in Hertz.
    modulation_freq : scalar
        The rate at which the amplitude is modulated, in Hertz. Zero
        leaves the noise unmodulated, which is the plain broadband
        stimulus.
    modulation_depth : scalar
        How deep the modulation goes, in ``[0, 1]``, as in
        :func:`amplitude_modulation`.
    duration : scalar
        The duration in seconds.
    min_freq : scalar
        The lowest frequency present, in Hertz.
    max_freq : scalar
        The highest frequency present, in Hertz.
    modulation_waveform_table : array_like
        The table the modulator is looked up in, so the modulation need
        not be sinusoidal.
    number_of_samples : integer
        The number of samples of the sound, taken instead of
        ``duration`` when it is given.
    sample_rate : integer
        The sampling frequency in Hertz.

    Returns
    -------
    ndarray
        A mono sequence of PCM samples.

    Raises
    ------
    ValueError
        If ``modulation_depth`` is outside ``[0, 1]``, where the
        envelope would go negative and invert the noise rather than
        deepening the modulation.
    ValueError
        If ``modulation_freq`` is negative. A negative rate is the same
        modulation as its absolute value for a symmetric table and a
        different one otherwise, which makes it a mistake rather than a
        choice.

    Notes
    -----
    **The modulation is physically present** when there is one. With
    ``modulation_freq`` at zero there is no modulation to be present or
    absent, and the stimulus is a spectral one rather than a temporal
    one: SSTIM types ``techBroadbandNoise`` as a non-entrainment
    technique for exactly that reason, and nothing here entrains until
    a rate is given.

    The noise is synthesized once at full length and then shaped, rather
    than being built per modulation period, so its spectrum is the one
    :func:`music.noise` describes and the modulation does not repeat a
    single block of samples.

    See Also
    --------
    amplitude_modulation : the same envelope on a pitched carrier.
    music.noise : the underlying colored noise, unshaped.

    Examples
    --------
    >>> stimulus = modulated_noise(noise_type='pink', modulation_freq=10)
    >>> stimulus.ndim
    1

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853
           (2017)
    .. [2] SSTIM, ``techBroadbandNoise``.
           https://w3id.org/sstim/vocab#techBroadbandNoise
    .. [3] SSTIM, ``techAmplitudeModulation``.
           https://w3id.org/sstim/vocab#techAmplitudeModulation

    """
    if not 0 <= modulation_depth <= 1:
        raise ValueError(
            f"modulation_depth must be in [0, 1], got {modulation_depth}")
    if modulation_freq < 0:
        raise ValueError(
            f"modulation_freq cannot be negative, got {modulation_freq}")
    count = _sample_count(duration, number_of_samples, sample_rate)
    bed = noise(noise_type=noise_type, min_freq=min_freq,
                max_freq=max_freq, number_of_samples=count,
                sample_rate=sample_rate)
    if not modulation_freq:
        return np.asarray(bed, dtype=np.float64)
    modulator = _oscillator(modulation_waveform_table, modulation_freq,
                            count, sample_rate)
    envelope = 1 - modulation_depth * (1 - modulator) / 2
    return bed * envelope


def spatial_motion(carrier_freq: float = 200.0, motion_rate: float = 0.2,
                   duration: float = 2.0, theta1: float = 180.0,
                   theta2: float = 0.0, dist: float = 0.1,
                   sonic_vector: ArrayLike | None = None,
                   waveform_table: ArrayLike = WAVEFORM_SINE,
                   zeta: float = 0.215, air_temp: float = 20.0,
                   number_of_samples: int = 0,
                   sample_rate: int = 44100) -> NDArray[np.float64]:
    """Orbit a source around the listener: ``sstim-v:techSpatialAuditory``.

    A sound whose apparent position sweeps back and forth between two
    azimuths at a chosen rate. SSTIM catalogues structured spatial
    trajectories -- orbiting sources, lateral sweeps, spatial
    alternation -- as ``techSpatialAuditory``, and distinguishes them
    from simple left/right crossfades [2]_. This is on the right side of
    that line: the two ears receive the interaural time and intensity
    differences the geometry implies, computed per sample, not a pan
    between two gains.

    Parameters
    ----------
    carrier_freq : scalar
        The frequency of the orbiting tone, in Hertz. Ignored when
        ``sonic_vector`` is given.
    motion_rate : scalar
        How many round trips the source makes per second, in Hertz. One
        cycle is a full there-and-back, so the source passes each
        endpoint once per cycle.
    duration : scalar
        The duration in seconds.
    theta1 : scalar
        The azimuth the motion starts from, in degrees, measured from
        the ear axis as everywhere else in this package: 0 is the right
        ear's side, 180 the left. The default is a full crossing of the
        head.
    theta2 : scalar
        The azimuth it travels to before returning, in degrees.
    dist : scalar
        The radius of the orbit, in meters.
    sonic_vector : array_like or None
        A mono sound to move instead of a synthesized tone, so a noise
        bed or a rendered stimulus can be given a trajectory. When it is
        given it sets the length, and ``duration``,
        ``number_of_samples``, ``carrier_freq`` and ``waveform_table``
        are not used.
    waveform_table : array_like
        The table the tone is looked up in.
    zeta : scalar
        The distance between the ears in meters.
    air_temp : scalar
        The temperature in Celsius used for calculating the speed of
        sound.
    number_of_samples : integer
        The number of samples of the sound, taken instead of
        ``duration`` when it is given.
    sample_rate : integer
        The sampling frequency in Hertz.

    Returns
    -------
    ndarray
        A ``(2, nsamples)`` array of stereo PCM samples.

    Raises
    ------
    ValueError
        If ``motion_rate`` is negative, which would reverse the
        trajectory rather than slow it.

    Notes
    -----
    **The motion is physically present**: the cues are in the signal,
    and a recording of the output carries them.

    The source travels along the arc at constant radius -- an orbit --
    where :func:`music.localize_linear` travels the straight line
    between its two endpoints, which is what makes that one linear. The
    azimuth moves at a constant rate, so the trajectory is triangular in
    angle and the source neither pauses nor lurches at the turns.

    **There is no head-related transfer function here**, as there is
    none anywhere in this package. Interaural time and intensity
    differences place a source on the left-right axis and nothing else:
    elevation, and whether the source is in front of or behind the
    listener, are cues an HRTF carries and this does not. An orbit
    rendered here is heard as a lateral sweep rather than as a circle
    around the head, and an orbit between 90 and -90 degrees -- ahead
    and behind -- renders no movement whatsoever, because those two
    positions differ only in the cue that is missing.

    See Also
    --------
    music.localize_linear : one straight pass, the primitive underneath.
    music.note_with_doppler : a moving source synthesized rather than
                              filtered, so it also shifts pitch.

    Examples
    --------
    >>> stimulus = spatial_motion(carrier_freq=200, motion_rate=0.5)
    >>> stimulus.shape[0]
    2

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853
           (2017)
    .. [2] SSTIM, ``techSpatialAuditory``.
           https://w3id.org/sstim/vocab#techSpatialAuditory

    """
    if motion_rate < 0:
        raise ValueError(
            f"motion_rate cannot be negative, got {motion_rate}")
    if sonic_vector is None:
        count = _sample_count(duration, number_of_samples, sample_rate)
        source = note(freq=carrier_freq, waveform_table=waveform_table,
                      number_of_samples=count, sample_rate=sample_rate)
    else:
        source = np.asarray(sonic_vector, dtype=np.float64)
        count = len(source)
    if count == 0:
        return np.zeros((2, 0))

    # The azimuth is triangular in time: one cycle runs theta1 to theta2
    # and back, so `progress` rises from 0 to 1 and falls again.
    samples = np.arange(count)
    phase = (samples * motion_rate / sample_rate) % 1.0
    progress = 1 - np.abs(1 - 2 * phase)
    theta = np.radians(theta1 + (theta2 - theta1) * progress)

    return _localize_positions(source, np.cos(theta) * dist,
                               np.sin(theta) * dist, zeta, air_temp,
                               sample_rate)
