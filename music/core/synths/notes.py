""" Module for the synthesis of notes and notes with some
    effects applied to them.
"""
import numpy as np
from ...utils import WAVEFORM_SINE, WAVEFORM_TRIANGULAR
from ..filters import adsr


def note(freq=220, duration=2, waveform_table=WAVEFORM_TRIANGULAR,
         number_of_samples=0, sample_rate=44100):
    """
    Synthesize a basic musical note.

    Parameters
    ----------
    freq : scalar
        The frequency of the note in Hertz.
    duration : scalar
        The duration of the note in seconds.
    waveform_table : array_like
        The table with the waveform to synthesize the sound.
    number_of_samples : integer
        The number of samples in the sound.
        If not 0, d is ignored.
    sample_rate : integer
        The sample rate.

    Returns
    -------
    result : ndarray
        A numpy array where each value is a PCM sample of the note.

    See Also
    --------
    V : A note with vibrato.
    T : A tremolo envelope.

    Examples
    --------
    >>> write_wav_mono(note())  # writes a WAV file of a note
    >>> s = H([note(i, j) for i, j in zip([200, 500, 100], [2, 1, 2])])
    >>> s2 = note(440, 1.5, waveform_table=WAVEFORM_SAWTOOTH)

    Notes
    -----
    In the MASS framework implementation, for a sound with a vibrato (or FM)
    to be synthesized using LUT, the vibrato pattern is considered when
    performing the lookup calculations.

    The tremolo and AM patterns are implemented as separate amplitude
    envelopes.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    waveform_table = np.array(waveform_table)
    if not number_of_samples:
        number_of_samples = int(duration * sample_rate)
    samples = np.arange(number_of_samples)
    waveform_table_length = len(waveform_table)

    gamma = (samples * freq * waveform_table_length /
             sample_rate).astype(np.int64)
    result = waveform_table[gamma % waveform_table_length]
    return result


def note_with_doppler(freq=220, duration=2, waveform_table=WAVEFORM_TRIANGULAR,
                      x=[-10, 10], y=[1, 1], stereo=True, zeta=0.215,
                      air_temp=20, number_of_samples=0, sample_rate=44100):
    """
    A simple note with a transition of localization and resulting Doppler
    effect.

    Parameters
    ----------
    freq : scalar
        The frequency of the note in Hertz.
    duration : scalar
        The duration of the note in seconds.
    waveform_table : array_like
        The table with the waveform to synthesize the sound.
    x : iterable of scalars
        The starting and ending x positions.
    y : iterable of scalars
        The starting and ending y positions.
    stereo : boolean
        If True, returns a (2, nsamples) array representing
        a stereo sound. Else it returns a simple array
        for a mono sound.
    zeta : float
        TODO:
    air_temp : scalar
        The air temperature in Celsius.
        (Used to calculate the acoustic velocity.)
    number_of_samples : integer
        The number of samples in the sound.
        If not 0, d is ignored.
    sample_rate : integer
        The sample rate.

    Returns
    -------
    s : ndarray
        The PCM samples of the resulting sound.

    See Also
    --------
    D_ : a note with arbitrary vibratos, transitions of pitch and transitions
         of localization.
    PV_ : a note with an arbitrary sequence of pitch transition and a
          meta-vibrato.

    Examples
    --------
    >>> write_wav_stereo(note_with_doppler())
    >>> write_wav_mono(T()*note_with_doppler(stereo=False))

    Notes
    -----

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    waveform_table = np.array(waveform_table)
    if not number_of_samples:
        number_of_samples = int(duration * sample_rate)
    #  samples = np.arange(number_of_samples)
    length = len(waveform_table)
    speed = 331.3 + .606 * air_temp

    x = x[0] + (x[1] - x[0]) * np.arange(number_of_samples + 1) / \
        number_of_samples
    y = y[0] + (y[1] - y[0]) * np.arange(number_of_samples + 1) / \
        number_of_samples
    if stereo:
        dl = np.sqrt((x + zeta / 2) ** 2 + y ** 2)
        dr = np.sqrt((x - zeta / 2) ** 2 + y ** 2)
        iid_al = 1 / dl
        iid_ar = 1 / dr

        vsl = sample_rate * (dl[1:] - dl[:-1])
        vsr = sample_rate * (dr[1:] - dr[:-1])
        fl = freq * speed / (speed + vsl)
        fr = freq * speed / (speed + vsr)

        gamma = np.cumsum(fl * length / sample_rate).astype(np.int64)
        sl = waveform_table[gamma % length] * iid_al[:-1]

        gamma = np.cumsum(fr * length / sample_rate).astype(np.int64)
        sr = waveform_table[gamma % length] * iid_ar[:-1]

        itd0 = (dl[0] - dr[0]) / speed
        lambda_itd = itd0 * sample_rate

        if x[0] > 0:
            tl = np.hstack((np.zeros(int(lambda_itd)), sl))
            tr = np.hstack((sr, np.zeros(int(lambda_itd))))
        else:
            tl = np.hstack((sl, np.zeros(-int(lambda_itd))))
            tr = np.hstack((np.zeros(-int(lambda_itd)), sr))
        result = np.vstack((tl, tr))
    else:
        duration = np.sqrt(x ** 2 + y ** 2)
        iid = 1 / duration

        # velocities at each point
        vs = sample_rate * (duration[1:] - duration[:-1])
        f_ = freq * speed / (speed + vs)

        gamma = np.cumsum(f_ * length / sample_rate).astype(np.int64)
        result = waveform_table[gamma % length] * iid[:-1]
    return result


def note_with_fm(freq=220, duration=2, fm=100, max_fm_deviation=2,
                 waveform_table=WAVEFORM_TRIANGULAR,
                 fm_waveform_table=WAVEFORM_SINE,
                 number_of_samples=0, sample_rate=44100):
    """
    Synthesize a musical note with FM synthesis.

    Set fm=0 or max_fm_deviation=0 (or use synth_note()) for a note without FM.
    A FM is a linear oscillatory pattern of frequency [1].

    Parameters
    ----------
    freq : scalar
        The frequency of the note in Hertz.
    duration : scalar
        The duration of the note in seconds.
    fm : scalar
        The frequency of the modulator in Hertz.
    max_fm_deviation : scalar
        The maximum deviation of frequency in the modulator in Hertz.
    waveform_table : array_like
        The table with the waveform for the carrier.
    fm_waveform_table : array_like
        The table with the waveform for the modulator.
    number_of_samples : integer
        The number of samples in the sound.
        If supplied, d is ignored.
    sample_rate : integer
        The sample rate.

    Returns
    -------
    result : ndarray
        A numpy array where each value is a PCM sample of the note.

    See Also
    --------
    N : A basic musical note without vibrato.
    V : A musical note with an oscillation of pitch.
    T : A tremolo, an oscillation of loudness.
    AM : A linear oscillation of amplitude (not linear loudness).

    Examples
    --------
    >>> write_wav_mono(note_with_fm())  # writes a WAV file of a note
    >>> sonic_vector = H([note_with_fm(i, j) for i, j in zip([200, 500, 100],
                                                             [2, 1, 2])])
    >>> s2 = note_with_fm(440, 1.5, 600, 10)

    Notes
    -----
    In the MASS framework implementation,
    for a sound with a vibrato (or FM) to be synthesized using LUT,
    the vibrato (or FM)
    pattern is considered when performing the lookup calculations.

    The tremolo and AM patterns are implemented as separate amplitude
    envelopes.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    waveform_table = np.array(waveform_table)
    fm_waveform_table = np.array(fm_waveform_table)
    if number_of_samples:
        lambda_fm = number_of_samples
    else:
        lambda_fm = int(sample_rate * duration)
    samples = np.arange(lambda_fm)

    fm_waveform_table_length = len(fm_waveform_table)
    gamma_m = (samples * fm * fm_waveform_table_length /
               sample_rate).astype(np.int64)  # LUT indexes
    # values of the oscillatory pattern at each sample
    t_m = fm_waveform_table[gamma_m % fm_waveform_table_length]

    # frequency in Hz at each sample
    f = freq + t_m * max_fm_deviation
    waveform_table_length = len(waveform_table)
    # shift in table between each sample
    d_gamma = f * (waveform_table_length / sample_rate)
    gamma = np.cumsum(d_gamma).astype(np.int64)  # total shift at each sample
    # final sample lookup
    result = waveform_table[gamma % waveform_table_length]
    return result


def note_with_phase(freq=220, duration=2, phase=0,
                    waveform_table=WAVEFORM_TRIANGULAR,
                    number_of_samples=0, sample_rate=44100):
    """
    Synthesize a basic musical note with a phase.

    It's useful in more complex synthesis routines. For synthesizing a musical
    note directly, you probably want to use note() and disconsider the phase.

    Parameters
    ----------
    freq : scalar
        The frequency of the note in Hertz.
    duration : scalar
        The duration of the note in seconds.
    phase : scalar
        The phase of the wave in radians.
    waveform_table : array_like
        The table with the waveform to synthesize the sound.
    number_of_samples : integer
        The number of samples in the sound.
        If not 0, d is ignored.
    sample_rate : integer
        The sample rate.

    Returns
    -------
    result : ndarray
        A numpy array where each value is a PCM sample of the note.

    See Also
    --------
    synth_note : A basic note.
    V : A note with vibrato.
    T : A tremolo envelope.

    Examples
    --------
    >>> write_wav_mono(synth_note_with_phase())  # writes a WAV file of a note
    >>> s = H([synth_note_with_phase(i, j) for i, j in zip([200, 500, 100],
                                                           [2, 1, 2])])
    >>> s2 = synth_note_with_phase(440, 1.5, waveform_table=Sa)

    Notes
    -----
    In the MASS framework implementation, for a sound with a vibrato (or FM)
    to be synthesized using LUT, the vibrato pattern is considered when
    performing the lookup calculations.

    The tremolo and AM patterns are implemented as separate amplitude
    envelopes.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    waveform_table = np.array(waveform_table)
    if not number_of_samples:
        number_of_samples = int(duration * sample_rate)
    samples = np.arange(number_of_samples)
    waveform_table_length = len(waveform_table)
    i0 = phase * waveform_table_length / (2 * np.pi)
    gamma = (i0 + samples * freq * waveform_table_length /
             sample_rate).astype(np.int64)
    result = waveform_table[gamma % waveform_table_length]
    return result


def note_with_glissando(start_freq=220, end_freq=440, duration=2, alpha=1,
                        waveform_table=WAVEFORM_SINE, method="exp",
                        number_of_samples=0, sample_rate=44100):
    """
    A note with a pitch transition: a glissando.

    Parameters
    ----------
    start_freq : scalar
        The starting frequency.
    end_freq : scalar
        The final frequency.
    duration : scalar
        The duration of the sound in seconds.
    alpha : scalar
        An index to begin the transition faster or slower.
        If alpha != 1, the transition is not of linear pitch.
    waveform_table : array_like
        The table with the waveform to synthesize the sound.
    number_of_samples : integer
        The number of samples of the sound.
        If supplied, d is not used.
    method : string
        "exp" for an exponential transition of frequency
        (linear pitch).
        "lin" for a linear transition of amplitude.
    sample_rate : integer
        The sample rate.

    Returns
    -------
    s : ndarray
        A numpy array where each value is a PCM sample of the sound.

    See Also
    --------
    N : A basic musical note without vibrato or pitch transition.
    V : A musical note with an oscillation of pitch.
    T : A tremolo, an oscillation of loudness.
    L : A transition of loudness.
    F : Fade in or out.

    Examples
    --------
    >>> write_wav_mono(note_with_pitch())  # writes file with a glissando
    >>> s = H([note_with_pitch(i, j) for i, j in zip([220, 440, 4000],
                                                     [440, 220, 220])])
    >>> write_wav_mono(s)  # writes a file with glissandi

    """
    waveform_table = np.array(waveform_table)
    if number_of_samples:
        lambda_p = number_of_samples
    else:
        lambda_p = int(sample_rate * duration)
    samples = np.arange(lambda_p)
    if method == "exp":
        if alpha != 1:
            f = start_freq * (end_freq / start_freq) ** \
                ((samples / (lambda_p - 1)) ** alpha)
        else:
            f = start_freq * (end_freq / start_freq) ** \
                (samples / (lambda_p - 1))
    else:
        f = start_freq + (end_freq - start_freq) * samples / (lambda_p - 1)
    waveform_table_length = len(waveform_table)
    gamma = np.cumsum(f * waveform_table_length / sample_rate).astype(np.int64)
    s = waveform_table[gamma % waveform_table_length]
    return s


def note_with_glissando_vibrato(start_freq=220, end_freq=440, duration=2,
                                vibrato_freq=4, max_pitch_dev=2, alpha=1,
                                alpha_vibrato=1, waveform_table=WAVEFORM_SINE,
                                vibrato_waveform_table=WAVEFORM_SINE,
                                number_of_samples=0, sample_rate=44100):
    """
    A note with a pitch transition (a glissando) and a vibrato.

    Parameters
    ----------
    start_freq : scalar
        The starting frequency.
    end_freq : scalar
        The final frequency.
    duration : scalar
        The duration of the sound in seconds.
    vibrato_freq : scalar
        The frequency of the vibrato oscillations in Hertz.
    max_pitch_dev : scalar
        The maximum deviation of pitch of the vibrato in semitones.
    alpha : scalar
        An index to begin the transitions faster or slower.
        If alpha != 1, the transition is not of linear pitch.
    alpha_vibrato : scalar
        An index to distort the pitch deviation of the vibrato.
    waveform_table : array_like
        The table with the waveform to synthesize the sound.
    vibrato_waveform_table : array_like
        The table with the waveform for the vibrato oscillatory pattern.
    number_of_samples : integer
        The number of samples of the sound.
        If supplied, d is not used.
    sample_rate : integer
        The sample rate.

    Returns
    -------
    s : ndarray
        A numpy array where each value is a PCM sample of the sound.

    See Also
    --------
    P : A glissando.
    V : A musical note with an oscillation of pitch.
    N : A basic musical note without vibrato.
    T : A tremolo, an oscillation of loudness.
    F : Fade in and out.
    L : A transition of loudness.

    Examples
    --------
    >>> W(note_with_pitch_vibrato())  # writes file with glissando and vibrato
    >>> s = H([AD(sonic_vector=note_with_pitch_vibrato(i, j)) \
            for i, j in zip([220, 440, 4000], [440, 220, 220])])
    >>> W(s)  # writes a file with glissandi and vibratos

    """
    waveform_table = np.array(waveform_table)
    vibrato_waveform_table = np.array(vibrato_waveform_table)
    if number_of_samples:
        lambda_pv = number_of_samples
    else:
        lambda_pv = int(sample_rate * duration)
    samples = np.arange(lambda_pv)

    lv = len(vibrato_waveform_table)
    # LUT indexes
    gammav = (samples * vibrato_freq * lv / sample_rate).astype(np.int64)
    # values of the oscillatory pattern at each sample
    tv = vibrato_waveform_table[gammav % lv]

    if alpha != 1 or alpha_vibrato != 1:
        f = start_freq * (end_freq / start_freq) ** \
            ((samples / (lambda_pv - 1)) ** alpha) * 2. ** \
            ((tv * max_pitch_dev / 12) ** alpha_vibrato)
    else:
        f = start_freq * (end_freq / start_freq) ** \
            (samples / (lambda_pv - 1)) * 2. ** \
            ((tv * max_pitch_dev / 12) ** alpha)
    length = len(waveform_table)
    gamma = np.cumsum(f * length / sample_rate).astype(np.int64)
    s = waveform_table[gamma % length]
    return s


# FIXME: Unused param (`number_of_samples`)
def note_with_vibrato_seq_localization(freqs=[220, 440, 330],
                                       durations=[[2, 3], [2, 5, 3],
                                                  [2, 5, 6, 1, .4],
                                                  [4, 6, 1]],
                                       vibratos_freqs=[[2, 6, 1],
                                                       [.5, 15, 2, 6, 3]],
                                       max_pitch_devs=[[2, 1, 5],
                                                       [4, 3, 7, 10, 3]],
                                       alpha=[[1, 1], [1, 1, 1],
                                              [1, 1, 1, 1, 1], [1, 1, 1]],
                                       x=[-10, 10, 5, 3], y=[1, 1, .1, .1],
                                       method=['lin', 'exp', 'lin'],
                                       waveform_tables=[
                                           [WAVEFORM_TRIANGULAR,
                                            WAVEFORM_TRIANGULAR],
                                           [WAVEFORM_SINE,
                                            WAVEFORM_TRIANGULAR,
                                            WAVEFORM_SINE],
                                           [WAVEFORM_SINE, WAVEFORM_SINE,
                                            WAVEFORM_SINE, WAVEFORM_SINE,
                                            WAVEFORM_SINE]],
                                       stereo=True, zeta=0.215, air_temp=20,
                                       number_of_samples=0, sample_rate=44100):
    """
    A sound with arbitrary meta-vibratos, transitions of frequency and
    localization.

    Parameters
    ----------
    freqs : list of lists of scalars
        The frequencies of the note at each end of the transitions.
    durations : list of lists of scalars
        The durations of the pitch transitions and then of the
        vibratos and then of the position transitions.
    vibratos_freqs :  list of lists of scalars
        The frequencies of each vibrato.
    max_pitch_devs : list of lists of scalars
        The maximum deviation of pitch in the vibratos in semitones.
    alpha : list of lists of scalars
        Indexes to distort the pitch deviations of the transitions
        and the vibratos.
    x : list of lists of scalars
        The x positions at each end of the transitions.
    y : list of lists of scalars
        The y positions at each end of the transitions.
    method : list of strings
        An entry for each transition of location: 'exp' for
        exponential and 'lin' (default) for linear.
    stereo : boolean
        If True, returns a (2, nsamples) array representing
        a stereo sound. Else it returns a simple array
        for a mono sound.
    waveform_tables : list of lists of array_likes
        The tables with the waveforms to synthesize the sound
        and then for the oscillatory patterns of the vibratos.
        All the tables for f should have the same size.
    zeta : scalar
        The distance between the ears in meters.
    air_temp : scalar
        The air temperature in Celsius.
        (Used to calculate the acoustic velocity.)
    number_of_samples : scalar
        The number of samples of the sound.
        If supplied, d is not used.
    sample_rate : scalar
        The sample rate.

    Returns
    -------
    s : ndarray
        A numpy array where each value is a PCM sample of the sound.

    See Also
    --------
    PV : A note with a glissando and a vibrato.
    D : A note with a simple linear transition of location.
    PVV : A note with a glissando and two vibratos.
    VV : A note with a vibrato with two oscillatory patterns.
    N : a basic musical note without vibrato.
    V : a musical note with an oscillation of pitch.
    T : a tremolo, an oscillation of loudness.
    F : fade in and out.
    L : a transition of loudness.

    Examples
    --------
    >>> write_wav_mono(note_with_pitch_vibratos_localization())

    Notes
    -----
    Check the functions above for more information about how each feature of
    this function is implemented.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    # pitch transition contributions
    f_ = []
    for i, dur in enumerate(durations[0]):
        lambda_ = int(sample_rate * dur)
        samples = np.arange(lambda_)
        f1, f2 = freqs[i:i + 2]
        if alpha[0][i] != 1:
            f = f1 * (f2 / f1) ** ((samples / (lambda_ - 1)) ** alpha[0][i])
        else:
            f = f1 * (f2 / f1) ** (samples / (lambda_ - 1))
        f_.append(f)
    ft = np.hstack(f_)

    # vibrato contributions
    v_ = []
    for i, vib in enumerate(durations[1:-1]):
        v_ = []
        for j, dur in enumerate(vib):
            samples = np.arange(dur * sample_rate)
            lv = len(waveform_tables[i + 1][j])
            gammav = (samples * vibratos_freqs[i][j] * lv /
                      sample_rate).astype(np.int64)  # LUT indexes
            # values of the oscillatory pattern at each sample
            tv = waveform_tables[i + 1][j][gammav % lv]
            if alpha[i + 1][j] != 0:
                f = 2. ** ((tv * max_pitch_devs[i][j] / 12) ** alpha[i + 1][j])
            else:
                f = 2. ** (tv * max_pitch_devs[i][j] / 12)
            v_.append(f)

        v = np.hstack(v_)
        v_.append(v)

    v_ = [ft] + v_

    # Doppler/location localization contributions
    speed = 331.3 + .606 * air_temp
    # dl_ = []
    # dr_ = []
    # d_ = []
    f_ = []
    iid_a = []
    if stereo:
        for i in range(len(method)):
            m = method[i]
            a = alpha[-1][i]
            lambda_d = int(sample_rate * durations[-1][i])
            if m == 'exp':
                if a == 1:
                    foo = np.arange(lambda_d + 1) / lambda_d
                else:
                    foo = (np.arange(lambda_d + 1) / lambda_d) ** a
                xi = x[i] * (x[i + 1] / x[i]) ** foo
                yi = y[i] * (y[i + 1] / y[i]) ** foo
            else:
                xi = x[i] + (x[i + 1] - x[i]) * np.arange(lambda_d + 1) / \
                    lambda_d
                yi = y[i] + (y[i + 1] - y[i]) * np.arange(lambda_d + 1) / \
                    lambda_d
            dl = np.sqrt((xi + zeta / 2) ** 2 + yi ** 2)
            dr = np.sqrt((xi - zeta / 2) ** 2 + yi ** 2)
            if len(f_) == 0:
                itd0 = (dl[0] - dr[0]) / speed
                lambda_itd = itd0 * sample_rate
            iid_al = 1 / dl
            iid_ar = 1 / dr

            vsl = sample_rate * (dl[1:] - dl[:-1])
            vsr = sample_rate * (dr[1:] - dr[:-1])
            fl = speed / (speed + vsl)
            fr = speed / (speed + vsr)

            f_.append(np.vstack((fl, fr)))
            iid_a.append(np.vstack((iid_al[:-1], iid_ar[:-1])))
    else:
        for i in range(len(method)):
            m = method[i]
            a = alpha[-1][i]
            lambda_d = int(sample_rate * durations[-1][i])
            if m == 'exp':
                if a == 1:
                    foo = np.arange(lambda_d + 1) / lambda_d
                else:
                    foo = (np.arange(lambda_d + 1) / lambda_d) ** a
                xi = x[i] * (x[i + 1] / x[i]) ** foo
                yi = y[i] * (y[i + 1] / y[i]) ** foo
            else:
                xi = x[i] + (x[i + 1] - x[i]) * np.arange(lambda_d + 1) / \
                    lambda_d
                yi = y[i] + (y[i + 1] - y[i]) * np.arange(lambda_d + 1) / \
                    lambda_d
            durations = np.sqrt(xi ** 2 + yi ** 2)
            iid = 1 / durations

            # velocities at each point
            vs = sample_rate * (durations[1:] - durations[:-1])
            f_ = speed / (speed + vs)

            f_.append(f_)
            iid_a.append(iid[:-1])
    f_ = np.hstack(f_)
    iid_a = np.hstack(iid_a)

    # find maximum size, fill others with ones
    amax = max([len(i) if len(i.shape) == 1 else len(i[0]) for i in v_ + [f_]])
    for i, contrib in enumerate(v_[1:]):
        v_[i + 1] = np.hstack((contrib, np.ones(amax - len(contrib))))
    v_[0] = np.hstack((v_[0], np.ones(amax - len(v_[0])) * freqs[-1]))
    if stereo:
        f_ = np.hstack((f_, np.ones((2, amax - len(f_[0])))))
    else:
        f_ = np.hstack((f_, np.ones(amax - len(f_))))

    length = len(waveform_tables[0][0])
    if not stereo:
        v_.extend(f_)
        f = np.prod(v_, axis=0)
        gamma = np.cumsum(f * length / sample_rate).astype(np.int64)
        s_ = []
        pointer = 0
        for i, t in enumerate(waveform_tables[0]):
            lambda_d = int(sample_rate * durations[0][i])
            s = t[gamma[pointer:pointer + lambda_d] % length]
            pointer += lambda_d
            s_.append(s)
        s = t[gamma[pointer:] % length]
        s_.append(s)
        s = np.hstack(s_)
        s[:len(iid_a)] *= iid_a
        s[len(iid_a):] *= iid_a[-1]
    else:
        # left channel
        Vl = v_ + [f_[0]]
        f = np.prod(Vl, axis=0)
        gamma = np.cumsum(f * length / sample_rate).astype(np.int64)
        s_ = []
        pointer = 0
        for i, t in enumerate(waveform_tables[0]):
            lambda_d = int(sample_rate * durations[0][i])
            s = t[gamma[pointer:pointer + lambda_d] % length]
            pointer += lambda_d
            s_.append(s)
        s = t[gamma[pointer:] % length]
        s_.append(s)
        tl = np.hstack(s_)
        tl[:len(iid_a[0])] *= iid_a[0]
        tl[len(iid_a[0]):] *= iid_a[0][-1]

        # right channel
        vr = v_ + [f_[1]]
        f = np.prod(vr, axis=0)
        gamma = np.cumsum(f * length / sample_rate).astype(np.int64)
        s_ = []
        pointer = 0
        for i, t in enumerate(waveform_tables[0]):
            lambda_d = int(sample_rate * durations[0][i])
            s = t[gamma[pointer:pointer + lambda_d] % length]
            pointer += lambda_d
            s_.append(s)
        s = t[gamma[pointer:] % length]
        s_.append(s)
        tr = np.hstack(s_)
        tr[:len(iid_a[1])] *= iid_a[1]
        tr[len(iid_a[1]):] *= iid_a[1][-1]

        if x[0] > 0:
            tl = np.hstack((np.zeros(int(lambda_itd)), tl))
            tr = np.hstack((tr, np.zeros(int(lambda_itd))))
        else:
            tl = np.hstack((tl, np.zeros(-int(lambda_itd))))
            tr = np.hstack((np.zeros(-int(lambda_itd)), tr))
        s = np.vstack((tl, tr))
    return s


def note_with_two_vibratos_glissando(start_freq=220, end_freq=440, duration=2,
                                     vibrato_freq=2, secondary_vibrato_freq=6,
                                     max_pitch_dev=2,
                                     secondary_max_pitch_dev=.5,
                                     alpha=1, alphav1=1, alphav2=1,
                                     waveform_table=WAVEFORM_TRIANGULAR,
                                     tabv1=WAVEFORM_SINE, tabv2=WAVEFORM_SINE,
                                     number_of_samples=0, sample_rate=44100):
    """
    A note with a glissando and a vibrato that also has a secondary
    oscillatory pattern.

    Parameters
    ----------
    start_freq : scalar
        The starting frequency.
    end_freq : scalar
        The final frequency.
    duration : scalar
        The duration of the sound in seconds.
    vibrato_freq : scalar
        The frequency of the vibrato.
    secondary_vibrato_freq : scalar
        The frequency of the secondary pattern of the vibrato.
    max_pitch_dev : scalar
        The maximum deviation of pitch in the vibrato in semitones.
    max_pitch_dev : scalar
        The maximum deviation in semitones of pitch in the
        secondary pattern of the vibrato.
    alpha : scalar
        An index to begin the transitions faster or slower.
        If alpha != 1, the transition is not of linear pitch.
    alphav1 : scalar
        An index to distort the pitch deviation of the vibrato.
    alphav2 : scalar
        An index to distort the pitch deviation of the
        secondary pattern of the vibrato.
    waveform_table : array_like
        The table with the waveform to synthesize the sound.
    tabv1 : array_like
        The table with the waveform for the vibrato oscillatory pattern.
    tabv2 : array_like
        The table with the waveform for the
        secondary pattern of the vibrato.
    number_of_samples : scalar
        The number of samples of the sound.
        If supplied, d is not used.
    sample_rate : scalar
        The sample rate.

    Returns
    -------
    s : ndarray
        A numpy array where each value is a PCM sample of the sound.

    See Also
    --------
    PV : A note with a glissando and a vibrato.
    VV : A note with a vibrato with two oscillatory patterns.
    PV_ : A note with arbitrary pitch transitions and vibratos.
    V : a musical note with an oscillation of pitch.
    N : a basic musical note without vibrato.
    T : a tremolo, an oscillation of loudness.
    F : fade in or out.
    L : a transition of loudness.

    Examples
    --------
    >>> W(note_with_pitch_vibratos())
    >>> s = H([AD(note_with_pitch_vibratos(secondary_vibrato_freq=i,
                                           max_pitch_dev=j)) \
                    for i, j in zip([330, 440, 100], [8, 2, 15])])
    >>> W(s)

    """
    waveform_table = np.array(waveform_table)
    tabv1 = np.array(tabv1)
    tabv2 = np.array(tabv2)
    if number_of_samples:
        lambda_pvv = number_of_samples
    else:
        lambda_pvv = int(sample_rate * duration)
    samples = np.arange(lambda_pvv)

    lv1 = len(tabv1)
    # LUT indexes
    gammav1 = (samples * vibrato_freq * lv1 / sample_rate).astype(np.int64)
    # values of the oscillatory pattern at each sample
    tv1 = tabv1[gammav1 % lv1]

    lv2 = len(tabv2)
    # LUT indexes
    gammav2 = (samples * secondary_vibrato_freq * lv2 /
               sample_rate).astype(np.int64)
    # values of the oscillatory pattern at each sample
    tv2 = tabv1[gammav2 % lv2]

    if alpha != 1 or alphav1 != 1 or alphav2 != 1:
        f = start_freq * (end_freq / start_freq) ** \
            ((samples / (lambda_pvv - 1)) ** alpha) * 2. ** \
            ((tv1 * max_pitch_dev / 12) ** alphav1) * 2. ** \
            ((tv2 * secondary_max_pitch_dev / 12) ** alphav2)
    else:
        f = start_freq * (end_freq / start_freq) ** \
            (samples / (lambda_pvv - 1)) * 2. ** \
            ((tv1 * max_pitch_dev / 12)) * 2. ** \
            (tv2 * secondary_max_pitch_dev / 12)
    length = len(waveform_table)
    gamma = np.cumsum(f * length / sample_rate).astype(np.int64)
    s = waveform_table[gamma % length]
    return s


def note_with_vibratos_glissandos(freqs=[220, 440, 330],
                                  durations=[[2, 3], [2, 5, 3],
                                             [2, 5, 6, 1, .4]],
                                  vibratos_freqs=[[2, 6, 1],
                                                  [.5, 15, 2, 6, 3]],
                                  vibratos_max_pitch_devs=[[2, 1, 5],
                                                           [4, 3, 7, 10, 3]],
                                  alpha=[[1, 1], [1, 1, 1], [1, 1, 1, 1, 1]],
                                  waveform_tables=[[WAVEFORM_TRIANGULAR,
                                                    WAVEFORM_TRIANGULAR],
                                                   [WAVEFORM_SINE,
                                                    WAVEFORM_TRIANGULAR,
                                                    WAVEFORM_SINE],
                                                   [WAVEFORM_SINE,
                                                    WAVEFORM_SINE,
                                                    WAVEFORM_SINE,
                                                    WAVEFORM_SINE,
                                                    WAVEFORM_SINE]],
                                  number_of_samples=0, sample_rate=44100):
    """
    A note with an arbitrary sequence of pitch transition and a meta-vibrato.

    A meta-vibrato consists in multiple vibratos.
    The sequence of pitch transitions is a glissandi.

    Parameters
    ----------
    freqs : list of lists of scalars
        The frequencies of the note at each end of the transitions.
    durations : list of lists of scalars
        The durations of the transitions and then of the vibratos.
    vibratos_freqs :  list of lists of scalars
        The frequencies of each vibrato.
    vibratos_max_pitch_devs : list of lists of scalars
        The maximum deviation of pitch in the vibratos in semitones.
    alpha : list of lists of scalars
        Indexes to distort the pitch deviations of the transitions
        and the vibratos.
    waveform_tables : list of lists of array_likes
        The tables with the waveforms to synthesize the sound
        and for the oscillatory patterns of the vibratos.
        All the tables for f should have the same size.
    number_of_samples : scalar
        The number of samples of the sound.
        If supplied, d is not used.
    sample_rate : scalar
        The sample rate.

    Returns
    -------
    s : ndarray
        A numpy array where each value is a PCM sample of the sound.

    See Also
    --------
    PV : A note with a glissando and a vibrato.
    PVV : A note with a glissando and two vibratos.
    VV : A note with a vibrato with two oscillatory patterns.
    N : a basic musical note without vibrato.
    V : a musical note with an oscillation of pitch.
    T : a tremolo, an oscillation of loudness.
    F : fade in and out.
    L : a transition of loudness.

    Examples
    --------
    >>> W(note_with_pitches_vibratos())

    """
    # pitch transition contributions
    f_ = []
    for i, dur in enumerate(durations[0]):
        lambda_ = int(sample_rate * dur)
        samples = np.arange(lambda_)
        f1, f2 = freqs[i:i + 2]
        if alpha[0][i] != 1:
            f = f1 * (f2 / f1) ** ((samples / (lambda_ - 1)) ** alpha[0][i])
        else:
            f = f1 * (f2 / f1) ** (samples / (lambda_ - 1))
        f_.append(f)
    ft = np.hstack(f_)

    # vibrato contributions
    v_ = []
    for i, vib in enumerate(durations[1:]):
        v_ = []
        for j, dur in enumerate(vib):
            samples = np.arange(dur * sample_rate)
            lv = len(waveform_tables[i + 1][j])
            gammav = (samples * vibratos_freqs[i][j] * lv /
                      sample_rate).astype(np.int64)  # LUT indexes
            # values of the oscillatory pattern at each sample
            tv = waveform_tables[i + 1][j][gammav % lv]
            if alpha[i + 1][j] != 0:
                f = 2. ** ((tv * vibratos_max_pitch_devs[i][j] / 12) **
                           alpha[i + 1][j])
            else:
                f = 2. ** (tv * vibratos_max_pitch_devs[i][j] / 12)
            v_.append(f)

        v = np.hstack(v_)
        v_.append(v)

    # find maximum size, fill others with ones
    v_ = [ft] + v_
    amax = max([len(i) for i in v_])
    for i, contrib in enumerate(v_[1:]):
        v_[i + 1] = np.hstack((contrib, np.ones(amax - len(contrib))))
    v_[0] = np.hstack((v_[0], np.ones(amax - len(v_[0])) * freqs[-1]))

    f = np.prod(v_, axis=0)
    length = len(waveform_tables[0][0])
    gamma = np.cumsum(f * length / sample_rate).astype(np.int64)
    s_ = []
    pointer = 0
    for i, t in enumerate(waveform_tables[0]):
        lambda_2 = int(sample_rate * durations[0][i])
        s = t[gamma[pointer:pointer + lambda_2] % length]
        pointer += lambda_2
        s_.append(s)
    s = t[gamma[pointer:] % length]
    s_.append(s)
    s = np.hstack(s_)
    return s


def note_with_vibrato(freq=220, duration=2, vibrato_freq=4,
                      max_pitch_dev=2, waveform_table=WAVEFORM_TRIANGULAR,
                      vibrato_waveform_table=WAVEFORM_SINE,
                      alpha=1, number_of_samples=0, sample_rate=44100):
    """
    Synthesize a musical note with a vibrato.

    Set fv=0 or nu=0 (or use N()) for a note without vibrato.
    A vibrato is an oscillatory pattern of pitch [1].

    Parameters
    ----------
    freq : scalar
        The frequency of the note in Hertz.
    duration : scalar
        The duration of the note in seconds.
    vibrato_freq : scalar
        The frequency of the vibrato oscillations in Hertz.
    max_pitch_deviation : scalar
        The maximum deviation of pitch in the vibrato in semitones.
    waveform_table : array_like
        The table with the waveform to synthesize the sound.
    vibrato_waveform_table : array_like
        The table with the waveform for the vibrato oscillatory pattern.
    alpha : scalar
        An index to distort the vibrato [1].
        If alpha != 1, the vibrato is not of linear pitch.
    number_of_samples : integer
        The number of samples in the sound.
        If supplied, d is ignored.
    sample_rate : integer
        The sample rate.

    Returns
    -------
    result : ndarray
        A numpy array where each value is a PCM sample of the note.

    See Also
    --------
    synth_note : A basic musical note without vibrato.
    T : A tremolo, an oscillation of loudness.
    fm : A linear oscillation of the frequency (not linear pitch).
    AM : A linear oscillation of amplitude (not linear loudness).
    V_ : A shorthand to render a note with vibrato using
        a reference frequency and a pitch interval.

    Examples
    --------
    >>> write_wav_mono(note_with_vibrato())  # writes a WAV file of a note
    >>> s = H([note_with_vibrato(i, j) for i, j in zip([200, 500, 100],
                                                       [2, 1, 2])])
    >>> s2 = note_with_vibrato(440, 1.5, 6, 1)

    Notes
    -----
    In the MASS framework implementation,
    for a sound with a vibrato (or FM) to be synthesized using LUT,
    the vibrato pattern is considered when performing the lookup calculations.

    The tremolo and AM patterns are implemented as separate amplitude
    envelopes.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    waveform_table = np.array(waveform_table)
    vibrato_waveform_table = np.array(vibrato_waveform_table)
    if number_of_samples:
        lambda_v = number_of_samples
    else:
        lambda_v = int(sample_rate * duration)
    samples = np.arange(lambda_v)

    vibrato_waveform_table_length = len(vibrato_waveform_table)
    gamma_v = (samples * vibrato_freq * vibrato_waveform_table_length /
               sample_rate).astype(
        np.int64)  # LUT indexes
    # values of the oscillatory pattern at each sample
    t_v = vibrato_waveform_table[gamma_v % vibrato_waveform_table_length]

    # frequency in Hz at each sample
    if alpha == 1:
        f = freq * 2. ** (t_v * max_pitch_dev / 12)
    else:
        f = freq * 2. ** ((t_v * max_pitch_dev / 12) ** alpha)
    waveform_table_length = len(waveform_table)
    # shift in table between each sample
    d_gamma = f * (waveform_table_length / sample_rate)
    gamma = np.cumsum(d_gamma).astype(np.int64)  # total shift at each sample
    # final sample lookup
    result = waveform_table[gamma % waveform_table_length]
    return result


def note_with_two_vibratos(freq=220, duration=2, vibrato_freq=2,
                           secondary_vibrato_freq=6, nu1=2, nu2=4, alphav1=1,
                           alphav2=1, waveform_table=WAVEFORM_TRIANGULAR,
                           vibrato_waveform_table=WAVEFORM_SINE,
                           sec_vibrato_waveform_table=WAVEFORM_SINE,
                           number_of_samples=0, sample_rate=44100):
    """
    A note with a vibrato that also has a secondary oscillatory pattern.

    Parameters
    ----------
    freq : scalar
        The frequency of the note.
    duration : scalar
        The duration of the sound in seconds.
    vibrato_freq : scalar
        The frequency of the vibrato.
    secondary_vibrato_freq : scalar
        The frequency of the secondary pattern of the vibrato.
    nu1 : scalar
        The maximum deviation of pitch in the vibrato in semitones.
    nu2 : scalar
        The maximum deviation in semitones of pitch in the
        secondary pattern of the vibrato.
    alphav1 : scalar
        An index to distort the pitch deviation of the vibrato.
    alphav2 : scalar
        An index to distort the pitch deviation of the
        secondary pattern of the vibrato.
    waveform_table : array_like
        The table with the waveform to synthesize the sound.
    vibrato_waveform_table : array_like
        The table with the waveform for the vibrato oscillatory pattern.
    secondary_vibrato_waveform_table : array_like
        The table with the waveform for the
        secondary pattern of the vibrato.
    number_of_samples : scalar
        The number of samples of the sound.
        If supplied, d is not used.
    sample_rate : scalar
        The sample rate.

    Returns
    -------
    s : ndarray
        A numpy array where each value is a PCM sample of the sound.

    See Also
    --------
    PV : A note with a glissando and a vibrato.
    PVV : A note with a glissando and a vibrato with two oscillatory patterns.
    N : A basic musical note without vibrato.
    V : A musical note with an oscillation of pitch.
    T : A tremolo, an oscillation of loudness.
    F : Fade in and out.
    L : A transition of loudness.

    Examples
    --------
    >>> W(note_with_vibratos())  # writes file with a two simultaneous vibratos
    >>> s = H([AD(note_with_vibratos(vibrato_freq=i, secondary_vibrato_freq=j))
              for i, j in zip([2, 6, 4], [8, 10, 15])])
    >>> W(s)  # writes a file with two vibratos

    """
    waveform_table = np.array(waveform_table)
    vibrato_waveform_table = np.array(vibrato_waveform_table)
    sec_vibrato_waveform_table = np.array(sec_vibrato_waveform_table)
    if number_of_samples:
        lambda_vv = number_of_samples
    else:
        lambda_vv = int(sample_rate * duration)
    samples = np.arange(lambda_vv)

    lv1 = len(vibrato_waveform_table)
    # LUT indexes
    gammav1 = (samples * vibrato_freq * lv1 / sample_rate).astype(np.int64)
    # values of the oscillatory pattern at each sample
    tv1 = vibrato_waveform_table[gammav1 % lv1]

    lv2 = len(sec_vibrato_waveform_table)
    gammav2 = (samples * secondary_vibrato_freq * lv2 /
               sample_rate).astype(np.int64)  # LUT indexes
    # values of the oscillatory pattern at each sample
    tv2 = vibrato_waveform_table[gammav2 % lv2]

    if alphav1 != 1 or alphav2 != 1:
        f = freq * 2. ** ((tv1 * nu1 / 12) ** alphav1) * 2. ** \
            ((tv2 * nu2 / 12) ** alphav2)
    else:
        f = freq * 2. ** (tv1 * nu1 / 12) * 2. ** (tv2 * nu2 / 12)
    length = len(waveform_table)
    gamma = np.cumsum(f * length / sample_rate).astype(np.int64)
    s = waveform_table[gamma % length]
    return s


def trill(freqs=[440, 440 * 2 ** (2 / 12)], notes_per_second=17, duration=5,
          sample_rate=44100):
    """
    Makes a trill.

    This is just a simple function for exemplifying
    the synthesis of trills.
    The user is encouraged to make its own functions
    for trills and set e.g. ADSR envelopes, tremolos
    and vibratos as intended.

    Parameters
    ----------
    freqs : iterable of scalars
        Frequencies to the iterated.
    notes_per_second : scalar
        The number of notes per second.
    duration : scalar
        The maximum duration of the trill in seconds.
    sample_rate : integer
        The sample rate.

    See Also
    --------
    V : A note with vibrato.
    PV_ : a note with an arbitrary sequence of pitch transition and
          a meta-vibrato.
    T : A tremolo envelope.

    Returns
    -------
    s : ndarray
        The PCM samples of the resulting sound.

    Examples
    --------
    >>> W(trill())

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    number_of_samples = 44100 / notes_per_second
    pointer = 0
    i = 0
    s = []
    while pointer + number_of_samples < duration * 44100:
        ns = int(number_of_samples * (i + 1) - pointer)
        note_ = note(freqs[i % len(freqs)], number_of_samples=ns,
                     waveform_table=WAVEFORM_TRIANGULAR,
                     sample_rate=sample_rate)
        s.append(adsr(sonic_vector=note_, release_duration=10))
        pointer += ns
        i += 1
    return np.hstack(s)
