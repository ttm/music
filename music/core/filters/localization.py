"""Stereo localization filters and related helpers."""

import numpy as np
import warnings
from music.core.synths.notes import note, note_with_phase
from music.utils import WAVEFORM_SINE


def localize(sonic_vector=None, theta=0, distance=0, x=.1, y=.01,
             zeta=0.215, air_temp=20, sample_rate=44100):
    """
    Make a mono sound stereo and localize it by a very naive method.

    See bellow for implementation notes.

    Parameters
    ----------
    sonic_vector : array_like
        An one dimensional with the PCM samples of the sound.
    x : scalar
        The lateral component of the position in meters.
    y : scalar
        The frontal component of the position in meters.
    theta : scalar
        The azimuthal angle of the position in degrees. If theta is supplied,
        x and y are ignored and dist must also be supplied for the sound
        localization to have effect.
    distance : scalar
        The distance of the source from the listener in meters.
    zeta : scalar
        The distance between the ears in meters.
    air_temp : scalar
        The temperature in Celsius used for calculating the speed of sound.
    sample_rate : integer
        The sample rate.

    Returns
    -------
    s : ndarray
        A (2, nsamples) shaped array with the PCM samples of the stereo sound.

    See Also
    --------
    reverb : A reverberator.
    localize2 : a less naive implementation of localization by ITD and IID.

    Examples
    --------
    >>> write_wav_stereo(localize())
    >>> write_wav_stereo(horizontal_stack([
    ...     localize(note_with_vibrato(duration=1), x=i, y=j)
    ...     for i, j in zip([.1, .7, np.pi - .1, np.pi - .7],
    ...                     [.1, .1, .1, .1])]))


    Notes
    -----
    A Head Related Transfer Function would localize more convincingly than
    either this or localize2; none is implemented yet.

    Uses the most naive ITD and IID calculations as described in [1]. A less
    naive method is implemented in localize2(). Nonetheless, if dist is small
    enough (e.g. <.3), the perception of theta occurs and might be used.
    The advantages of this method are:

      - It is fast.
      - It is simple.
      - It is true to sound propagation phenomenon (although it does not
        consider the human body beyond the localization of the ears).
      - It can be used easily for tweaks (such as for a moving source
        resulting in a Doppler Effect).

    When az = tan^{-1}(y/x) lies in the 'cone of confusion', many values of x
    and y have the same ITD and IID [1]. Furthermore, lateral sources have the
    low frequencies diffracted and reach the opposite ear with a delay of
    ~0.7s [1]. The height of a source and if it is in front or behind a
    listener are cues given by the HRTF [1]. These issues are not taken into
    account in this function.

    The value of zeta is ~0.215 for adult humans [1].

    This implementation assumes that the speed of sound (in air) is
    s = 331.2 + 0.606 * temp.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if sonic_vector is None:
        sonic_vector = note()
    if theta:
        theta = 2 * np.pi * theta / 360
        x = np.cos(theta) * distance
        y = np.sin(theta) * distance
    speed = 331.3 + .606 * air_temp

    dr = np.sqrt((x - zeta / 2) ** 2 + y ** 2)  # distance from right ear
    dl = np.sqrt((x + zeta / 2) ** 2 + y ** 2)  # distance from left ear

    iid_a = dr / dl  # proportion of amplitudes from left to right ear
    itd = (dl - dr) / speed  # seconds
    lambda_itd = int(itd * sample_rate)

    if x > 0:
        tl = np.hstack((np.zeros(lambda_itd), iid_a * sonic_vector))
        tr = np.hstack((sonic_vector, np.zeros(lambda_itd)))
    else:
        tl = np.hstack((sonic_vector, np.zeros(-lambda_itd)))
        tr = np.hstack((np.zeros(-lambda_itd), sonic_vector * (1 / iid_a)))
    s = np.vstack((tl, tr))
    return s


def _delayed(signal, delay):
    """Read `signal` `delay` samples ago, for a delay that need not be whole.

    Cubic Hermite (Catmull-Rom) interpolation between the four samples
    around each read position. Linear interpolation would do, but it is a
    lowpass: at a half-sample delay it costs about 1.5 dB at 8 kHz, where
    this keeps all but 0.3 dB. Rounding to whole samples instead would
    quantize a smoothly moving source into audible steps.

    Parameters
    ----------
    signal : ndarray
        The samples to read from.
    delay : ndarray
        How far back to read at each output sample, in samples. Values
        beyond either end read the nearest sample.

    Returns
    -------
    ndarray
        The delayed signal, the same length as `signal`.
    """
    position = np.arange(len(signal)) - delay
    index = np.floor(position).astype(np.int64)
    fraction = position - index

    def tap(offset):
        return signal[np.clip(index + offset, 0, len(signal) - 1)]

    before, at, after, beyond = tap(-1), tap(0), tap(1), tap(2)
    return at + .5 * fraction * (
        after - before + fraction * (
            2 * before - 5 * at + 4 * after - beyond + fraction * (
                3 * (at - after) + beyond - before)))


def _localize_positions(sonic_vector, xpos, ypos, zeta, air_temp,
                        sample_rate):
    """Apply the binaural cues of an arbitrary per-sample trajectory.

    :func:`localize_linear` walks a straight line and
    :func:`music.spatial_motion` walks a periodic one, but from the
    positions onwards the two do the same thing, so they do it here
    rather than each in its own copy.

    Parameters
    ----------
    sonic_vector : ndarray
        A one-dimensional array with the PCM samples of the sound.
    xpos, ypos : ndarray
        The source's coordinates at each sample, in meters, the same
        length as `sonic_vector`.
    zeta : scalar
        The distance between the ears in meters.
    air_temp : scalar
        The temperature in Celsius used for calculating the speed of
        sound.
    sample_rate : integer
        The sample rate.

    Returns
    -------
    ndarray
        A (2, nsamples) array with the PCM samples of the stereo sound.

    Notes
    -----
    Both cues are measured against the nearer ear at each sample, so the
    nearer ear is heard undelayed and unattenuated and only the
    *difference* between the ears is applied. The propagation delay
    common to both is not, so the result stays aligned with its input
    and keeps its length.
    """
    speed = 331.3 + .606 * air_temp

    # The distance from each ear at each sample.
    dist_l = np.sqrt((xpos + zeta / 2) ** 2 + ypos ** 2)
    dist_r = np.sqrt((xpos - zeta / 2) ** 2 + ypos ** 2)
    nearest = np.minimum(dist_l, dist_r)

    # IID: the nearer ear is heard in full, the farther one attenuated by
    # how much farther it is.
    iid_l = nearest / dist_l
    iid_r = nearest / dist_r

    # ITD: likewise, only the extra distance to the farther ear becomes a
    # delay, so the nearer ear is undelayed and the sound stays in step
    # with its input.
    delay_l = (dist_l - nearest) * sample_rate / speed
    delay_r = (dist_r - nearest) * sample_rate / speed

    return np.vstack((_delayed(sonic_vector, delay_l) * iid_l,
                      _delayed(sonic_vector, delay_r) * iid_r))


def localize_linear(sonic_vector=None, theta1=90, theta2=0, dist=.1,
                    zeta=0.215, air_temp=20, sample_rate=44100):
    """
    Localize a sound along a straight path between two angles.

    The source moves at a constant rate from `theta1` to `theta2`, both at
    `dist` from the listener, over the duration of `sonic_vector`. Its
    position is computed for every sample, and from each position the
    interaural intensity and time differences for that sample.

    Parameters
    ----------
    sonic_vector : array_like
        A one-dimensional array with the PCM samples of the sound.
    theta1 : scalar
        The azimuthal angle of the starting position in degrees.
    theta2 : scalar
        The azimuthal angle of the ending position in degrees.
    dist : scalar
        The distance of the source from the listener in meters, held for
        both endpoints.
    zeta : scalar
        The distance between the ears in meters.
    air_temp : scalar
        The temperature in Celsius used for calculating the speed of sound.
    sample_rate : integer
        The sample rate.

    Returns
    -------
    s : ndarray
        A (2, nsamples) array with the PCM samples of the stereo sound, the
        same length as `sonic_vector`.

    See Also
    --------
    localize : the same cues for a fixed position.
    music.note_with_doppler : a moving source, synthesized rather than
        filtered, so it also shifts pitch.

    Examples
    --------
    >>> write_wav_stereo(localize_linear(note(duration=3)))
    >>> # a pass from the left to the right and back
    >>> horizontal_stack(localize_linear(note(), theta1=180, theta2=0),
    ...                  localize_linear(note(), theta1=0, theta2=180))

    Notes
    -----
    The path is a straight line between the two positions, not an arc, which
    is what makes it linear.

    **The angle is measured from the ear axis, not from straight ahead.**
    Zero degrees is the right ear's side, 180 the left, and 90 and -90 are
    both on the median plane -- ahead and behind. Two positions that differ
    only in whether they are in front or behind therefore render
    identically, because the cue that separates them is one an HRTF carries
    and this does not; see :func:`localize2` for the same gap stated there.
    A pass across the head is ``theta1=180, theta2=0``.

    Both cues are measured against the nearer ear at each sample, as
    :func:`localize` measures them against the nearer ear of its one fixed
    position. The nearer ear is therefore heard undelayed and unattenuated,
    and the farther ear is delayed by the extra distance the sound travels
    to reach it, and attenuated by the ratio of the two distances. Only the
    *difference* between the ears is applied: the propagation delay common
    to both is not, so the result stays aligned with its input and keeps its
    length.

    The delay varies by a fraction of a sample from one sample to the next,
    so it is applied by reading the input at interpolated positions -- see
    :func:`_delayed`. Rounding to whole samples instead would quantize a
    smoothly moving source into audible steps.

    This filters an existing sound and so does not model the Doppler shift
    that a physically moving source would produce; use
    :func:`music.note_with_doppler` for that.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if sonic_vector is None:
        sonic_vector = note()
    sonic_vector = np.asarray(sonic_vector, dtype=np.float64)
    lambda_l = len(sonic_vector)
    if lambda_l == 0:
        return np.zeros((2, 0))

    theta1 = 2 * np.pi * theta1 / 360
    theta2 = 2 * np.pi * theta2 / 360
    x1, y1 = np.cos(theta1) * dist, np.sin(theta1) * dist
    x2, y2 = np.cos(theta2) * dist, np.sin(theta2) * dist

    # The position at each sample, moving in a straight line. The divisor
    # is guarded so a single-sample input stays at its starting point.
    span = max(lambda_l - 1, 1)
    progress = np.arange(lambda_l) / span
    xpos = x1 + (x2 - x1) * progress
    ypos = y1 + (y2 - y1) * progress

    return _localize_positions(sonic_vector, xpos, ypos, zeta, air_temp,
                               sample_rate)


def localize2(sonic_vector=None, theta=-70, x=.1, y=.01, zeta=0.215,
              air_temp=20, method="ifft", sample_rate=44100):
    """
    Make a mono sound stereo and localize it by experimental methods.

    See bellow for implementation notes. These implementations are not
    standard and are only to illustrate the method of using ITD and IID that
    are frequency dependent.

    Parameters
    ----------
    sonic_vector : array_like
        An one dimensional with the PCM samples of the sound.
    x : scalar
        The lateral component of the position in meters.
    y : scalar
        The frontal component of the position in meters.
    theta : scalar
        The azimuthal angle of the position in degrees.  If theta is supplied,
        x and y are ignored and dist must also be supplied for the sound
        localization to have effect.
    zeta : scalar
        The distance between the ears in meters.
    air_temp : scalar
        The temperature in Celsius used for calculating
        the speed of sound.
    method : string
        Set to "ifft" for a working method that changes the fourier spectral
        coefficients. Set to "brute" for using an implementation that
        sinthesizes each sinusoid in the fourier spectrum separately
        (currently not giving good results for all sounds).
    sample_rate : integer
        The sample rate.

    Returns
    -------
    s : ndarray
        A (2, nsamples) shaped array with the PCM samples of the stereo sound.

    See Also
    --------
    reverb : A reverberator.
    localize : a more naive and fast implementation of localization by ITD and
               IID.

    Examples
    --------
    >>> write_wav_stereo(localized2())
    >>> write_wav_stereo(horizontal_stack([
    ...     localized2(note_with_vibrato(duration=1), x=i, y=j)
    ...     for i, j in zip([.1, .7, np.pi - .1, np.pi - .7],
    ...                     [.1, .1, .1, .1])]))

    Notes
    -----
    Uses a less naive ITD and IID calculations as described in [1].

    See localize() for further notes.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if method not in ("ifft", "brute"):
        raise ValueError("The only methods implemented are ifft and brute")
    if sonic_vector is None:
        sonic_vector = note()
    if not theta:
        theta_ = np.arctan2(-x, y)
    else:
        theta_ = 2 * np.pi * theta / 360
        theta_ = np.arcsin(np.sin(theta_))  # sign of theta is used
    speed = 331.3 + .606 * air_temp

    c = np.fft.fft(sonic_vector)
    norms = np.abs(c)
    angles = np.angle(c)

    lambda_l = len(sonic_vector)
    max_coef = int(lambda_l / 2)
    df = 2 * sample_rate / lambda_l

    # zero theta in right ahead and counter-clockwise is positive
    # theta_ = 2*np.pi*theta/360
    freqs = np.arange(max_coef) * df
    # max_size = len(sonic_vector) + 300*zeta*np.sin(theta_)*fs
    # s = np.zeros( (2, max_size) )
    if method == "ifft":
        normsl = np.copy(norms)
        anglesl = np.copy(angles)
        normsr = np.copy(norms)
        anglesr = np.copy(angles)
    else:
        # limit the number of coeffs considered
        energy = np.cumsum(norms[:max_coef] ** 2)
        p = 0.01
        cutoff = energy.max() * (1 - p)
        ncoeffs = (energy < cutoff).sum()
        maxfreq = ncoeffs * df
        if maxfreq <= 4000:
            foo = .3
        else:
            foo = .2
        # A sample count, so a whole number of them: it sizes the buffers
        # below, all of which np.zeros refuses to build from a float.
        maxsize = int(np.ceil(
            len(sonic_vector)
            + sample_rate * foo * np.sin(abs(theta_)) / speed
        ))
        # Annotated without a shape: it is rebuilt by np.vstack further
        # down, and numpy's stubs narrow np.zeros((2, n)) to a 2-tuple shape
        # that the vstack result does not match.
        s: np.ndarray = np.zeros((2, maxsize))

    if method == "ifft":
        # ITD implies a phase change
        # IID implies a change in the norm
        for i in range(max_coef):
            if i == 0:
                continue
            f = freqs[i]
            if f <= 4000:
                itd = .3 * zeta * np.sin(theta_) / speed
            else:
                itd = .2 * zeta * np.sin(theta_) / speed
            iid = 1 + ((f / 1000) ** .8) * np.sin(abs(theta_))
            # not needed, coefs are duplicated afterwards:
            # if i != Lambda/2:
            #     IID *= 2
            # IID > 0 : left ear has amplification
            # ITD > 0 : right ear has a delay
            # relate ITD to phase change (anglesl)
            lamb = 1 / f
            if theta_ > 0:
                change = itd - (itd // lamb) * lamb
                change_ = (change / lamb) * 2 * np.pi
                anglesr[i] += change_
                normsl[i] *= iid
            else:
                itd = -itd
                change = itd - (itd // lamb) * lamb
                change_ = (change / lamb) * 2 * np.pi
                anglesl[i] += change_
                normsr[i] *= iid

    elif method == "brute":
        warnings.warn("This can take a long time...")
        for i in range(ncoeffs):
            if i == 0:
                continue
            f = freqs[i]
            if f <= 4000:
                itd = .3 * zeta * np.sin(theta_) / speed
            else:
                itd = .2 * zeta * np.sin(theta_) / speed
            iid = 1 + ((f / 1000) ** .8) * np.sin(abs(theta_))
            # IID > 0 : left ear has amplification
            # ITD > 0 : right ear has a delay
            itd_l = abs(int(sample_rate * itd))
            # The Nyquist bin is not doubled, having no conjugate partner.
            # Unreachable as written: the loop runs to ncoeffs - 1, and
            # ncoeffs <= max_coef == int(lambda_l / 2), so i never reaches
            # lambda_l / 2. Kept in case that bound is ever widened.
            if i == lambda_l / 2:  # pragma: no cover
                amplitude = norms[i] / lambda_l
            else:
                amplitude = 2 * norms[i] / lambda_l
            sine = note_with_phase(freq=f, number_of_samples=lambda_l,
                                   waveform_table=WAVEFORM_SINE,
                                   sample_rate=sample_rate,
                                   phase=angles[i]) * amplitude

            # Account for phase and energy
            if theta_ > 0:
                tl = sine * iid
                tr = np.copy(sine)
            else:
                tl = np.copy(sine)
                tr = sine * iid

            if theta > 0:
                tl = np.hstack((tl, np.zeros(itd_l)))
                tr = np.hstack((np.zeros(itd_l), tr))
            else:
                tl = np.hstack((np.zeros(itd_l), tl))
                tr = np.hstack((tr, np.zeros(itd_l)))

            tl = np.hstack((tl, np.zeros(maxsize - len(tl))))
            tr = np.hstack((tr, np.zeros(maxsize - len(tr))))
            s_ = np.vstack((tl, tr))
            s += s_
    if method == "ifft":
        coefsl = normsl * np.e ** (anglesl * 1j)
        # The conjugate mirror runs to lambda_l - max_coef, not max_coef:
        # for an odd length those differ and the assignment did not fit.
        mirror = slice(1, lambda_l - max_coef)
        coefsl[max_coef + 1:] = np.real(
                coefsl[mirror])[::-1] - 1j * np.imag(
                        coefsl[mirror])[::-1]
        sl = np.fft.ifft(coefsl).real

        coefsr = normsr * np.e ** (anglesr * 1j)
        # The conjugate mirror runs to lambda_l - max_coef, not max_coef:
        # for an odd length those differ and the assignment did not fit.
        mirror = slice(1, lambda_l - max_coef)
        coefsr[max_coef + 1:] = np.real(
                coefsr[mirror])[::-1] - 1j * np.imag(
                        coefsr[mirror])[::-1]
        sr = np.fft.ifft(coefsr).real
        s = np.vstack((sl, sr))
    # If in need to force energy to be preserved, try:
    # energy1 = np.sum(sonic_vector**2)
    # energy2 = np.sum(s**2)
    # s = s*(energy1/energy2)**.5
    return s
