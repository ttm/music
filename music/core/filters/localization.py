import numpy as np
from music.core.synths.notes import note, note_with_phase
from music.utils import WAVEFORM_SINE


def localize(sonic_vector=note(), theta=0, distance=0, x=.1, y=.01,
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
    # FIXME: hrtf?
    hrtf : performs localization by means of a Head Related Transfer Function.

    Examples
    --------
    >>> write_wav_stereo(localize())
    >>> write_wav_stereo(horizontal_stack([
    ...     localize(note_with_vibrato(duration=1), x=i, y=j)
    ...     for i, j in zip([.1, .7, np.pi - .1, np.pi - .7],
    ...                     [.1, .1, .1, .1])]))


    Notes
    -----
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


def localize_linear(sonic_vector=note(), theta1=90, theta2=0, dist=.1,
                    zeta=0.215, air_temp=20, sample_rate=44100):
    """
    A linear variation of the localize function.

    See localize.

    """
    theta1 = 2 * np.pi * theta1 / 360
    x1 = np.cos(theta1) * dist
    y1 = np.sin(theta1) * dist
    theta2 = 2 * np.pi * theta2 / 360
    x2 = np.cos(theta2) * dist
    y2 = np.sin(theta2) * dist
    speed = 331.3 + .606 * air_temp

    lambda_l = len(sonic_vector)
    L = lambda_l  # FIXME: assure L is lambda_l
    l_ = L - 1
    xpos = x1 + (x2 - x1) * np.arange(lambda_l) / l_
    ypos = y1 + (y2 - y1) * np.arange(lambda_l) / l_
    d = np.sqrt((xpos - zeta / 2) ** 2 + ypos ** 2)
    d2 = np.sqrt((xpos + zeta / 2) ** 2 + ypos ** 2)
    iid_a = d / d2
    itd = (d2 - d) / speed
    lambda_itd = int(itd * sample_rate)

    if x1 > 0:
        tl = np.zeros(lambda_itd)
        tr = np.array([])
    else:
        tl = np.array([])
        tr = np.zeros(-lambda_itd)
    d_ = d[1:] - d[:-1]
    d2_ = d2[1:] - d2[:-1]
    d__ = np.cumsum(d_).astype(np.int64)
    d2__ = np.cumsum(d2_).astype(np.int64)
    # FIXME: here we have missing the correct use of the variables calculated
    # and also the return statement
    return iid_a, tl, tr, d__, d2__


def localize2(sonic_vector=note(), theta=-70, x=.1, y=.01, zeta=0.215,
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
    # FIXME: hrtf?
    hrtf : performs localization by means of a Head Related Transfer Function.

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
        print("The only methods implemented are ifft and brute")
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
        s = []
        energy = np.cumsum(norms[:max_coef] ** 2)
        p = 0.01
        cutoff = energy.max() * (1 - p)
        ncoeffs = (energy < cutoff).sum()
        maxfreq = ncoeffs * df
        if maxfreq <= 4000:
            foo = .3
        else:
            foo = .2
        maxsize = len(sonic_vector) + sample_rate * foo * np.sin(
                abs(theta_)) / speed
        s = np.zeros((2, maxsize))

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
        print("This can take a long time...")
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
            if i == lambda_l / 2:
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
        coefsl[max_coef + 1:] = np.real(
                coefsl[1:max_coef])[::-1] - 1j * np.imag(
                        coefsl[1:max_coef])[::-1]
        sl = np.fft.ifft(coefsl).real

        coefsr = normsr * np.e ** (anglesr * 1j)
        coefsr[max_coef + 1:] = np.real(
                coefsr[1:max_coef])[::-1] - 1j * np.imag(
                        coefsr[1:max_coef])[::-1]
        sr = np.fft.ifft(coefsr).real
        s = np.vstack((sl, sr))
    # If in need to force energy to be preserved, try:
    # energy1 = np.sum(sonic_vector**2)
    # energy2 = np.sum(s**2)
    # s = s*(energy1/energy2)**.5
    return s
