import numpy as n
from .n import N
from .n_ import N_
from ...utils import S

def loc_(sonic_vector=N(), theta=-70, x=.1, y=.01, zeta=0.215,
        temp=20, method="ifft", fs=44100):
    """
    Make a mono sound stereo and localize it by experimental methods.

    See bellow for implementation notes.
    These implementations are not standard and are only
    to illustrate the method of using ITD and IID
    that are frequency dependent.

    Parameters
    ----------
    sonic_vector : array_like
        An one dimensional with the PCM samples of the sound.
    x : scalar
        The lateral component of the position in meters.
    y : scalar
        The frontal component of the position in meters.
    theta : scalar
        The azimuthal angle of the position in degrees.
        If theta is supplied, x and y are ignored
        and dist must also be supplied
        for the sound localization to have effect.
    dist : scalar
        The distance of the source from the listener
        in meters.
    zeta : scalar
        The distance between the ears in meters.
    temp : scalar
        The temperature in Celsius used for calculating
        the speed of sound.
    method : string
        Set to "ifft" for a working method that changes the
        fourier spectral coefficients.
        Set to "brute" for using an implementation that
        sinthesizes each sinusoid in the fourier spectrum
        separately (currently not giving good results for
        all sounds).
    fs : integer
        The sample rate.

    Returns
    -------
    s : ndarray
        A (2, nsamples) shaped array with the PCM
        samples of the stereo sound.

    See Also
    --------
    R : A reverberator.
    loc : a more naive and fast implementation of localization
    by ITD and IID.
    hrtf : performs localization by means of a
    Head Related Transfer Function.

    Examples
    --------
    >>> WS(loc_())  # write a soundfile that is localized
    >>> WS(H([loc_(V(d=1), x=i, y=j) for i, j in
    ...   zip([.1,.7,n.pi-.1,n.pi-.7], [.1,.1,.1,.1])]))

    Notes
    -----
    Uses a less naive ITD and IID calculations as described in [1].

    See loc() for further notes.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if method not in ("ifft", "brute"):
        print("The only methods implemented are ifft and brute")
    if not theta:
        theta_ = n.arctan2(-x, y)
    else:
        theta_ = 2*n.pi*theta/360
        theta_ = n.arcsin(n.sin(theta_))  # sign of theta is used
    speed = 331.3 + .606*temp

    c = n.fft.fft(sonic_vector)
    norms = n.abs(c)
    angles = n.angle(c)

    Lambda = len(sonic_vector)
    max_coef = int(Lambda/2)
    df = 2*fs/Lambda

    # zero theta in right ahead and counter-clockwise is positive
    # theta_ = 2*n.pi*theta/360
    freqs = n.arange(max_coef)*df
    # max_size = len(sonic_vector) + 300*zeta*n.sin(theta_)*fs
    # s = n.zeros( (2, max_size) )
    if method == "ifft":
        normsl = n.copy(norms)
        anglesl = n.copy(angles)
        normsr = n.copy(norms)
        anglesr = n.copy(angles)
    else:
        # limit the number of coeffs considered
        s = []
        energy = n.cumsum(norms[:max_coef]**2)
        p = 0.01
        cutoff = energy.max()*(1 - p)
        ncoeffs = (energy < cutoff).sum()
        maxfreq = ncoeffs*df
        if maxfreq <= 4000:
            foo = .3
        else:
            foo = .2
        maxsize = len(sonic_vector) + fs*foo*n.sin(abs(theta_))/speed
        s = n.zeros( (2, maxsize) )

    if method == "ifft":
        # ITD implies a phase change
        # IID implies a change in the norm
        for i in range(max_coef):
            if i==0:
                continue
            f = freqs[i]
            if f <= 4000:
                ITD = .3*zeta*n.sin(theta_)/speed
            else:
                ITD = .2*zeta*n.sin(theta_)/speed
            IID = 1 + ( (f/1000)**.8 )*n.sin(abs(theta_))
            # not needed, coefs are duplicated afterwards:
            # if i != Lambda/2:
            #     IID *= 2
            # IID > 0 : left ear has amplification
            # ITD > 0 : right ear has a delay
            # relate ITD to phase change (anglesl)
            lamb = 1/f
            if theta_ > 0:
                change = ITD - (ITD//lamb)*lamb
                change_ = (change/lamb)*2*n.pi
                anglesr[i] += change_
                normsl[i] *= IID
            else:
                ITD = -ITD
                change = ITD - (ITD//lamb)*lamb
                change_ = (change/lamb)*2*n.pi
                anglesl[i] += change_
                normsr[i] *= IID

    elif method == "brute":
        print("This can take a long time...")
        for i in range(ncoeffs):
            if i==0:
                continue
            f = freqs[i]
            if f <= 4000:
                ITD = .3*zeta*n.sin(theta_)/speed
            else:
                ITD = .2*zeta*n.sin(theta_)/speed
            IID = 1 + ( (f/1000)**.8 )*n.sin(abs(theta_))
            # IID > 0 : left ear has amplification
            # ITD > 0 : right ear has a delay
            ITD_l = abs(int(fs*ITD))
            if i == Lambda/2:
                amplitude = norms[i]/Lambda
            else:
                amplitude = 2*norms[i]/Lambda
            sine = N_(f=f, nsamples=Lambda, tab=S(),
                    fs=fs, phase=angles[i])*amplitude

            # Account for phase and energy
            if theta_ > 0:
                TL = sine*IID
                TR = n.copy(sine)
            else:
                TL = n.copy(sine)
                TR = sine*IID

            if theta > 0:
                TL = n.hstack(( TL, n.zeros(ITD_l) ))
                TR = n.hstack(( n.zeros(ITD_l), TR ))
            else:
                TL = n.hstack(( n.zeros(ITD_l), TL ))
                TR = n.hstack(( TR, n.zeros(ITD_l) ))

            TL = n.hstack(( TL, n.zeros(maxsize - len(TL)) ))
            TR = n.hstack(( TR, n.zeros(maxsize - len(TR)) ))
            s_ = n.vstack(( TL, TR ))
            s += s_
    if method == "ifft":
        coefsl = normsl*n.e**(anglesl*1j)
        coefsl[max_coef+1:] = n.real(coefsl[1:max_coef])[::-1] - 1j * \
            n.imag(coefsl[1:max_coef])[::-1]
        sl = n.fft.ifft(coefsl).real

        coefsr = normsr*n.e**(anglesr*1j)
        coefsr[max_coef+1:] = n.real(coefsr[1:max_coef])[::-1] - 1j * \
            n.imag(coefsr[1:max_coef])[::-1]
        sr = n.fft.ifft(coefsr).real
        s = n.vstack(( sl, sr ))
    # If in need to force energy to be preserved, try:
    # energy1 = n.sum(sonic_vector**2)
    # energy2 = n.sum(s**2)
    # s = s*(energy1/energy2)**.5
    return s