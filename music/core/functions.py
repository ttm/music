# This file is a copy of mass/src/aux/functions.py
# imported
# in music/utils.py
# as
# from .functions import *
# but it should be integrated into music package more
# properly.
# 
import numpy as n
from scipy.io import wavfile as w
from scipy.signal import fftconvolve as convolve_
# from HRTF import *
from numbers import Number


def convolve(sig1, sig2):
    if len(sig1) > len(sig2):
        return convolve(sig1, sig2)
    else:
        return convolve(sig2, sig1)

__doc__ = """
This file holds minimal implementations
to avoid repetitions in the
musical pieces of the MASS framework:
    https://github.com/ttm/mass

Sounds are represented as arrays of
PCM samples.
Stereo files are represented
by arrays of shape (2, nsamples).

See the music Python Package:
    https://github.com/ttm/music
for a usage of these implementations
within a package and derived routines.

See the file HRTF.py, in this same directory,
for the functions that use impulse responses of
Head Related Transfer Functions (HRTFs).

This file is a copy of mass/src/aux/functions.py
imported
in music/utils.py
as
from .functions import *
but it should be integrated into music package more
properly.

"""

### In this file are functions (only) for:
# IO
# Vibrato and Tremolo
# ADSR

H = n.hstack
def H_(*args):
    stereo = 0
    args = [n.array(a) for a in args]
    for a in args:
        if len(a.shape) == 2:
            stereo = 1
    if stereo:
        for i,a in enumerate(args):
            if len(a.shape) == 1:
                args[i] = n.array(( a, a ))
    return n.hstack(args)
#####################
# IO
def __n(sonic_vector, remove_bias=True):
    """
    Normalize mono sonic_vector.
    
    The final array will have values only between -1 and 1.
    
    Parameters
    ----------
    sonic_vector : array_like
        A (nsamples,) shaped array.
    remove_bias : boolean
        Whether to remove or not the bias (or offset)
    
    Returns
    -------
    s : ndarray
        A numpy array with values between -1 and 1.
    remove_bias : boolean
        Whether to remove or not the bias (or offset)

    """
    t = n.array(sonic_vector)
    if n.all(t==0):
        return t
    else:
        if remove_bias:
            s = t - t.mean()
            fact = max(s.max(), -s.min())
            s = s/fact
        else:
            s = ( (t-t.min()) / (t.max() -t.min()) )*2. -1.
        return s


def __ns(sonic_vector, remove_bias=True, normalize_sep=False):
    """
    Normalize a stereo sonic_vector.
    
    The final array will have values only between -1 and 1.
    
    Parameters
    ----------
    sonic_vector : array_like
        A (2, nsamples) shaped array.
    remove_bias : boolean
        Whether to remove or not the bias (or offset)
    normalize_sep : boolean
        Set to True if each channel should be normalized
        separately. If False (default), the arrays will be
        rescaled in the same proportion
        (preserves loudness proportion).
    
    Returns
    -------
    s : ndarray
        A numpy array with values between -1 and 1.

    """
    t = n.array(sonic_vector)
    if n.all(t==0):
        return t
    else:
        if remove_bias:
            s = t
            s[0] = s[0] - s[0].mean()
            s[1] = s[1] - s[1].mean()
            if normalize_sep:
                fact = max(s[0].max(), -s[0].min())
                s[0] = s[0]/fact
                fact = max(s[1].max(), -s[1].min())
                s[1] = s[1]/fact
            else:
                fact = max(s.max(), -s.min())
                s = s/fact
        else:
            if normalize_sep:
                amb1 = t[0].max() - t[0].min()
                amb2 = t[1].max() - t[1].min()
                t[0] = (t[0] - t[0].min())/amb1
                t[1] = (t[1] - t[1].min())/amb2
                s = t*2 - 1
            else:
                amb1 = t.max() - t.min()
                amb = max(amb1, amb2)
                t = (t - t.min())/amb
                t = (t - t.min())/amb
                s = t*2 - 1
        return s


monos = n.random.uniform(size=100000) 
def W(sonic_vector=monos, filename="asound.wav", fs=44100,
        fades=0, bit_depth=16, remove_bias=True):
    """
    Write a mono WAV file for a numpy array.
    
    One can also use, for example:
        import sounddevice as S
        S.play(__n(array))
    
    Parameters
    ----------
    sonic_vector : array_like
        The PCM samples to be written as a WAV sound file.
        The samples are always normalized by __n(sonic_vector)
        to have samples between -1 and 1.
    filename : string
        The filename to use for the file to be written.
    fs : scalar
        The sample frequency.
    fades : interable
        An iterable with two values for the milliseconds you
        want for the fade in and out (to avoid clicks).
    bit_depth : integer
        The number of bits in each sample of the final file.
    remove_bias : boolean
        Whether to remove or not the bias (or offset)

    See Also
    --------
    __n : Normalizes an array to [-1,1]
    W_ : Writes an array with the same arguments
    and order of them as scipy.io.wavfile.
    WS ; Write a stereo file.
    
    """
    s = __n(sonic_vector, remove_bias)*(2**(bit_depth-1)-1)
    if fades:
        s = AD(A=fades[0], S=0, R=fades[1], sonic_vector=s)
    if bit_depth not in (8, 16, 32, 64):
        print("bit_depth values allowed are only 8, 16, 32 and 64")
        print("File {} not written".format(filename))
    nn = eval("n.int"+str(bit_depth))
    s = nn(s)
    w.write(filename, fs, s)


stereos = n.vstack((n.random.uniform(size=100000), n.random.uniform(size=100000)))

def WS(sonic_vector=stereos, filename="asound.wav", fs=44100,
        fades=0, bit_depth=16, remove_bias=True, normalize_sep=False):
    """
    Write a stereo WAV files for a numpy array.
    
    Parameters
    ----------
    sonic_vector : array_like
        The PCM samples to be written as a WAV sound file.
        The samples are always normalized by __n(sonic_vector)
        to have samples between -1 and 1 and remove the offset.
        Use array of shape (nchannels, nsamples).
    filename : string
        The filename to use for the file to be written.
    fs : scalar
        The sample frequency.
    fades : interable
        An iterable with two values for the milliseconds you
        want for the fade in and out (to avoid clicks).
    bit_depth : integer
        The number of bits in each sample of the final file.
    remove_bias : boolean
        Whether to remove or not the bias (or offset)
    normalize_sep : boolean
        Set to True if each channel should be normalized
        separately. If False (default), the arrays will be
        rescaled in the same proportion.

    See Also
    --------
    __ns : Normalizes a stereo array to [-1,1]
    W ; Write a mono file.
    
    """
    s = __ns(sonic_vector, remove_bias, normalize_sep)*(2**(bit_depth-1)-1)
    if fades:
        s = ADS(A=fades[0], S=0, R=fades[1], sonic_vector=s)
    if bit_depth not in (8, 16, 32, 64):
        print("bit_depth values allowed are only 8, 16, 32 and 64")
        print("File {} not written".format(filename))
    nn = eval("n.int"+str(bit_depth))
    s = nn(s)
    w.write(filename, fs, s.T)


def W_(fn, fs, sa): 
    """To mimic scipy.io.wavefile input"""
    W(sa, fn, fs=44100)

###################
# Synthesis
fs = 44100  # Hz, standard sample rate

# very large tables, we are not worried about real time
# use Lt = 1024 if in need of better performance
Lambda_tilde = Lt = 1024*16

# Sine
foo = n.linspace(0, 2*n.pi,Lt, endpoint=False)
S = n.sin(foo)  # one period of a sinusoid with Lt samples

# Square
Q = n.hstack(  ( n.ones(int(Lt/2))*-1, n.ones(int(Lt/2)) )  )

# Triangular
foo = n.linspace(-1, 1, Lt/2, endpoint=False)
Tr = n.hstack(  ( foo, foo[::-1] )   )

# Sawtooth
Sa = n.linspace(-1, 1, Lt)


def N(f=220, d=2, tab=Tr, nsamples=0, fs=44100):
    """
    Synthesize a basic musical note.

    Parameters
    ----------
    f : scalar
        The frequency of the note in Hertz.
    d : scalar
        The duration of the note in seconds.
    tab : array_like
        The table with the waveform to synthesize the sound.
    nsamples : integer
        The number of samples in the sound.
        If not 0, d is ignored.
    fs : integer
        The sample rate.

    Returns
    -------
    s : ndarray
        A numpy array where each value is a PCM sample of the note.

    See Also
    --------
    V : A note with vibrato.
    T : A tremolo envelope.

    Examples
    --------
    >>> W(N())  # writes a WAV file of a note
    >>> s = H( [N(i, j) for i, j in zip([200, 500, 100], [2, 1, 2])] )
    >>> s2 = N(440, 1.5, tab=Sa)

    Notes
    -----
    In the MASS framework implementation,
    for a sound with a vibrato (or FM) to be synthesized using LUT,
    the vibrato pattern is considered when performing the lookup calculations.

    The tremolo and AM patterns are implemented as separate amplitude envelopes.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    tab = n.array(tab)
    if not nsamples:
        nsamples = int(d*fs)
    samples = n.arange(nsamples)
    l = len(tab)

    Gamma = (samples*f*l/fs).astype(n.int)
    s = tab[ Gamma % l ]
    return s


def N_(f=220, d=2, phase=0, tab=Tr, nsamples=0, fs=44100):
    """
    Synthesize a basic musical note with a phase.

    Is useful in more complex synthesis routines.
    For synthesizing a musical note directly,
    you probably want to use N() and disconsider
    the phase.

    Parameters
    ----------
    f : scalar
        The frequency of the note in Hertz.
    d : scalar
        The duration of the note in seconds.
    phase : scalar
        The phase of the wave in radians.
    tab : array_like
        The table with the waveform to synthesize the sound.
    nsamples : integer
        The number of samples in the sound.
        If not 0, d is ignored.
    fs : integer
        The sample rate.

    Returns
    -------
    s : ndarray
        A numpy array where each value is a PCM sample of the note.

    See Also
    --------
    N : A basic note.
    V : A note with vibrato.
    T : A tremolo envelope.

    Examples
    --------
    >>> W(N_())  # writes a WAV file of a note
    >>> s = H( [N_(i, j) for i, j in zip([200, 500, 100], [2, 1, 2])] )
    >>> s2 = N_(440, 1.5, tab=Sa)

    Notes
    -----
    In the MASS framework implementation,
    for a sound with a vibrato (or FM) to be synthesized using LUT,
    the vibrato pattern is considered when performing the lookup calculations.

    The tremolo and AM patterns are implemented as separate amplitude envelopes.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    tab = n.array(tab)
    if not nsamples:
        nsamples = int(d*fs)
    samples = n.arange(nsamples)
    l = len(tab)
    i0 = phase*l/(2*n.pi)
    Gamma = (i0 + samples*f*l/fs).astype(n.int)
    s = tab[ Gamma % l ]
    return s

def loc(sonic_vector=N(), theta=0, dist=0, x=.1, y=.01, zeta=0.215, temp=20, fs=44100):
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
    loc_ : a less naive implementation of localization
    by ITD and IID.
    hrtf : performs localization by means of a
    Head Related Transfer Function.

    Examples
    --------
    >>> WS(loc())  # write a soundfile that is localized
    >>> WS(H([loc(V(d=1), x=i, y=j) for i, j in
    ...   zip([.1,.7,n.pi-.1,n.pi-.7], [.1,.1,.1,.1])]))

    Notes
    -----
    Uses the most naive ITD and IID calculations as described in [1].
    A less naive method is implemented in loc_().
    Nonetheless, if dist is small enough (e.g. <.3),
    the perception of theta occurs and might be used.
    The advantages of this method are:
      - It is fast.
      - It is simple.
      - It is true to sound propagation phenomenon
      (although it does not consider the human body
      beyond the localization of the ears).
      - It can be used easily for tweaks
      (such as for a moving source resulting
      in a Doppler Effect).

    When az = tan^{-1}(y/x) lies in the 'cone of confusion',
    many values of x and y have the same ITD and IID [1].
    Furthermore, lateral sources have the low frequencies
    diffracted and reach the opposite ear with a delay
    of ~0.7s [1].
    The height of a source and if it is in front or
    behind a listener are cues given by te HRTF [1].
    These issues are not taken into account in this
    function.

    The value of zeta is ~0.215 for adult humans [1].

    This implementation assumes that the speed
    of sound (in air) is s = 331.3+0.606*temp.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if theta:
        theta = 2*n.pi*theta/360
        x = n.cos(theta)*dist
        y = n.sin(theta)*dist
    speed = 331.3 + .606*temp

    dr = n.sqrt((x-zeta/2)**2+y**2)  # distance from right ear
    dl = n.sqrt((x+zeta/2)**2+y**2)  # distance from left ear

    IID_a = dr/dl  # proportion of amplitudes from left to right ear
    ITD = (dl-dr)/speed  # seconds
    Lambda_ITD = int(ITD*fs)

    if x > 0:
        TL = n.hstack((n.zeros(Lambda_ITD), IID_a*sonic_vector))
        TR = n.hstack((sonic_vector, n.zeros(Lambda_ITD)))
    else:
        TL = n.hstack((sonic_vector, n.zeros(-Lambda_ITD)))
        TR = n.hstack((n.zeros(-Lambda_ITD), sonic_vector*(1/IID_a)))
    s = n.vstack((TL, TR))
    return s

def loc2(sonic_vector=N(), theta1=90, theta2=0, dist1=.1,
        dist2=.1, zeta=0.215, temp=20, fs=44100):
    """
    A linear variation of localization

    """
    theta1 = 2*n.pi*theta1/360
    x1 = n.cos(theta1)*dist
    y1 = n.sin(theta1)*dist
    theta2 = 2*n.pi*theta2/360
    x2 = n.cos(theta2)*dist
    y2 = n.sin(theta2)*dist
    speed = 331.3 + .606*temp

    Lambda = len(sonic_vector)
    L_ = L-1
    xpos = x1 + (x2 - x1)*n.arange(Lambda)/L_
    ypos = y1 + (y2 - y1)*n.arange(Lambda)/L_
    d = n.sqrt( (xpos-zeta/2)**2 + ypos**2 )
    d2 = n.sqrt( (xpos+zeta/2)**2 + ypos**2 )
    IID_a = d/d2
    ITD = (d2-d)/speed
    Lambda_ITD = int(ITD*fs)

    if x1 > 0:
        TL = n.zeros(Lambda_ITD)
        TR = n.array([])
    else:
        TL = n.array([])
        TR = n.zeros(-Lambda_ITD)
    d_ = d[1:] - d[:-1]
    d2_ = d2[1:] - d2[:-1]
    d__ = n.cumsum(d_).astype(n.int)
    d2__ = n.cumsum(d2_).astype(n.int)


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
            sine = N_(f=f, nsamples=Lambda, tab=S,
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


def V(f=220, d=2, fv=4, nu=2, tab=Tr, tabv=S,
        alpha=1, nsamples=0, fs=44100):
    """
    Synthesize a musical note with a vibrato.
    
    Set fv=0 or nu=0 (or use N()) for a note without vibrato.
    A vibrato is an oscillatory pattern of pitch [1].
    
    Parameters
    ----------
    f : scalar
        The frequency of the note in Hertz.
    d : scalar
        The duration of the note in seconds.
    fv : scalar
        The frequency of the vibrato oscillations in Hertz.
    nu : scalar
        The maximum deviation of pitch in the vibrato in semitones.
    tab : array_like
        The table with the waveform to synthesize the sound.
    tabv : array_like
        The table with the waveform for the vibrato oscillatory pattern.
    alpha : scalar
        An index to distort the vibrato [1]. 
        If alpha != 1, the vibrato is not of linear pitch.
    nsamples : integer
        The number of samples in the sound.
        If supplied, d is ignored.
    fs : integer
        The sample rate.

    Returns
    -------
    s : ndarray
        A numpy array where each value is a PCM sample of the note.

    See Also
    --------
    N : A basic musical note without vibrato.
    T : A tremolo, an oscillation of loudness.
    FM : A linear oscillation of the frequency (not linear pitch).
    AM : A linear oscillation of amplitude (not linear loudness).
    V_ : A shorthand to render a note with vibrato using
        a reference frequency and a pitch interval.

    Examples
    --------
    >>> W(V())  # writes a WAV file of a note
    >>> s = H( [V(i, j) for i, j in zip([200, 500, 100], [2, 1, 2])] )
    >>> s2 = V(440, 1.5, 6, 1)

    Notes
    -----
    In the MASS framework implementation,
    for a sound with a vibrato (or FM) to be synthesized using LUT,
    the vibrato pattern is considered when performing the lookup calculations.

    The tremolo and AM patterns are implemented as separate amplitude envelopes.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    tab = n.array(tab)
    tabv = n.array(tabv)
    if nsamples:
        Lambda = nsamples
    else:
        Lambda = int(fs*d)
    samples = n.arange(Lambda)

    lv = len(tabv)
    Gammav = (samples*fv*lv/fs).astype(n.int)  # LUT indexes
    # values of the oscillatory pattern at each sample
    Tv = tabv[ Gammav % lv ] 

    # frequency in Hz at each sample
    if alpha == 1:
        F = f*2.**(  Tv*nu/12  ) 
    else:
        F = f*2.**(  (Tv*nu/12)**alpha  ) 
    l = len(tab)
    D_gamma = F*(l/fs)  # shift in table between each sample
    Gamma = n.cumsum(D_gamma).astype(n.int)  # total shift at each sample
    s = tab[ Gamma % l ]  # final sample lookup
    return s


def T(d=2, fa=2, dB=10, alpha=1, taba=S, nsamples=0, sonic_vector=0, fs=44100):
    """
    Synthesize a tremolo envelope or apply it to a sound.
    
    Set fa=0 or dB=0 for a constant envelope with value 1.
    A tremolo is an oscillatory pattern of loudness [1].
    
    Parameters
    ----------
    d : scalar
        The duration of the envelope in seconds.
    fa : scalar
        The frequency of the tremolo oscillations in Hertz.
    dB : scalar
        The maximum deviation of loudness in the tremolo in decibels.
    alpha : scalar
        An index to distort the tremolo pattern [1].
    taba : array_like
        The table with the waveform for the tremolo oscillatory pattern.
    nsamples : integer
        The number of samples of the envelope. If supplied, d is ignored.
    sonic_vector : array_like
        Samples for the tremolo to be applied to.
        If supplied, d and nsamples are ignored.
    fs : integer
        The sample rate.

    Returns
    -------
    T : ndarray
        A numpy array where each value is a PCM sample
        of the envelope.
        if sonic_vector is 0.
        If sonic_vector is input,
        T is the sonic vector with the tremolo applied to it.

    See Also
    --------
    V : A musical note with an oscillation of pitch.
    FM : A linear oscillation of fundamental frequency.
    AM : A linear oscillation of amplitude.

    Examples
    --------
    >>> W(V()*T())  # writes a WAV file of a note with tremolo
    >>> s = H( [V()*T(fa=i, dB=j) for i, j in zip([6, 15, 100], [2, 1, 20])] )  # OR
    >>> s = H( [T(fa=i, dB=j, sonic_vector=V()) for i, j in zip([6, 15, 100], [2, 1, 20])] )
    >>> envelope2 = T(440, 1.5, 60)  # a lengthy envelope

    Notes
    -----
    In the MASS framework implementation, for obtaining a sound with a tremolo (or AM),
    the tremolo pattern is considered separately from a synthesis of the sound.

    The vibrato and FM patterns are considering when synthesizing the sound.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """

    taba = n.array(taba)
    if type(sonic_vector) in (n.ndarray, list):
        Lambda = len(sonic_vector)
    elif nsamples:
        Lambda = nsamples
    else:
        Lambda = n.floor(fs*d)
    samples = n.arange(Lambda)

    l = len(taba)
    Gammaa = (samples*fa*l/fs).astype(n.int)  # indexes for LUT
    # amplitude variation at each sample
    Ta = taba[ Gammaa % l ] 
    if alpha != 1:
        T = 10.**((Ta*dB/20)**alpha)
    else:
        T = 10.**(Ta*dB/20)
    if type(sonic_vector) in (n.ndarray, list):
        return T*sonic_vector
    else:
        return T


def AD(d=2, A=20, D=20, S=-5, R=50, trans="exp", alpha=1,
        dB=-80, to_zero=1, nsamples=0, sonic_vector=0, fs=44100):
    """
    Synthesize an ADSR envelope.
    
    ADSR (Atack, Decay, Sustain, Release) is a very traditional
    loudness envelope in sound synthesis [1].
    
    Parameters
    ----------
    d : scalar
        The duration of the envelope in seconds.
    A : scalar
        The duration of the Attack in milliseconds.
    D : scalar
        The duration of the Decay in milliseconds.
    S : scalar
        The Sustain level after the Decay in decibels.
        Usually negative.
    R : scalar
        The duration of the Release in milliseconds.
    trans : string
        "exp" for exponential transitions of amplitude 
        (linear loudness).
        "linear" for linear transitions of amplitude.
    alpha : scalar or array_like
        An index to make the exponential fade slower or faster [1].
        Ignored it transitions="linear" or alpha=1.
        If it is an array_like, it should hold three values to be used
        in Attack, Decay and Release.
    dB : scalar or array_like
        The decibels deviation to reach before using a linear fade
        to reach zero amplitude.
        If it is an array_like, it should hold two values,
        one for Attack and another for Release.
        Ignored if trans="linear".
    to_zero : scalar or array_like
        The duration in milliseconds for linearly departing from zero
        in the Attack and reaching the value of zero at the end
        of the Release.
        If it is an array_like, it should hold two values,
        one for Attack and another for Release.
        Is ignored if trans="linear".
    nsamples : integer
        The number of samples of the envelope.
        If supplied, d is ignored.
    sonic_vector : array_like
        Samples for the ADSR envelope to be applied to.
        If supplied, d and nsamples are ignored.
    fs : integer
        The sample rate.

    Returns
    -------
    AD : ndarray
        A numpy array where each value is a value of
        the envelope for the PCM samples if sonic_vector is 0.
        If sonic_vector is input,
        AD is the sonic vector with the ADSR envelope applied to it.

    See Also
    --------
    T : An oscillation of loudness.
    L : A loudness transition.
    F : A fade in or fade out.

    Examples
    --------
    >>> W(V()*AD())  # writes a WAV file of a note with ADSR envelope
    >>> s = H( [V()*AD(A=i, R=j) for i, j in zip([6, 50, 300], [100, 10, 200])] )  # OR
    >>> s = H( [AD(A=i, R=j, sonic_vector=V()) for i, j in zip([6, 15, 100], [2, 2, 20])] )
    >>> envelope = AD(d=440, A=10e3, D=0, R=5e3)  # a lengthy envelope

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if type(sonic_vector) in (n.ndarray, list):
        Lambda = len(sonic_vector)
    elif nsamples:
        Lambda = nsamples
    else:
        Lambda = int(d*fs)
    Lambda_A = int(A*fs*0.001)
    Lambda_D = int(D*fs*0.001)
    Lambda_R = int(R*fs*0.001)

    perc = to_zero/A
    A = F(out=0, method=trans, alpha=alpha, dB=dB, perc=perc, nsamples=Lambda_A)

    D = L(dev=S, method=trans, alpha=alpha, nsamples=Lambda_D)

    a_S = 10**(S/20.)
    S = n.ones( Lambda - (Lambda_A+Lambda_R+Lambda_D) )*a_S

    perc = to_zero/R
    R = F(method=trans, alpha=alpha, dB=dB, perc=perc, nsamples=Lambda_R)*a_S

    AD = n.hstack((A,D,S,R))
    if type(sonic_vector) in (n.ndarray, list):
        return sonic_vector*AD
    else:
        return AD

def ADS(d=2, A=20, D=20, S=-5, R=50, trans="exp", alpha=1,
        dB=-80, to_zero=1, nsamples=0, sonic_vector=0, fs=44100):
    """
    A shorthand to make an ADSR envelope for a stereo sound.

    See ADSR() for more information.

    """
    if type(sonic_vector) in (n.ndarray, list):
        sonic_vector1 = sonic_vector[0]
        sonic_vector2 = sonic_vector[1]
    else:
        sonic_vector1 = 0
        sonic_vector2 = 0
    s1 = AD(d=d, A=A, D=D, S=S, R=R, trans=trans, alpha=alpha,
        dB=dB, to_zero=to_zero, nsamples=nsamples, sonic_vector=sonic_vector1, fs=fs)
    s2 = AD(d=d, A=A, D=D, S=S, R=R, trans=trans, alpha=alpha,
        dB=dB, to_zero=to_zero, nsamples=nsamples, sonic_vector=sonic_vector2, fs=fs)
    s = n.vstack(( s1, s2 ))
    return s


def L(d=2, dev=10, alpha=1, to=True, method="exp",
        nsamples=0, sonic_vector=0, fs=44100):
    """
    An envelope for linear or exponential transition of amplitude.

    An exponential transition of loudness yields a linean
    transition of loudness (theoretically).

    Parameters
    ----------
    d : scalar
        The duration of the envelope in seconds.
    dev : scalar
        The deviation of the transition.
        If method="exp" the deviation is in decibels.
        If method="linear" the deviation is an amplitude proportion.
    alpha : scalar
        An index to make the transition slower or faster [1].
        Ignored it method="linear".
    to : boolean
        If True, the transition ends at the deviation.
        If False, the transition starts at the deviation.
    method : string
        "exp" for exponential transitions of amplitude (linear loudness).
        "linear" for linear transitions of amplitude.
    nsamples : integer
        The number of samples of the envelope.
        If supplied, d is ignored.
    sonic_vector : array_like
        Samples for the envelope to be applied to.
        If supplied, d and nsamples are ignored.
    fs : integer
        The sample rate.
        Only used if nsamples and sonic_vector are not supplied.

    Returns
    -------
    E : ndarray
        A numpy array where each value is a value of the envelope 
        for the PCM samples.
        If sonic_vector is supplied,
        ai is the sonic vector with the envelope applied to it.

    See Also
    --------
    L_ : An envelope with an arbitrary number of transitions.
    F : Fade in and out.
    AD : An ADSR envelope.
    T : An oscillation of loudness.

    Examples
    --------
    >>> W(V()*L())  # writes a WAV file of a loudness transition
    >>> s = H( [V()*L(dev=i, method=j) for i, j in zip([6, -50, 2.3], ["exp", "exp", "linear"])] )  # OR
    >>> s = H( [L(dev=i, method=j, sonic_vector=V()) for i, j in zip([6, -50, 2.3], ["exp", "exp", "linear"])] )
    >>> envelope = L(d=10, dev=-80, to=False, alpha=2)  # a lengthy fade in 

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if type(sonic_vector) in (n.ndarray, list):
        N = len(sonic_vector)
    elif nsamples:
        N = nsamples
    else:
        N = int(fs*d)
    samples = n.arange(N)
    N_ = N-1
    if 'lin' in method:
        if to:
            a0 = 1
            al = dev
        else:
            a0 = dev
            al = 1
        E = a0 + (al - a0)*samples/N_
    if 'exp' in method:
        if to:
            if alpha != 1:
                samples_ = (samples/N_)**alpha
            else:
                samples_ = (samples/N_)
        else:
            if alpha != 1:
                samples_ = ( (N_-samples)/N_)**alpha
            else:
                samples_ = ( (N_-samples)/N_)
        E = 10**(samples_*dev/20)
    if type(sonic_vector) in (n.ndarray, list):
        return E*sonic_vector
    else:
        return E
        

def F(d=2, out=True, method="exp", dB=-80, alpha=1, perc=1,
        nsamples=0, sonic_vector=0, fs=44100):
    """
    A fade in or out.

    Implements the loudness transition and asserts that it reaches
    zero amplitude.

    Parameters
    ----------
    d : scalar
        The duration in seconds of the fade.
    out : boolean
        If True, the fade is a fade out, else it is a fade in.
    method : string
        "exp" for an exponential transition of amplitude (linear loudness).
        "linear" for a linear transition of amplitude.
    dB : scalar
        The decibels from which to reach before using
        the linear transition to reach zero.
        Not used if method="linear".
    alpha : scalar
        An index to make the exponential fade slower or faster [1].
        Ignored it transitions="linear". 
    perc : scalar
        The percentage of the fade that is linear to assure it reaches zero.
        Has no effect if method="linear".
    nsamples : integer
        The number of samples of the fade. If supplied, d is ignored.
    sonic_vector : array_like
        Samples for the fade to be applied to.
        If supplied, d and nsamples are ignored.
    fs : integer
        The sample rate. Only used if nsamples and sonic_vector are not supplied.

    Returns
    -------
    T : ndarray
        A numpy array where each value is a value of the envelope for the PCM samples.
        If sonic_vector is input, T is the sonic vector with the fade applied to it.

    See Also
    --------
    AD : An ADSR envelope.
    L : A transition of loudness.
    L_ : An envelope with an arbitrary number or loudness transitions.
    T : An oscillation of loudness.

    Examples
    --------
    >>> W(V()*F())  # writes a WAV file with a fade in
    >>> s = H( [V()*F(out=i, method=j) for i, j in zip([1, 0, 1], ["exp", "exp", "linear"])] )  # OR
    >>> s = H( [F(out=i, method=j, sonic_vector=V()) for i, j in zip([1, 0, 1], ["exp", "exp", "linear"])] )
    >>> envelope = F(d=10, out=0, perc=0.1)  # a lengthy fade in 

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if type(sonic_vector) in (n.ndarray, list):
        if len(sonic_vector.shape) == 2:
            return resolveStereo(F, locals())
        N = len(sonic_vector)
    elif nsamples:
        N = nsamples
    else:
        N = int(fs*d)
    if 'lin' in method:
        if out:
            ai = L(method="linear", dev=0, nsamples=N)
        else:
            ai = L(method="linear", to=0, dev=0, nsamples=N)
    if 'exp' in method:
        N0 = int(N*perc/100)
        N1 = N - N0
        if out:
            ai1 = L(dev=dB, alpha=alpha, nsamples=N1)
            if N0:
                ai0 = L(method="linear", dev=0, nsamples=N0)*ai1[-1]
            else:
                ai0 = []
            ai = n.hstack((ai1, ai0))
        else:
            ai1 = L(dev=dB, to=0, alpha=alpha, nsamples=N1)
            if N0:
                ai0 = L(method="linear", to=0, dev=0, nsamples=N0)*ai1[0]
            else:
                ai0 = []
            ai = n.hstack((ai0, ai1))
    if type(sonic_vector) in (n.ndarray, list):
        return ai*sonic_vector
    else:
        return ai


def P(f1=220, f2=440, d=2, alpha=1, tab=S, method="exp",
        nsamples=0, fs=44100):
    """
    A note with a pitch transition: a glissando.

    Parameters
    ----------
    f1 : scalar
        The starting frequency.
    f2 : scalar
        The final frequency.
    d : scalar
        The duration of the sound in seconds.
    alpha : scalar
        An index to begin the transition faster or slower. 
        If alpha != 1, the transition is not of linear pitch.
    tab : array_like
        The table with the waveform to synthesize the sound.
    nsamples : integer
        The number of samples of the sound.
        If supplied, d is not used.
    method : string
        "exp" for an exponential transition of frequency
        (linear pitch).
        "lin" for a linear transition of amplitude.
    fs : integer
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
    >>> W(P())  # writes file with a glissando
    >>> s = H( [P(i, j) for i, j in zip([220, 440, 4000], [440, 220, 220])] )
    >>> W(s)  # writes a file with glissandi

    """
    tab = n.array(tab)
    if nsamples:
        Lambda = nsamples
    else:
        Lambda = int(fs*d)
    samples = n.arange(Lambda)
    if method=="exp":
        if alpha != 1:
            F = f1*(f2/f1)**( (samples / (Lambda-1))**alpha )
        else:
            F = f1*(f2/f1)**( samples / (Lambda-1) )
    else:
        F = f1 + (f2 - f1)*samples/(Lambda-1)
    l = len(tab)
    Gamma = n.cumsum( F*l/fs ).astype(n.int)
    s = tab[ Gamma % l ]
    return s


def PV(f1=220, f2=440, d=2, fv=4, nu=2, alpha=1,
        alphav=1, tab=S, tabv=S, nsamples=0, fs=44100):
    """
    A note with a pitch transition (a glissando) and a vibrato.

    Parameters
    ----------
    f1 : scalar
        The starting frequency.
    f2 : scalar
        The final frequency.
    d : scalar
        The duration of the sound in seconds.
    fv : scalar
        The frequency of the vibrato oscillations in Hertz.
    nu : scalar
        The maximum deviation of pitch of the vibrato in semitones.
    alpha : scalar
        An index to begin the transitions faster or slower. 
        If alpha != 1, the transition is not of linear pitch.
    alphav : scalar
        An index to distort the pitch deviation of the vibrato. 
    tab : array_like
        The table with the waveform to synthesize the sound.
    tabv : array_like
        The table with the waveform for the vibrato oscillatory pattern.
    nsamples : integer
        The number of samples of the sound.
        If supplied, d is not used.
    fs : integer
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
    >>> W(PV())  # writes file with a glissando and vibrato
    >>> s = H( [AD(sonic_vector=PV(i, j)) for i, j in zip([220, 440, 4000], [440, 220, 220])] )
    >>> W(s)  # writes a file with glissandi and vibratos

    """
    tab = n.array(tab)
    tabv = n.array(tabv)
    if nsamples:
        Lambda = nsamples
    else:
        Lambda = int(fs*d)
    samples = n.arange(Lambda)

    lv = len(tabv)
    Gammav = (samples*fv*lv/fs).astype(n.int)  # LUT indexes
    # values of the oscillatory pattern at each sample
    Tv = tabv[ Gammav % lv ] 

    if alpha != 1 or alphav != 1:
        F = f1*(f2/f1)**( (samples / (Lambda-1))**alpha )*2.**( (Tv*nu/12)**alphav )
    else:
        F = f1*(f2/f1)**( samples / (Lambda-1) )*2.**( (Tv*nu/12)**alpha )
    l = len(tab)
    Gamma = n.cumsum( F*l/fs ).astype(n.int)
    s = tab[ Gamma % l ]
    return s


def VV(f=220, d=2, fv1=2, fv2=6, nu1=2, nu2=4, alphav1=1,
        alphav2=1, tab=Tr, tabv1=S, tabv2=S, nsamples=0, fs=44100):
    """
    A note with a vibrato that also has a secondary oscillatory pattern.

    Parameters
    ----------
    f : scalar
        The frequency of the note.
    d : scalar
        The duration of the sound in seconds.
    fv1 : scalar
        The frequency of the vibrato.
    fv2 : scalar
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
    tab : array_like
        The table with the waveform to synthesize the sound.
    tabv1 : array_like
        The table with the waveform for the vibrato oscillatory pattern.
    tabv2 : array_like
        The table with the waveform for the
        secondary pattern of the vibrato.
    nsamples : scalar
        The number of samples of the sound.
        If supplied, d is not used.
    fs : scalar
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
    >>> W(VV())  # writes file with a two simultaneous vibratos
    >>> s = H( [AD(sonic_vector=VV(fv1=i, fv2=j)) for i, j in zip([2, 6, 4], [8, 10, 15])] )
    >>> W(s)  # writes a file with two vibratos

    """
    tab = n.array(tab)
    tabv1 = n.array(tabv1)
    tabv2 = n.array(tabv2)
    if nsamples:
        Lambda = nsamples
    else:
        Lambda = int(fs*d)
    samples = n.arange(Lambda)

    lv1 = len(tabv1)
    Gammav1 = (samples*fv1*lv1/fs).astype(n.int)  # LUT indexes
    # values of the oscillatory pattern at each sample
    Tv1 = tabv1[ Gammav1 % lv1 ] 

    lv2 = len(tabv2)
    Gammav2 = (samples*fv2*lv2/fs).astype(n.int)  # LUT indexes
    # values of the oscillatory pattern at each sample
    Tv2 = tabv1[ Gammav2 % lv2 ] 

    if alphav1 != 1 or alphav2 != 1:
        F = f*2.**( (Tv1*nu1/12)**alphav1 )*2.**( (Tv2*nu2/12)**alphav2 )
    else:
        F = f*2.**( (Tv1*nu1/12))*2.**( (Tv2*nu2/12))
    l = len(tab)
    Gamma = n.cumsum( F*l/fs ).astype(n.int)
    s = tab[ Gamma % l ]
    return s


def PVV(f1=220, f2=440, d=2, fv1=2, fv2=6, nu1=2, nu2=.5, alpha=1,
        alphav1=1, alphav2=1, tab=Tr, tabv1=S, tabv2=S, nsamples=0, fs=44100):
    """
    A note with a glissando and a vibrato that also has a secondary oscillatory pattern.

    Parameters
    ----------
    f1 : scalar
        The starting frequency.
    f2 : scalar
        The final frequency.
    d : scalar
        The duration of the sound in seconds.
    fv1 : scalar
        The frequency of the vibrato.
    fv2 : scalar
        The frequency of the secondary pattern of the vibrato.
    nu1 : scalar
        The maximum deviation of pitch in the vibrato in semitones.
    nu1 : scalar
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
    tab : array_like
        The table with the waveform to synthesize the sound.
    tabv1 : array_like
        The table with the waveform for the vibrato oscillatory pattern.
    tabv2 : array_like
        The table with the waveform for the
        secondary pattern of the vibrato.
    nsamples : scalar
        The number of samples of the sound.
        If supplied, d is not used.
    fs : scalar
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
    >>> W(PVV())  # writes file with a two simultaneous vibratos and a glissando
    >>> s = H( [AD(sonic_vector=PVV(fv2=i, nu1=j)) for i, j in zip([330, 440, 100], [8, 2, 15])] )
    >>> W(s)  # writes a file with two vibratos and a glissando

    """
    tab = n.array(tab)
    tabv1 = n.array(tabv1)
    tabv2 = n.array(tabv2)
    if nsamples:
        Lambda = nsamples
    else:
        Lambda = int(fs*d)
    samples = n.arange(Lambda)

    lv1 = len(tabv1)
    Gammav1 = (samples*fv1*lv1/fs).astype(n.int)  # LUT indexes
    # values of the oscillatory pattern at each sample
    Tv1 = tabv1[ Gammav1 % lv1 ] 

    lv2 = len(tabv2)
    Gammav2 = (samples*fv2*lv2/fs).astype(n.int)  # LUT indexes
    # values of the oscillatory pattern at each sample
    Tv2 = tabv1[ Gammav2 % lv2 ] 

    if alpha !=1 or alphav1 != 1 or alphav2 != 1:
        F = f1*(f2/f1)**( (samples / (Lambda-1))**alpha )*2.**( (Tv1*nu1/12)**alphav1 )*2.**( (Tv2*nu2/12)**alphav2 )
    else:
        F = f1*(f2/f1)**( samples / (Lambda-1) )*2.**( (Tv1*nu1/12))*2.**( (Tv2*nu2/12))
    l = len(tab)
    Gamma = n.cumsum( F*l/fs ).astype(n.int)
    s = tab[ Gamma % l ]
    return s

def PV_(f=[220, 440, 330], d=[[2,3],[2,5,3], [2,5,6,1,.4]],
        fv=[[2,6,1], [.5,15,2,6,3]], nu=[[2,1, 5], [4,3,7,10,3]],
        alpha=[[1, 1] , [1, 1, 1], [1, 1, 1, 1, 1]],
        tab=[[Tr,Tr], [S,Tr,S], [S,S,S,S,S]], nsamples=0, fs=44100):
    """
    A note with an arbitrary sequence of pitch transition and a meta-vibrato.

    A meta-vibrato consists in multiple vibratos.
    The sequence of pitch transitions is a glissandi.

    Parameters
    ----------
    f : list of lists of scalars
        The frequencies of the note at each end of the transitions.
    d : list of lists of scalars
        The durations of the transitions and then of the vibratos.
    fv :  list of lists of scalars
        The frequencies of each vibrato.
    nu : list of lists of scalars
        The maximum deviation of pitch in the vibratos in semitones.
    alpha : list of lists of scalars
        Indexes to distort the pitch deviations of the transitions
        and the vibratos.
    tab : list of lists of array_likes
        The tables with the waveforms to synthesize the sound
        and for the oscillatory patterns of the vibratos.
        All the tables for f should have the same size.
    nsamples : scalar
        The number of samples of the sound.
        If supplied, d is not used.
    fs : scalar
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
    >>> W(PV_())  # writes file with glissandi and vibratos

    """
    # pitch transition contributions
    F_ = []
    for i, dur in enumerate(d[0]):
        Lambda_ = int(fs*dur)
        samples = n.arange(Lambda_)
        f1, f2 = f[i:i+2]
        if alpha[0][i] != 1:
            F = f1*(f2/f1)**( (samples / (Lambda_-1))**alpha[0][i] )
        else:
            F = f1*(f2/f1)**( samples / (Lambda_-1) )
        F_.append(F)
    Ft = n.hstack(F_)

    # vibrato contributions
    V_=[]
    for i, vib in enumerate(d[1:]):
        v_=[]
        for j, dur in enumerate(vib):
            samples = n.arange(dur*fs)
            lv = len(tab[i+1][j])
            Gammav = (samples*fv[i][j]*lv/fs).astype(n.int)  # LUT indexes
            # values of the oscillatory pattern at each sample
            Tv = tab[i+1][j][ Gammav % lv ] 
            if alpha[i+1][j] != 0:
                F = 2.**( (Tv*nu[i][j]/12)**alpha[i+1][j] )
            else:
                F = 2.**( Tv*nu[i][j]/12 )
            v_.append(F)

        V=n.hstack(v_)
        V_.append(V)

    # find maximum size, fill others with ones
    V_ = [Ft] + V_
    amax = max([len(i) for i in V_])
    for i, contrib in enumerate(V_[1:]):
        V_[i+1] = n.hstack(( contrib, n.ones(amax - len(contrib)) ))
    V_[0] = n.hstack(( V_[0], n.ones(amax - len(V_[0]))*f[-1] ))

    F = n.prod(V_, axis=0)
    l = len(tab[0][0])
    Gamma = n.cumsum( F*l/fs ).astype(n.int)
    s_ = []
    pointer = 0
    for i, t in enumerate(tab[0]):
        Lambda = int(fs*d[0][i])
        s = t[ Gamma[pointer:pointer+Lambda] % l ]
        pointer += Lambda
        s_.append(s)
    s =  t[ Gamma[pointer:] % l ]
    s_.append(s)
    s = n.hstack(s_)
    return s


def L_(d=[2,4,2], dev=[5,-10,20], alpha=[1,.5, 20], method=["exp", "exp", "exp"],
        nsamples=0, sonic_vector=0, fs=44100):
    """
    An envelope with linear or exponential transitions of amplitude.

    See L() for more details.

    Parameters
    ----------
    d : iterable
        The durations of the transitions in seconds.
    dev : iterable
        The deviation of the transitions.
        If method="exp" the deviation is in decibels.
        If method="linear" the deviation is an amplitude proportion.
    alpha : iterable
        Indexes to make the transitions slower or faster [1].
        Ignored it method[1]="linear".
    method : iterable
        Methods for each transition.
        "exp" for exponential transitions of amplitude (linear loudness).
        "linear" for linear transitions of amplitude.
    nsamples : interable
        The number of samples of each transition.
        If supplied, d is ignored.
    sonic_vector : array_like
        Samples for the envelope to be applied to.
        If supplied, d or nsamples is used, the final
        sound has the greatest duration of sonic_array
        and d (or nsamples) and missing samples are
        replaced with silence (if sonic_vector is shorter)
        or with a constant value (if d or nsamples yield shorter
        sequences).
    fs : integer
        The sample rate.
        Only used if nsamples and sonic_vector are not supplied.

    Returns
    -------
    E : ndarray
        A numpy array where each value is a value of the envelope
        for the PCM samples.
        If sonic_vector is supplied,
        E is the sonic vector with the envelope applied to it.

    See Also
    --------
    L : An envelope for a loudness transition.
    F : Fade in and out.
    AD : An ADSR envelope.
    T : An oscillation of loudness.

    Examples
    --------
    >>> W(V(d=8)*L_())  # writes a WAV file with a loudness transitions

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if type(sonic_vector) in (n.ndarray, list):
        N = len(sonic_vector)
    elif nsamples:
        N = sum(nsamples)
    else:
        N = int(fs*sum(d))
    samples = n.arange(N)
    s = []
    fact = 1
    if nsamples:
        for i, ns in enumerate(nsamples):
            s_ = L(dev[i], alpha[i], nsamples=ns, 
                    method=method[i])*fact
            s.append(s_)
            fact = s_[-1]
    else:
        for i, dur in enumerate(d):
            s_ = L(dur, dev[i], alpha[i],
                    method=method[i], fs=fs)*fact
            s.append(s_)
            fact = s_[-1]
    E = n.hstack(s)
    if type(sonic_vector) in (n.ndarray, list):
        if len(E) < len(sonic_vector):
            s = n.hstack((E, n.ones(len(sonic_vector)-len(E))*E[-1]))
        if len(E) > len(sonic_vector):
            sonic_vector = n.hstack((sonic_vector, n.ones(len(E)-len(sonic_vector))*E[-1]))
        return sonic_vector*E
    else:
        return E


def T_(d=[[3,4,5],[2,3,7,4]], fa=[[2,6,20],[5,6.2,21,5]],
        dB=[[10,20,1],[5,7,9,2]], alpha=[[1,1,1],[1,1,1,9]],
            taba=[[S,S,S],[Tr,Tr,Tr,S]],
        nsamples=0, sonic_vector=0, fs=44100):
    """
    An envelope with multiple tremolos.

    Parameters
    ----------
    d : iterable of iterable of scalars
        the durations of each tremolo.
    fa : iterable of iterable of scalars
        The frequencies of each tremolo.
    dB : iterable of iterable of scalars
        The maximum loudness variation
        of each tremolo.
    alpha : iterable of iterable of scalars
        Indexes for distortion of each tremolo [1].
    taba : iterable of iterable of array_likes
        Tables for lookup for each tremolo.
    nsamples : iterable of iterable of scalars
        The number of samples or each tremolo.
    sonic_vector : array_like
        The sound to which apply the tremolos.
        If supplied, the tremolo lines are
        applied to the sound and missing samples
        are completed by zeros (if sonic_vector
        is smaller then the lengthiest tremolo)
        or ones (is sonic_vector is larger).
    fs : integer
        The sample rate

    Returns
    -------
    E : ndarray
        A numpy array where each value is a value of the envelope
        for the PCM samples.
        If sonic_vector is supplied,
        E is the sonic vector with the envelope applied to it.

    See Also
    --------
    L : An envelope for a loudness transition.
    L_ : An envelope with an arbitrary number of transitions.
    F : Fade in and out.
    AD : An ADSR envelope.
    T : An oscillation of loudness.

    Examples
    --------
    >>> W(V(d=8)*L_())  # writes a WAV file with a loudness transitions

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)
        

    """
    for i in range(len(taba)):
        for j in range(i):
            taba[i][j] = n.array(taba[i][j])
    T_ = []
    if nsamples:
        for i, ns in enumerate(nsamples):
            T_.append([])
            for j, ns_ in enumerate(ns):
                s = T(fa=fa[i][j], dB=dB[i][j], alpha=alpha[i][j],
                    taba=taba[i][j], nsamples=ns_)
                T_[-1].append(s)
    else:
        for i, durs in enumerate(d):
            T_.append([])
            for j, dur in enumerate(durs):
                s = T(dur, fa[i][j], dB[i][j], alpha[i][j],
                    taba=taba[i][j])
                T_[-1].append(s)
    amax = 0
    if type(sonic_vector) in (n.ndarray, list):
        amax = len(sonic_vector)
    for i in range(len(T_)):
        T_[i] = n.hstack(T_[i])
        amax = max(amax, len(T_[i]))
    for i in range(len(T_)):
        if len(T_[i]) < amax:
            T_[i] = n.hstack((T_[i], n.ones(amax-len(T_[i]))*T_[i][-1]))
    if type(sonic_vector) in (n.ndarray, list):
        if len(sonic_vector) < amax:
            sonic_vector = n.hstack(( sonic_vector, n.zeros(amax-len(sonic_vector)) ))
        T_.append(sonic_vector)
    s = n.prod(T_, axis=0)
    return s


def T_(d=[[3,4,5],[2,3,7,4]], fa=[[2,6,20],[5,6.2,21,5]],
        dB=[[10,20,1],[5,7,9,2]], alpha=[[1,1,1],[1,1,1,9]],
            taba=[[S,S,S],[Tr,Tr,Tr,S]],
        nsamples=0, sonic_vector=0, fs=44100):
    """
    An envelope with multiple tremolos.

    Parameters
    ----------
    d : iterable of iterable of scalars
        the durations of each tremolo.
    fa : iterable of iterable of scalars
        The frequencies of each tremolo.
    dB : iterable of iterable of scalars
        The maximum loudness variation
        of each tremolo.
    alpha : iterable of iterable of scalars
        Indexes for distortion of each tremolo [1].
    taba : iterable of iterable of array_likes
        Tables for lookup for each tremolo.
    nsamples : iterable of iterable of scalars
        The number of samples or each tremolo.
    sonic_vector : array_like
        The sound to which apply the tremolos.
        If supplied, the tremolo lines are
        applied to the sound and missing samples
        are completed by zeros (if sonic_vector
        is smaller then the lengthiest tremolo)
        or ones (is sonic_vector is larger).
    fs : integer
        The sample rate

    Returns
    -------
    E : ndarray
        A numpy array where each value is a value of the envelope
        for the PCM samples.
        If sonic_vector is supplied,
        E is the sonic vector with the envelope applied to it.

    See Also
    --------
    L : An envelope for a loudness transition.
    L_ : An envelope with an arbitrary number of transitions.
    F : Fade in and out.
    AD : An ADSR envelope.
    T : An oscillation of loudness.

    Examples
    --------
    >>> W(V(d=8)*L_())  # writes a WAV file with a loudness transitions

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)
        

    """
    for i in range(len(taba)):
        for j in range(i):
            taba[i][j] = n.array(taba[i][j])
    T_ = []
    if nsamples:
        for i, ns in enumerate(nsamples):
            T_.append([])
            for j, ns_ in enumerate(ns):
                s = T(fa=fa[i][j], dB=dB[i][j], alpha=alpha[i][j],
                    taba=taba[i][j], nsamples=ns_)
                T_[-1].append(s)
    else:
        for i, durs in enumerate(d):
            T_.append([])
            for j, dur in enumerate(durs):
                s = T(dur, fa[i][j], dB[i][j], alpha[i][j],
                    taba=taba[i][j])
                T_[-1].append(s)
    amax = 0
    if type(sonic_vector) in (n.ndarray, list):
        amax = len(sonic_vector)
    for i in range(len(T_)):
        T_[i] = n.hstack(T_[i])
        amax = max(amax, len(T_[i]))
    for i in range(len(T_)):
        if len(T_[i]) < amax:
            T_[i] = n.hstack((T_[i], n.ones(amax-len(T_[i]))*T_[i][-1]))
    if type(sonic_vector) in (n.ndarray, list):
        if len(sonic_vector) < amax:
            sonic_vector = n.hstack(( sonic_vector, n.zeros(amax-len(sonic_vector)) ))
        T_.append(sonic_vector)
    s = n.prod(T_, axis=0)
    return s



def trill(f=[440,440*2**(2/12)], ft=17, d=5, fs=44100):
    """
    Make a trill.

    This is just a simple function for exemplifying
    the synthesis of trills.
    The user is encouraged to make its own functions
    for trills and set e.g. ADSR envelopes, tremolos
    and vibratos as intended.

    Parameters
    ----------
    f : iterable of scalars
        Frequencies to the iterated.
    ft : scalar
        The number of notes per second.
    d : scalar
        The maximum duration of the trill in seconds.
    fs : integer
        The sample rate.

    See Also
    --------
    V : A note with vibrato.
    PV_ : a note with an arbitrary sequence of pitch transition and a meta-vibrato.
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
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    nsamples = 44100/ft
    pointer = 0
    i = 0
    s = []
    while pointer+nsamples < d*44100:
        ns = int(nsamples*(i+1) - pointer)
        note = N(f[i%len(f)], nsamples=ns,
                tab=Tr, fs=fs)
        s.append(AD(sonic_vector=note, R=10))
        pointer += ns
        i += 1
    trill = n.hstack(s)
    return trill

def D(f=220, d=2, tab=Tr, x=[-10, 10], y=[1,1], stereo=True,
        zeta=0.215, temp = 20, nsamples=0, fs=44100):
    """
    A simple note with a transition of localization and resulting Doppler effect.

    Parameters
    ----------
    f : scalar
        The frequency of the note in Hertz.
    d : scalar
        The duration of the note in seconds.
    tab : array_like
        The table with the waveform to synthesize the sound.
    x : iterable of scalars
        The starting and ending x positions.
    y : iterable of scalars
        The starting and ending y positions.
    stereo : boolean
        If True, returns a (2, nsamples) array representing
        a stereo sound. Else it returns a simple array
        for a mono sound.
    temp : scalar
        The air temperature in Celsius.
        (Used to calculate the acoustic velocity.)
    nsamples : integer
        The number of samples in the sound.
        If not 0, d is ignored.
    fs : integer
        The sample rate.

    Returns
    -------
    s : ndarray
        The PCM samples of the resulting sound.

    See Also
    --------
    D_ : a note with arbitrary vibratos, transitions of pitch
    and transitions of localization.
    PV_ : a note with an arbitrary sequence of pitch transition and a meta-vibrato.

    Examples
    --------
    >>> WS(D())
    >>> W(T()*D(stereo=False))

    Notes
    -----

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    tab = n.array(tab)
    if not nsamples:
        nsamples = int(d*fs)
    samples = n.arange(nsamples)
    l = len(tab)
    speed = 331.3 + .606*temp

    x = x[0] + (x[1] - x[0])*n.arange(nsamples+1)/(nsamples)
    y = y[0] + (y[1] - y[0])*n.arange(nsamples+1)/(nsamples)
    if stereo:
        dl = n.sqrt( (x+zeta/2)**2 + y**2 )
        dr = n.sqrt( (x-zeta/2)**2 + y**2 )
        IID_al = 1/dl
        IID_ar = 1/dr

        vsl = fs*(dl[1:]-dl[:-1])
        vsr = fs*(dr[1:]-dr[:-1])
        fl = f*speed/(speed+vsl)
        fr = f*speed/(speed+vsr)

        Gamma = n.cumsum(fl*l/fs).astype(n.int)
        sl = tab[ Gamma % l ]*IID_al[:-1]

        Gamma = n.cumsum(fr*l/fs).astype(n.int)
        sr = tab[ Gamma % l ]*IID_ar[:-1]

        ITD0 = (dl[0]-dr[0])/speed
        Lambda_ITD = ITD0*fs

        if x[0] > 0:
            TL = n.hstack(( n.zeros(int(Lambda_ITD)), sl ))
            TR = n.hstack(( sr, n.zeros(int(Lambda_ITD)) ))
        else:
            TL = n.hstack(( sl, n.zeros(-int(Lambda_ITD)) ))
            TR = n.hstack(( n.zeros(-int(Lambda_ITD)), sr ))
        s = n.vstack(( TL, TR ))
    else:
        d = n.sqrt( x**2 + y**2 )
        IID = 1/d

        vs = fs*(d[1:]-d[:-1])  # velocities at each point
        f_ = f*speed/(speed+vs)

        Gamma = n.cumsum(f_*l/fs).astype(n.int)
        s = tab[ Gamma % l ]*IID[:-1]
    return s


def D_(f=[220, 440, 330], d=[[2,3],[2,5,3], [2,5,6,1,.4],[4,6,1]],
        fv=[[2,6,1], [.5,15,2,6,3]], nu=[[2,1, 5], [4,3,7,10,3]],
        alpha=[[1, 1] , [1,1,1], [1,1,1,1,1], [1,1,1]],
        x=[-10,10,5,3], y=[1,1,.1,.1], method=['lin','exp','lin'],
        tab=[[Tr,Tr], [S,Tr,S], [S,S,S,S,S]], stereo=True,
        zeta=0.215, temp = 20, nsamples=0, fs=44100):
    """
    A sound with arbitrary meta-vibratos, transitions of frequency and localization.

    Parameters
    ----------
    f : list of lists of scalars
        The frequencies of the note at each end of the transitions.
    d : list of lists of scalars
        The durations of the pitch transitions and then of the 
        vibratos and then of the position transitions.
    fv :  list of lists of scalars
        The frequencies of each vibrato.
    nu : list of lists of scalars
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
    tab : list of lists of array_likes
        The tables with the waveforms to synthesize the sound
        and then for the oscillatory patterns of the vibratos.
        All the tables for f should have the same size.
    zeta : scalar
        The distance between the ears in meters.
    temp : scalar
        The air temperature in Celsius.
        (Used to calculate the acoustic velocity.)
    nsamples : scalar
        The number of samples of the sound.
        If supplied, d is not used.
    fs : scalar
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
    >>> W(D_())  # writes file with glissandi and vibratos

    Notes
    -----
    Check the functions above for more information about
    how each feature of this function is implemented.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    # pitch transition contributions
    F_ = []
    for i, dur in enumerate(d[0]):
        Lambda_ = int(fs*dur)
        samples = n.arange(Lambda_)
        f1, f2 = f[i:i+2]
        if alpha[0][i] != 1:
            F = f1*(f2/f1)**( (samples / (Lambda_-1))**alpha[0][i] )
        else:
            F = f1*(f2/f1)**( samples / (Lambda_-1) )
        F_.append(F)
    Ft = n.hstack(F_)

    # vibrato contributions
    V_=[]
    for i, vib in enumerate(d[1:-1]):
        v_=[]
        for j, dur in enumerate(vib):
            samples = n.arange(dur*fs)
            lv = len(tab[i+1][j])
            Gammav = (samples*fv[i][j]*lv/fs).astype(n.int)  # LUT indexes
            # values of the oscillatory pattern at each sample
            Tv = tab[i+1][j][ Gammav % lv ] 
            if alpha[i+1][j] != 0:
                F = 2.**( (Tv*nu[i][j]/12)**alpha[i+1][j] )
            else:
                F = 2.**( Tv*nu[i][j]/12 )
            v_.append(F)

        V=n.hstack(v_)
        V_.append(V)

    V_ = [Ft] + V_

    # Doppler/location localization contributions
    speed = 331.3 + .606*temp
    dl_ = []
    dr_ = []
    d_ = []
    F_ = []
    IID_a = []
    if stereo:
        for i in range(len(method)):
            m = method[i]
            a = alpha[-1][i]
            Lambda = int(fs*d[-1][i])
            if m == 'exp':
                if a == 1:
                    foo = n.arange(Lambda+1)/Lambda
                else:
                    foo = ( n.arange(Lambda+1)/Lambda )**a
                xi = x[i]*(x[i+1] / x[i])**( foo )
                yi = y[i]*(y[i+1] / y[i])**( foo )
            else:
                xi = x[i] + (x[i+1] - x[i])*n.arange(Lambda+1)/Lambda
                yi = y[i] + (y[i+1] - y[i])*n.arange(Lambda+1)/Lambda
            dl = n.sqrt( (xi+zeta/2)**2 + yi**2 )
            dr = n.sqrt( (xi-zeta/2)**2 + yi**2 )
            if len(F_) == 0:
                ITD0 = (dl[0]-dr[0])/speed
                Lambda_ITD = ITD0*fs
            IID_al = 1/dl
            IID_ar = 1/dr

            vsl = fs*(dl[1:]-dl[:-1])
            vsr = fs*(dr[1:]-dr[:-1])
            fl = speed/(speed+vsl)
            fr = speed/(speed+vsr)

            F_.append( n.vstack(( fl, fr )) )
            IID_a.append( n.vstack(( IID_al[:-1], IID_ar[:-1] )) )
    else:
        for i in range(len(methods)):
            m = methods[i]
            a = alpha[-1][i]
            Lambda = int(fs*d[-1][i])
            if m == 'exp':
                if a == 1:
                    foo = n.arange(Lambda+1)/Lambda
                else:
                    foo = ( n.arange(Lambda+1)/Lambda )**a
                xi = x[i]*(x[i+1] / x[i])**( foo )
                yi = y[i]*(y[i+1] / y[i])**( foo )
            else:
                xi = x[i] + (x[i+1] - x[i])*n.arange(Lambda+1)/(Lambda)
                yi = y[i] + (y[i+1] - y[i])*n.arange(Lambda+1)/(Lambda)
            d = n.sqrt( xi**2 + yi**2 )
            IID = 1/d

            vs = fs*(d[1:]-d[:-1])  # velocities at each point
            f_ = speed/(speed+vs)

            F_.append(f_)
            IID_a.append(IID[:-1])
    F_ = n.hstack( F_ )
    IID_a = n.hstack( IID_a )

    # find maximum size, fill others with ones
    amax = max([len(i) if len(i.shape)==1 else len(i[0]) for i in V_+[F_]])
    for i, contrib in enumerate(V_[1:]):
            V_[i+1] = n.hstack(( contrib, n.ones(amax - len(contrib)) ))
    V_[0] = n.hstack(( V_[0], n.ones(amax - len(V_[0]))*f[-1] ))
    if stereo:
        F_ = n.hstack(( F_, n.ones( (2, amax - len(F_[0]) )) ))
    else:
        F_ = n.hstack(( F_, n.ones( amax - len(F_) ) ))

    l = len(tab[0][0])
    if not stereo:
        V_.extend(F_)
        F = n.prod(V_, axis=0)
        Gamma = n.cumsum( F*l/fs ).astype(n.int)
        s_ = []
        pointer = 0
        for i, t in enumerate(tab[0]):
            Lambda = int(fs*d[0][i])
            s = t[ Gamma[pointer:pointer+Lambda] % l ]
            pointer += Lambda
            s_.append(s)
        s =  t[ Gamma[pointer:] % l ]
        s_.append(s)
        s = n.hstack(s_)
        s[:len(IID_a)] *= IID_a
        s[len(IID_a):] *= IID_a[-1]
    else:
        # left channel
        Vl = V_ + [F_[0]]
        F = n.prod(Vl, axis=0)
        Gamma = n.cumsum( F*l/fs ).astype(n.int)
        s_ = []
        pointer = 0
        for i, t in enumerate(tab[0]):
            Lambda = int(fs*d[0][i])
            s = t[ Gamma[pointer:pointer+Lambda] % l ]
            pointer += Lambda
            s_.append(s)
        s =  t[ Gamma[pointer:] % l ]
        s_.append(s)
        TL = n.hstack(s_)
        TL[:len(IID_a[0])] *=  IID_a[0]
        TL[len( IID_a[0]):] *= IID_a[0][-1]

        # right channel
        Vr = V_ + [F_[1]]
        F = n.prod(Vr, axis=0)
        Gamma = n.cumsum( F*l/fs ).astype(n.int)
        s_ = []
        pointer = 0
        for i, t in enumerate(tab[0]):
            Lambda = int(fs*d[0][i])
            s = t[ Gamma[pointer:pointer+Lambda] % l ]
            pointer += Lambda
            s_.append(s)
        s =  t[ Gamma[pointer:] % l ]
        s_.append(s)
        TR = n.hstack(s_)
        TR[:len(IID_a[1])] *=  IID_a[1]
        TR[len( IID_a[1]):] *= IID_a[1][-1]

        if x[0] > 0:
            TL = n.hstack(( n.zeros(int(Lambda_ITD)), TL ))
            TR = n.hstack(( TR, n.zeros(int(Lambda_ITD)) ))
        else:
            TL = n.hstack(( TL, n.zeros(-int(Lambda_ITD)) ))
            TR = n.hstack(( n.zeros(-int(Lambda_ITD)), TR ))
        s = n.vstack(( TL, TR ))
    return s


def rhythymToDurations(durations=[4, 2, 2, 4, 1,1,1,1, 2, 2, 4],
        frequencies=None, duration=.25, BPM=None, total_duration=None):
    """
    Returns durations from rhythmic patterns.

    Parameters
    ----------
    durations : interable of scalars
        The relative durations of each item (e.g. note).
    frequencies : iterable of scalars
        The number of the entry's duration that fits into the pulse.
        If supplied, durations is ignored.
    duration : scalar
        A basic duration (e.g. for the pulse) in seconds.
    BPM : scalar
        The number of beats per second.
        If supplied, duration is ignored.
    total_duration: scalar
        The total duration of the sequence in seconds.
        If supplied, both BPM and duration are ignored.

    Returns
    -------
    durs : List of durations in seconds.

    Examples
    --------
    >>> dt = [4, 2, 2, 4, 1,1,1,1, 2, 2, 4]
    >>> durs0 = rhythymToDurations(dt, duration=.25)
    >>> df = [4, 8, 8, 4, 16, 16, 16, 16, 8, 8, 4]
    >>> durs0_ = rhythymToDurations(frequencies=df, duration=4)
    >>> dtut = [4,2,2, [8, 1,1,1], 4, [4, 1,1,.5,.5], 3,1, 3,1, 4]
    >>> durs1 = rhythymToDurations(dtut)
    >>> dtuf2 = [4,8,8, [2, 3,3,3], 4, [4, 3,3,6,6], 16/3, 16, 16/3, 16, 4]
    >>> durs1_ = rhythymToDurations(frequencies=dtut2, duration=4)
    
    Notes
    -----
    The durations parameter is considered to be in a temporal notation
    for durations/rhythm: each entry is a relative duration to
    be multiplied by the base duration given through duration,
    BPM or total_duration.
    >>> durs = [i*duration for i in durations]

    The frequencies parameter is considered to be in a
    frequential notation: each entry is the number of the
    entry that fits a same duration (also given through duration,
    BPM or total_duration).
    >>> durs = [duration/i for i in frequencies]

    The examples above yield (two by two) the same sequences of durations
    by using duration=0.25 when in temporal notation or
    duration=4 when in frequency notation.

    To facilitate the description of rhythms (e.g. for tuplets),
    some set of durations might be an iterable inside durations
    or frequencies. In this case:
        ### if mode is temporal:
            total_dur = cell[0]*duration
            # durations are proportional to remaining values:
            d_ = [i/sum(cell[1:]) for i in cell[1:]]
            durs = [i*total_dur for i in d_]
        ### if mode is frequential:
            total_dur = duration/cell[0]
            # durations are inversely proportional to remaining values:
            d_ = [i/sum(cell[1:]) for i in cell[1:]]
            durs = [i*total_dur for i in d_]

    An example for achieving the same sequence of durations through
    temporal or frequential notation and with cells for tuplets
    is the last two sequences of the examples.

    It might be a good idea to incorporate also this notation:
        d2 = [1, 4, 1, 4]  # [quarter note + 4 sixteenth notes] x 2

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if not BPM and not total_duration:
        dur = duration
    elif BPM:
        dur = BPM/60
    else:
        dur = None
    durs = []
    if frequencies:
        if not dur:  # obtain from total_dur
            durs_ = [1/i if not isinstance(i, (list, tuple, n.ndarray)) else 1/i[0] 
                for i in frequencies]
            dur = total_duration/sum(durs_)
        for d in frequencies:
            if isinstance(d, (list, tuple, n.ndarray)):
                t_ = dur/d[0] # total timespan
                d_ = [1/i for i in d[1:]]  # relative durations from the frequency
                # normalize d_ to sum to t_
                d__ = [t_*i/sum(d_) for i in d_]
                # durs = [t_*i/sum(d[1:]) for i in d[1:]]
                durs.extend(d__)
            else:
                durs.append(dur/d)
    else:
        if not dur:  #obtain from total_dur
            durs_ = [i if not isinstance(i, (list, tuple, n.ndarray)) else i[0] 
                for i in durations]
            dur = total_duration/sum(durs_)
        for d in durations:
            if isinstance(d, (list, tuple, n.ndarray)):
                t_ = d[0]*dur  # total timespan
                # relative durations for the potential tuplet
                d_ = [i/sum(d[1:]) for i in d[1:]]
                # normalize d_ to fit t_
                d__ = [i*t_ for i in d_]
                # durs = [t_*i for i in d[1:]]
                durs.extend(d__)
            else:
                durs.append(d*dur)
    return durs
            
R = rhythymToDurations

def FM(f=220, d=2, fm=100, mu=2, tab=Tr, tabm=S,
        nsamples=0, fs=44100):
    """
    Synthesize a musical note with FM synthesis.
    
    Set fm=0 or mu=0 (or use N()) for a note without FM.
    A FM is a linear oscillatory pattern of frequency [1].
    
    Parameters
    ----------
    f : scalar
        The frequency of the note in Hertz.
    d : scalar
        The duration of the note in seconds.
    fm : scalar
        The frequency of the modulator in Hertz.
    mu : scalar
        The maximum deviation of frequency in the modulator in Hertz.
    tab : array_like
        The table with the waveform for the carrier.
    tabv : array_like
        The table with the waveform for the modulator.
    nsamples : integer
        The number of samples in the sound.
        If supplied, d is ignored.
    fs : integer
        The sample rate.

    Returns
    -------
    s : ndarray
        A numpy array where each value is a PCM sample of the note.

    See Also
    --------
    N : A basic musical note without vibrato.
    V : A musical note with an oscillation of pitch.
    T : A tremolo, an oscillation of loudness.
    AM : A linear oscillation of amplitude (not linear loudness).

    Examples
    --------
    >>> W(FM())  # writes a WAV file of a note
    >>> s = H( [FM(i, j) for i, j in zip([200, 500, 100], [2, 1, 2])] )
    >>> s2 = FM(440, 1.5, 600, 10)

    Notes
    -----
    In the MASS framework implementation,
    for a sound with a vibrato (or FM) to be synthesized using LUT,
    the vibrato (or FM)
    pattern is considered when performing the lookup calculations.

    The tremolo and AM patterns are implemented as separate amplitude envelopes.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    tab = n.array(tab)
    tabm = n.array(tabm)
    if nsamples:
        Lambda = nsamples
    else:
        Lambda = int(fs*d)
    samples = n.arange(Lambda)

    lm = len(tabm)
    Gammam = (samples*fm*lm/fs).astype(n.int)  # LUT indexes
    # values of the oscillatory pattern at each sample
    Tm = tabm[ Gammam % lm ] 

    # frequency in Hz at each sample
    F = f + Tm*mu 
    l = len(tab)
    D_gamma = F*(l/fs)  # shift in table between each sample
    Gamma = n.cumsum(D_gamma).astype(n.int)  # total shift at each sample
    s = tab[ Gamma % l ]  # final sample lookup
    return s


def AM(d=2, fm=50, a=.4, taba=S, nsamples=0, sonic_vector=0, fs=44100):
    """
    Synthesize an AM envelope or apply it to a sound.
    
    Set fm=0 or a=0 for a constant envelope with value 1.
    An AM is a linear oscillatory pattern of amplitude [1].
    
    Parameters
    ----------
    d : scalar
        The duration of the envelope in seconds.
    fm : scalar
        The frequency of the modultar in Hertz.
    a : scalar in [0,1]
        The maximum deviation of amplitude of the AM.
    tabm : array_like
        The table with the waveform for the tremolo oscillatory pattern.
    nsamples : integer
        The number of samples of the envelope. If supplied, d is ignored.
    sonic_vector : array_like
        Samples for the tremolo to be applied to.
        If supplied, d and nsamples are ignored.
    fs : integer
        The sample rate.

    Returns
    -------
    T : ndarray
        A numpy array where each value is a PCM sample
        of the envelope.
        if sonic_vector is 0.
        If sonic_vector is input,
        T is the sonic vector with the AM applied to it.

    See Also
    --------
    V : A musical note with an oscillation of pitch.
    FM : A linear oscillation of fundamental frequency.
    T : A tremolo, an oscillation of loudness.

    Examples
    --------
    >>> W(V()*AM())  # writes a WAV file of a note with tremolo
    >>> s = H( [V()*AM(fm=i, a=j) for i, j in zip([60, 150, 100], [2, 1, 20])] )  # OR
    >>> s = H( [AM(fm=i, a=j, sonic_vector=V()) for i, j in zip([60, 150, 100], [2, 1, 20])] )
    >>> envelope2 = AM(440, 150, 60)  # a lengthy envelope

    Notes
    -----
    In the MASS framework implementation, for obtaining a sound with a tremolo (or AM),
    the tremolo pattern is considered separately from a synthesis of the sound.

    The vibrato and FM patterns are considering when synthesizing the sound.

    One might want to run this function twice to obtain
    a stereo reverberation.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """

    taba = n.array(taba)
    if type(sonic_vector) in (n.ndarray, list):
        Lambda = len(sonic_vector)
    elif nsamples:
        Lambda = nsamples
    else:
        Lambda = n.floor(fs*d)
    samples = n.arange(Lambda)

    l = len(taba)
    Gammaa = (samples*fs*l/fs).astype(n.int)  # indexes for LUT
    # amplitude variation at each sample
    Ta = taba[ Gammaa % l ] 
    T = 1 + Ta*a
    if type(sonic_vector) in (n.ndarray, list):
        return T*sonic_vector
    else:
        return T

def noises(ntype="brown", d=2, fmin=15, fmax=15000, nsamples=0, fs=44100):
    """
    Return a colored or user-refined noise.

    Parameters
    ----------
    ntype : string or scalar
        Specifies the decibels gain or attenuation per octave.
        It can be specified numerically 
        (e.g. ntype=3.5 is 3.5 decibels gain per octave)
        or by strings:
          "brown" is -6dB/octave
          "pink" is -3dB/octave
          "white" is 0dB/octave
          "blue" is 3dB/octave
          "violet" is 6dB/octave
          "black" is -12/dB/octave but, in theory, is any < -6dB/octave
        See [1] for more information.
    d : scalar
        The duration of the noise in seconds.
    fmin : scalar in [0, fs/2]
        The lowest frequency allowed.
    fmax : scalar in [0, fs/2]
        The highest frequency allowed.
        It should be > fmin.
    nsamples : integer
        The number of samples of the resulting sonic vector.

    Notes
    -----
    The noise is synthesized with components with random phases,
    with the moduli that are related to the decibels/octave,
    and with a frequency resolution of
      fs/nsamples = fs/(fs*d) = 1/d Hz

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound."
    arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if nsamples:
        Lambda = nsamples
    else:
        Lambda = int(d*fs)
    if ntype == "white":
        prog = 0
    elif ntype == "pink":
        prog = -3
    elif ntype == "brown":
        prog = -6
    elif ntype == "blue":
        prog = 3
    elif ntype == "violet":
        prog = 6
    elif ntype == "black":
        prog = -12
    elif isinstance(ntype, Number):
        prog = ntype
    else:
        print("Set ntype to a number or one of the following strings:\
                'white', 'pink', 'brown', 'blue', 'violet', 'black'.\
                Check docstring for more information.")
        return
    # random phases
    coefs = n.zeros(Lambda)
    coefs[:Lambda//2] = n.exp(1j*n.random.uniform(0, 2*n.pi, Lambda//2))
    if Lambda%2==0:
        coefs[Lambda/2] = 1.  # max freq is only real (as explained in Sec. 2.5)

    df = fs/Lambda
    i0 = n.floor(fmin/df)  # first coefficient to be considered
    il = n.floor(fmax/df)  # last coefficient to be considered
    coefs[:i0] = 0
    coefs[il:] = 0

    factor = 10.**(prog/20.)
    fi = n.arange(coefs.shape[0])*df # frequencies related to the coefficients
    alphai = factor**(n.log2(fi[i0:il]/fmin))
    coefs[i0:il] *= alphai

    # coefficients have real part even and imaginary part odd
    if Lambda%2 == 0:
        coefs[Lambda//2+1:] = n.conj(coefs[1:-1][::-1])
    else:
        coefs[Lambda//2+1:] = n.conj(coefs[1:][::-1])

    # Achievement of the temporal samples of the noise
    noise = n.fft.ifft(coefs).real
    return noise


def FIR(samples, sonic_vector, freq=True, max_freq=True):
    """
    Apply a FIR filter to a sonic_array.

    Parameters
    ----------
    samples : array_like
        A sequence of absolute values for the frequencies
        (if freq=True) or samples of an impulse response.
    sonic_vector : array_like
        An one-dimensional array with the PCM samples of
        the signal (e.g. sound) for the FIR filter
        to be applied to.
    freq : boolean
        Set to True if samples are frequency absolute values
        or False if samples is an impulse response.
        If max_freq=True, the separations between the frequencies
        are fs/(2*N-2).
        If max_freq=False, the separation between the frequencies
        are fs/(2*N-1).
        Where N is the length of the provided samples.
    max_freq : boolean
        Set to true if the last item in the samples is related
        to the Nyquist frequency fs/2.
        Ignored if freq=False.

    Notes
    -----
    If freq=True, the samples are the absolute values of
    the frequency components.
    The phases are set to zero to maintain the phases
    of the components of the original signal.

    """
    if not freq:
        return convolve(samples, sonic_vector)
    if max_freq:
        s = n.hstack(( samples, samples[1:-1][::-1] ))
    else:
        s = n.hstack(( samples, samples[1:][::-1] ))
    return convolve(samples, sonic_vector)

def IIR(sonic_vector, A, B):
    """
    Apply an IIR filter to a signal.
    
    Parameters
    ----------
    sonic_vector : array_like
        An one dimensional array representing the signal
        (potentially a sound) for the filter to by applied to.
    A : iterable of scalars
        The feedforward coefficients.
    B : iterable of scalars
        The feedback filter coefficients.

    Notes
    -----
    Check [1] to know more about this function.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound."
    arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    signal = sonic_vector
    signal_ = []
    for i in range(len(signal)):
        samples_A = signal[i::-1][:len(A)]
        A_coeffs = A[:i+1]
        A_contrib = (samples_A*A_coeffs).sum()

        samples_B = signal_[-1:-1-i:-1][:len(B)-1]
        B_coeffs = B[1:i+1]
        B_contrib = (samples_B*B_coeffs).sum()
        t_i = (A_contrib + B_contrib)/B[0]
        signal_.append(t_i)
    return n.array(signal_)


def R(d=1.9, d1=0.15, decay=-50, stat="brown", sonic_vector=0, fs=44100):
    """
    Apply an artificial reverberation or return the impulse response.

    Parameters
    ----------
    d : scalar
        The total duration of the reverberation in seconds.
    d1 : scalar
        The duration of the first phase of the reverberation
        in seconds.
    decay : scalar
        The total decay of the last incidence in decibels.
    stat : string or scalar
        A string or scalar specifying the noise.
        Passed to noises(ntype=scalar).
    sonic_vector : array_like
        An optional one dimensional array for the reverberation to
        be applied.
    fs : scalar
        The sampling frequency.

    Returns
    -------
    s : numpy.ndarray
        An array if the impulse response of the reverberation
        (if sonic_vector is not specified),
        or with the reverberation applied to sonic_vector.

    Notes
    -----
    This is a simple artificial reverberation with a progressive
    loudness decay of the reincidences of the sound and with
    two periods: the first consists of scattered reincidences,
    the second period reincidences is modeled by a noise.

    Comparing with the description in [1], the frequency bands
    are ignored.

    One might want to run this function twice to obtain
    a stereo reverberation.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound."
    arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    Lambda = int(d*fs)
    Lambda1 =  int(d1*fs)
    # Sound reincidence probability probability in the first period:
    ii = n.arange(Lambda)
    P = (ii[:Lambda1]/Lambda1)**2.
    # incidences:
    R1_ = n.random.random(Lambda1) < P
    A = 10.**( (decay1/20)*(ii/(Lambda-1)) )
    ### Eq. 76 First period of reverberation:
    R1 = R1_*A[:Lambda1]  # first incidences

    ### Eq. 77 Second period of reverberation:
    noise = noises(ntype, fmax=fs/2, nsamples=Lambda-Lambda1)
    R2 = noise*A[Lambda1:Lambda]
    ### Eq. 78 Impulse response of the reverberation
    R = n.hstack((R1,R2))
    R[0] = 1.
    if type(sonic_vector) in (n.ndarray, list):
        return convolve(sonic_vector, R)
    else:
        return R


test = False
if __name__ == "__main__" and test:
        import doctest
        doctest.testmod()

