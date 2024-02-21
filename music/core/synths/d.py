import numpy as n
from music.utils import Tr

def D(f=220, d=2, tab=Tr(), x=[-10, 10], y=[1,1], stereo=True,
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

        Gamma = n.cumsum(fl*l/fs).astype(n.int64)
        sl = tab[ Gamma % l ]*IID_al[:-1]

        Gamma = n.cumsum(fr*l/fs).astype(n.int64)
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

        Gamma = n.cumsum(f_*l/fs).astype(n.int64)
        s = tab[ Gamma % l ]*IID[:-1]
    return s
