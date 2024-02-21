import numpy as n
from music.utils import S, Tr

def D_(f=[220, 440, 330], d=[[2,3],[2,5,3], [2,5,6,1,.4],[4,6,1]],
        fv=[[2,6,1], [.5,15,2,6,3]], nu=[[2,1, 5], [4,3,7,10,3]],
        alpha=[[1, 1] , [1,1,1], [1,1,1,1,1], [1,1,1]],
        x=[-10,10,5,3], y=[1,1,.1,.1], method=['lin','exp','lin'],
        tab=[[Tr(),Tr()], [S(),Tr(),S()], [S(),S(),S(),S(),S()]], stereo=True,
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
            Gammav = (samples*fv[i][j]*lv/fs).astype(n.int64)  # LUT indexes
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
        Gamma = n.cumsum( F*l/fs ).astype(n.int64)
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
        Gamma = n.cumsum( F*l/fs ).astype(n.int64)
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
        Gamma = n.cumsum( F*l/fs ).astype(n.int64)
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
