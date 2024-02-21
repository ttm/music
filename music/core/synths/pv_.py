import numpy as n
from music.utils import S, Tr

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

    # find maximum size, fill others with ones
    V_ = [Ft] + V_
    amax = max([len(i) for i in V_])
    for i, contrib in enumerate(V_[1:]):
        V_[i+1] = n.hstack(( contrib, n.ones(amax - len(contrib)) ))
    V_[0] = n.hstack(( V_[0], n.ones(amax - len(V_[0]))*f[-1] ))

    F = n.prod(V_, axis=0)
    l = len(tab[0][0])
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
    return s
