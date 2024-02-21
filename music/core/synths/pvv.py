import numpy as n
from music.utils import S, Tr

def PVV(f1=220, f2=440, d=2, fv1=2, fv2=6, nu1=2, nu2=.5, alpha=1,
        alphav1=1, alphav2=1, tab=Tr(), tabv1=S(), tabv2=S(), nsamples=0, fs=44100):
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
    Gammav1 = (samples*fv1*lv1/fs).astype(n.int64)  # LUT indexes
    # values of the oscillatory pattern at each sample
    Tv1 = tabv1[ Gammav1 % lv1 ] 

    lv2 = len(tabv2)
    Gammav2 = (samples*fv2*lv2/fs).astype(n.int64)  # LUT indexes
    # values of the oscillatory pattern at each sample
    Tv2 = tabv1[ Gammav2 % lv2 ] 

    if alpha !=1 or alphav1 != 1 or alphav2 != 1:
        F = f1*(f2/f1)**( (samples / (Lambda-1))**alpha )*2.**( (Tv1*nu1/12)**alphav1 )*2.**( (Tv2*nu2/12)**alphav2 )
    else:
        F = f1*(f2/f1)**( samples / (Lambda-1) )*2.**( (Tv1*nu1/12))*2.**( (Tv2*nu2/12))
    l = len(tab)
    Gamma = n.cumsum( F*l/fs ).astype(n.int64)
    s = tab[ Gamma % l ]
    return s

