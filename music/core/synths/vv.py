import numpy as n
from music.utils import S, Tr

def VV(f=220, d=2, fv1=2, fv2=6, nu1=2, nu2=4, alphav1=1,
        alphav2=1, tab=Tr(), tabv1=S(), tabv2=S(), nsamples=0, fs=44100):
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
    Gammav1 = (samples*fv1*lv1/fs).astype(n.int64)  # LUT indexes
    # values of the oscillatory pattern at each sample
    Tv1 = tabv1[ Gammav1 % lv1 ] 

    lv2 = len(tabv2)
    Gammav2 = (samples*fv2*lv2/fs).astype(n.int64)  # LUT indexes
    # values of the oscillatory pattern at each sample
    Tv2 = tabv1[ Gammav2 % lv2 ] 

    if alphav1 != 1 or alphav2 != 1:
        F = f*2.**( (Tv1*nu1/12)**alphav1 )*2.**( (Tv2*nu2/12)**alphav2 )
    else:
        F = f*2.**( (Tv1*nu1/12))*2.**( (Tv2*nu2/12))
    l = len(tab)
    Gamma = n.cumsum( F*l/fs ).astype(n.int64)
    s = tab[ Gamma % l ]
    return s

