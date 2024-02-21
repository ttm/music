import numpy as n
from music.utils import S

def PV(f1=220, f2=440, d=2, fv=4, nu=2, alpha=1,
        alphav=1, tab=S(), tabv=S(), nsamples=0, fs=44100):
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
    Gammav = (samples*fv*lv/fs).astype(n.int64)  # LUT indexes
    # values of the oscillatory pattern at each sample
    Tv = tabv[ Gammav % lv ] 

    if alpha != 1 or alphav != 1:
        F = f1*(f2/f1)**( (samples / (Lambda-1))**alpha )*2.**( (Tv*nu/12)**alphav )
    else:
        F = f1*(f2/f1)**( samples / (Lambda-1) )*2.**( (Tv*nu/12)**alpha )
    l = len(tab)
    Gamma = n.cumsum( F*l/fs ).astype(n.int64)
    s = tab[ Gamma % l ]
    return s
