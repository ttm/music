import numpy as n
from music.utils import S

def P(f1=220, f2=440, d=2, alpha=1, tab=S(), method="exp",
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
    Gamma = n.cumsum( F*l/fs ).astype(n.int64)
    s = tab[ Gamma % l ]
    return s
