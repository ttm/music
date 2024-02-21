import numpy as n
from ...utils import S, Tr

def V(f=220, d=2, fv=4, nu=2, tab=Tr(), tabv=S(),
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
    Gammav = (samples*fv*lv/fs).astype(n.int64)  # LUT indexes
    # values of the oscillatory pattern at each sample
    Tv = tabv[ Gammav % lv ] 

    # frequency in Hz at each sample
    if alpha == 1:
        F = f*2.**(  Tv*nu/12  ) 
    else:
        F = f*2.**(  (Tv*nu/12)**alpha  ) 
    l = len(tab)
    D_gamma = F*(l/fs)  # shift in table between each sample
    Gamma = n.cumsum(D_gamma).astype(n.int64)  # total shift at each sample
    s = tab[ Gamma % l ]  # final sample lookup
    return s