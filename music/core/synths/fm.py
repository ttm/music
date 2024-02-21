import numpy as n
from music.utils import S, Tr

def FM(f=220, d=2, fm=100, mu=2, tab=Tr(), tabm=S(),
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
    Gammam = (samples*fm*lm/fs).astype(n.int64)  # LUT indexes
    # values of the oscillatory pattern at each sample
    Tm = tabm[ Gammam % lm ] 

    # frequency in Hz at each sample
    F = f + Tm*mu 
    l = len(tab)
    D_gamma = F*(l/fs)  # shift in table between each sample
    Gamma = n.cumsum(D_gamma).astype(n.int64)  # total shift at each sample
    s = tab[ Gamma % l ]  # final sample lookup
    return s
