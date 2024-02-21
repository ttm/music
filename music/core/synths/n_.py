import numpy as n
from music.utils import Tr

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
    Gamma = (i0 + samples*f*l/fs).astype(n.int64)
    s = tab[ Gamma % l ]
    return s