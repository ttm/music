import numpy as n
from ...utils import Tr

def N(f=220, d=2, tab=Tr(), nsamples=0, fs=44100):
    """
    Synthesize a basic musical note.

    Parameters
    ----------
    f : scalar
        The frequency of the note in Hertz.
    d : scalar
        The duration of the note in seconds.
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
    V : A note with vibrato.
    T : A tremolo envelope.

    Examples
    --------
    >>> W(N())  # writes a WAV file of a note
    >>> s = H( [N(i, j) for i, j in zip([200, 500, 100], [2, 1, 2])] )
    >>> s2 = N(440, 1.5, tab=Sa)

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

    Gamma = (samples*f*l/fs).astype(n.int64)
    s = tab[ Gamma % l ]
    return s
