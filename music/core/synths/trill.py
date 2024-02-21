import numpy as n
from music.utils import Tr
from music.core.filters import AD
from music.core.synths import N

def trill(f=[440,440*2**(2/12)], ft=17, d=5, fs=44100):
    """
    Make a trill.

    This is just a simple function for exemplifying
    the synthesis of trills.
    The user is encouraged to make its own functions
    for trills and set e.g. ADSR envelopes, tremolos
    and vibratos as intended.

    Parameters
    ----------
    f : iterable of scalars
        Frequencies to the iterated.
    ft : scalar
        The number of notes per second.
    d : scalar
        The maximum duration of the trill in seconds.
    fs : integer
        The sample rate.

    See Also
    --------
    V : A note with vibrato.
    PV_ : a note with an arbitrary sequence of pitch transition and a meta-vibrato.
    T : A tremolo envelope.

    Returns
    -------
    s : ndarray
        The PCM samples of the resulting sound.

    Examples
    --------
    >>> W(trill())

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    nsamples = 44100/ft
    pointer = 0
    i = 0
    s = []
    while pointer+nsamples < d*44100:
        ns = int(nsamples*(i+1) - pointer)
        note = N(f[i%len(f)], nsamples=ns,
                tab=Tr(), fs=fs)
        s.append(AD(sonic_vector=note, R=10))
        pointer += ns
        i += 1
    trill = n.hstack(s)
    return trill
