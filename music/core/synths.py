import numpy as n, music as M

__doc__ = """Basic routines of synthesizers

All routines are directly derived from the
MASS framework: https://github.com/ttm/mass
Should be used at ../synths.py"""


def V_(st=0, f=220, d=2., fv=2., nu=2., tab=Tr_i, tabv=S_):
    """A shorthand for using V() with semitones"""

    f_ = f*2**(st/12)
    return V(f=_, d=2., fv=2., nu=2., tab=Tr_i, tabv=S_)


def V(f=220, d=2., fv=2., nu=2., tab=Tr_i, tabv=S_):
    """
    Synthesize a musical note with a vibrato.
    
    Set fv=0 for a note without vibrato.
    A vibrato is an oscillation of the pitch [1].
    
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
        The table with the waveform of the vibrato oscillatory pattern.

    Returns
    -------
    s : ndarray
        A numpy array where each value is a PCM sample of the sound.

    See Also
    --------
    T : an oscillation of amplitude.
    FM : a linear oscillation of fundamental frequency.
    AM : a linear oscillation of amplitude.
    V_ : a shorthand to render a note with vibrato using
        a reference frequency and a pitch interval.
    raw.vibrato : a very slim implementation of this function.

    Examples
    --------
    >>> W(V())  # writes a WAV file of a note
    >>> s = H( [V(i, j) for i, j in zip([200, 500, 100], [2, 1, 2])] )
    >>> s2 = V(440, 1.5, 6, 1)

    Notes
    -----
    In the MASS framework implementation, for a sound with a vibrato (or FM) to be synthesized using LUT,
    the vibrato pattern is considered when performing the lookup calculations.

    The tremolo and AM patterns implemented as separate amplitude envelopes.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    Lambda=n.floor(f_s*d)
    ii=n.arange(Lambda)
    Lv=float(len(tabv))

    Gammav_i=n.floor(ii*fv*Lv/f_s) # índices para a LUT
    Gammav_i=n.array(Gammav_i,n.int)
    # padrão de variação do vibrato para cada amostra
    Tv_i=tabv[Gammav_i%int(Lv)] 

    # frequência em Hz em cada amostra
    F_i=f*(   2.**(  Tv_i*nu/12.  )   ) 
    # a movimentação na tabela por amostra
    D_gamma_i=F_i*(Lt/float(f_s))
    Gamma_i=n.cumsum(D_gamma_i) # a movimentação na tabela total
    Gamma_i=n.floor( Gamma_i) # já os índices
    Gamma_i=n.array( Gamma_i, dtype=n.int) # já os índices
    return tab[Gamma_i%int(Lt)] # busca dos índices na tabela

