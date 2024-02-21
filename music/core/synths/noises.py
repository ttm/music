import numpy as n

def noises(ntype="brown", d=2, fmin=15, fmax=15000, nsamples=0, fs=44100):
    """
    Return a colored or user-refined noise.

    Parameters
    ----------
    ntype : string or scalar
        Specifies the decibels gain or attenuation per octave.
        It can be specified numerically 
        (e.g. ntype=3.5 is 3.5 decibels gain per octave)
        or by strings:
          "brown" is -6dB/octave
          "pink" is -3dB/octave
          "white" is 0dB/octave
          "blue" is 3dB/octave
          "violet" is 6dB/octave
          "black" is -12/dB/octave but, in theory, is any < -6dB/octave
        See [1] for more information.
    d : scalar
        The duration of the noise in seconds.
    fmin : scalar in [0, fs/2]
        The lowest frequency allowed.
    fmax : scalar in [0, fs/2]
        The highest frequency allowed.
        It should be > fmin.
    nsamples : integer
        The number of samples of the resulting sonic vector.

    Notes
    -----
    The noise is synthesized with components with random phases,
    with the moduli that are related to the decibels/octave,
    and with a frequency resolution of
      fs/nsamples = fs/(fs*d) = 1/d Hz

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound."
    arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if nsamples:
        Lambda = nsamples
    else:
        Lambda = int(d*fs)
    if ntype == "white":
        prog = 0
    elif ntype == "pink":
        prog = -3
    elif ntype == "brown":
        prog = -6
    elif ntype == "blue":
        prog = 3
    elif ntype == "violet":
        prog = 6
    elif ntype == "black":
        prog = -12
    elif isinstance(ntype, Number):
        prog = ntype
    else:
        print("Set ntype to a number or one of the following strings:\
                'white', 'pink', 'brown', 'blue', 'violet', 'black'.\
                Check docstring for more information.")
        return
    # random phases
    coefs = n.zeros(Lambda)
    coefs[:Lambda//2] = n.exp(1j*n.random.uniform(0, 2*n.pi, Lambda//2))
    if Lambda%2==0:
        coefs[Lambda/2] = 1.  # max freq is only real (as explained in Sec. 2.5)

    df = fs/Lambda
    i0 = n.floor(fmin/df)  # first coefficient to be considered
    il = n.floor(fmax/df)  # last coefficient to be considered
    coefs[:i0] = 0
    coefs[il:] = 0

    factor = 10.**(prog/20.)
    fi = n.arange(coefs.shape[0])*df # frequencies related to the coefficients
    alphai = factor**(n.log2(fi[i0:il]/fmin))
    coefs[i0:il] *= alphai

    # coefficients have real part even and imaginary part odd
    if Lambda%2 == 0:
        coefs[Lambda//2+1:] = n.conj(coefs[1:-1][::-1])
    else:
        coefs[Lambda//2+1:] = n.conj(coefs[1:][::-1])

    # Achievement of the temporal samples of the noise
    noise = n.fft.ifft(coefs).real
    return noise
