import numpy as np

def noise(noise_type="brown", duration=2, min_freq=15, max_freq=15000,
          number_of_samples=0, sample_rate=44100):
    """
    Return a colored or user-refined noise.

    Parameters
    ----------
    noise_type : string or scalar
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
    duration : scalar
        The duration of the noise in seconds.
    min_freq : scalar in [0, fs/2]
        The lowest frequency allowed.
    max_freq : scalar in [0, fs/2]
        The highest frequency allowed.
        It should be > fmin.
    number_of_samples : integer
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
    if number_of_samples:
        Lambda = number_of_samples
    else:
        Lambda = int(duration * sample_rate)
    if noise_type == "white":
        prog = 0
    elif noise_type == "pink":
        prog = -3
    elif noise_type == "brown":
        prog = -6
    elif noise_type == "blue":
        prog = 3
    elif noise_type == "violet":
        prog = 6
    elif noise_type == "black":
        prog = -12
    elif isinstance(noise_type, Number):
        prog = noise_type
    else:
        print("Set ntype to a number or one of the following strings:\
                'white', 'pink', 'brown', 'blue', 'violet', 'black'.\
                Check docstring for more information.")
        return
    # random phases
    coefs = np.zeros(Lambda)
    coefs[:Lambda//2] = np.exp(1j * np.random.uniform(0, 2 * np.pi, Lambda // 2))
    if Lambda%2==0:
        coefs[Lambda/2] = 1.  # max freq is only real (as explained in Sec. 2.5)

    df = sample_rate / Lambda
    i0 = np.floor(min_freq / df)  # first coefficient to be considered
    il = np.floor(max_freq / df)  # last coefficient to be considered
    coefs[:i0] = 0
    coefs[il:] = 0

    factor = 10.**(prog/20.)
    fi = np.arange(coefs.shape[0]) * df # frequencies related to the coefficients
    alphai = factor**(np.log2(fi[i0:il] / min_freq))
    coefs[i0:il] *= alphai

    # coefficients have real part even and imaginary part odd
    if Lambda%2 == 0:
        coefs[Lambda//2+1:] = np.conj(coefs[1:-1][::-1])
    else:
        coefs[Lambda//2+1:] = np.conj(coefs[1:][::-1])

    # Achievement of the temporal samples of the noise
    noise = np.fft.ifft(coefs).real
    return noise


def makeGaussianNoise(self,mean,std,DUR=2):
    Lambda = DUR*self.samples_beat # Lambda sempre par
    df = self.samplerate/float(Lambda)
    MEAN=mean
    STD=.1
    coefs = np.exp(1j * np.random.uniform(0, 2 * np.pi, Lambda))
    # real par, imaginaria impar
    coefs[Lambda/2+1:] = np.real(coefs[1:Lambda / 2])[::-1] - 1j * \
                         np.imag(coefs[1:Lambda / 2])[::-1]
    coefs[0] = 0.  # sem bias
    if Lambda%2==0:
        coefs[Lambda/2] = 0.  # freq max eh real simplesmente

    # as frequências relativas a cada coeficiente
    # acima de Lambda/2 nao vale
    fi = np.arange(coefs.shape[0]) * df
    f0 = 15.  # iniciamos o ruido em 15 Hz
    f1=(mean-std/2)*3000
    f2=(mean+std/2)*3000
    i1 = np.floor(f1 / df)  # primeiro coef a valer
    i2 = np.floor(f2 / df)  # ultimo coef a valer
    coefs[:i1] = np.zeros(i1)
    coefs[i2:]=np.zeros(len(coefs[i2:]))

    # obtenção do ruído em suas amostras temporais
    ruido = np.fft.ifft(coefs)
    r = np.real(ruido)
    r = ((r-r.min())/(r.max()-r.min()))*2-1

    # fazer tre_freq variar conforme measures2
    return r