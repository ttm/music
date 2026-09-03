"""Module for the synthesis of noises and silences."""
from numbers import Real
import numpy as np
from numpy.typing import NDArray
import music


def noise(noise_type: str | float = "brown", duration: float = 2,
          min_freq: float = 15, max_freq: float = 15000,
          number_of_samples: int = 0,
          sample_rate: int = 44100) -> NDArray[np.float64]:
    """
    Return a colored or user-refined noise.

    Parameters
    ----------
    noise_type : string or scalar
        Specifies the decibels gain or attenuation per octave. It can be
        specified numerically (e.g. ntype=3.5 is 3.5 decibels gain per octave)
        or by strings:
        - "brown" is -6dB/octave
        - "pink" is -3dB/octave
        - "white" is 0dB/octave
        - "blue" is 3dB/octave
        - "violet" is 6dB/octave
        - "black" is -12/dB/octave but, in theory, is any < -6dB/octave
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
    sample_rate : integer
        The sample rate to use, by default 44100.

    Returns
    -------
    ndarray
        A mono sequence of PCM samples, normalized to [-1, 1].

    Raises
    ------
    ValueError
        If ``noise_type`` is neither a number of decibels per octave nor
        one of the named colours. A misspelt colour would otherwise
        render as silence or as an unrelated slope.

    Notes
    -----
    The noise is synthesized with components with random phases, with the
    moduli that are related to the decibels/octave, and with a frequency
    resolution of fs / nsamples = fs / (fs*d) = 1/d Hz

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if number_of_samples:
        length = number_of_samples
    else:
        length = int(duration * sample_rate)
    prog: float
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
    elif isinstance(noise_type, Real):
        # Real rather than Number: a complex gain per octave is meaningless,
        # and float() would reject it anyway.
        prog = float(noise_type)
    else:
        raise ValueError(
            "Set ntype to a number or one of the following strings: "
            "'white', 'pink', 'brown', 'blue', 'violet', 'black'. "
            "Check docstring for more information.")

    coeffs = np.zeros(length, dtype=complex)
    coeffs[:length // 2] = np.exp(1j *
                                  np.random.uniform(0, 2 * np.pi, length // 2))
    if length % 2 == 0:
        coeffs[length // 2] = 1.

    freq_res = sample_rate / length
    first_coeff = int(np.floor(min_freq / freq_res))
    first_coeff = max(1, first_coeff)
    last_coeff = int(np.floor(max_freq / freq_res))
    coeffs[:first_coeff] = 0
    coeffs[last_coeff:] = 0

    factor = 10. ** (prog / 20.)
    freq_i = np.arange(coeffs.shape[0]) * freq_res
    denom = max(min_freq, freq_res)
    freqs = np.clip(freq_i[first_coeff:last_coeff], freq_res, None)
    attenuation_factors = factor ** (np.log2(freqs / denom))
    coeffs[first_coeff:last_coeff] *= attenuation_factors

    if length % 2 == 0:
        high_freq_conj_coeffs = np.conj(coeffs[1:length // 2][::-1])
        coeffs[length // 2 + 1:] = high_freq_conj_coeffs
    else:
        high_freq_conj_coeffs = np.conj(coeffs[1:length // 2][::-1])
        coeffs[length // 2 + 1:-1] = high_freq_conj_coeffs

    noise_vector = np.fft.ifft(coeffs).real
    return music.core.normalize_mono(noise_vector)


def gaussian_noise(mean: float = 1, std: float = 0.5, duration: float = 2,
                   sample_rate: int = 44100) -> NDArray[np.float64]:
    """Synth gaussian noise

    Parameters
    ----------
    mean : int, optional
        _description_, by default 1
    std : float, optional
        _description_, by default 0.5
    duration : int, optional
        How long in seconds will the noise be, by default 2
    sample_rate : int, optional
        The sample rate to use, by default 44100

    Returns
    -------
    array
        An array for the gaussian noise
    """

    # int(): the length indexes arrays and sets a sample count, so a
    # fractional duration raised TypeError out of np.random.uniform
    # rather than rendering the half second it was asked for.
    length = int(duration * sample_rate)
    freq_res = sample_rate / float(length)
    coeffs = np.exp(1j * np.random.uniform(0, 2 * np.pi, length))
    coeffs[length // 2 + 1:] = np.real(coeffs[1:length // 2])[::-1] - 1j * \
        np.imag(coeffs[1:length // 2])[::-1]
    coeffs[0] = 0.  # sem bias
    if length % 2 == 0:
        coeffs[length // 2] = 0.
    f1 = (mean - std / 2) * 3000
    f2 = (mean + std / 2) * 3000
    first_coeff = int(np.floor(f1 / freq_res))
    last_coeff = int(np.floor(f2 / freq_res))
    coeffs[:first_coeff] = np.zeros(first_coeff)
    coeffs[last_coeff:] = np.zeros(len(coeffs[last_coeff:]))

    # obtenção do ruído em suas amostras temporais
    noise_vector = np.real(np.fft.ifft(coeffs))
    noise_vector = ((noise_vector - noise_vector.min()) /
                    (noise_vector.max() - noise_vector.min())) * 2 - 1

    # fazer tre_freq variar conforme measures2
    return music.core.normalize_mono(noise_vector)


def silence(duration: float = 1.0,
            sample_rate: int = 44100) -> NDArray[np.float64]:
    """Generate a silence of specified length.

    Parameters
    ----------
    duration : int, optional
        How many seconds will silence last, by default 1
    sample_rate : int, optional
        The sample rate to use, by default 44100

    Returns
    -------
    array
        An array with no sound
    """

    return np.zeros(int(duration * sample_rate))
