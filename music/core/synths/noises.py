"""Module for the synthesis of noises and silences."""
import math
from numbers import Real
import numpy as np
from numpy.typing import NDArray
import music


def noise(noise_type: str | float = "brown", duration: float = 2,
          min_freq: float = 15, max_freq: float = 15000,
          number_of_samples: int = 0,
          sample_rate: int = 44100,
          seed: int | None = None) -> NDArray[np.float64]:
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
        The lowest frequency allowed. A component exactly at it is kept.
    max_freq : scalar in [0, fs/2]
        The highest frequency allowed, kept too if a component falls
        exactly on it. It should be > fmin.
    number_of_samples : integer
        The number of samples of the resulting sonic vector.
    sample_rate : integer
        The sample rate to use, by default 44100.
    seed : integer or None
        Optional deterministic seed for local random phases. Default None
        preserves legacy use of NumPy's global random state.

    Returns
    -------
    ndarray
        A mono sequence of PCM samples, normalized to [-1, 1].

    Raises
    ------
    ValueError
        If ``noise_type`` is neither a finite number of decibels per
        octave nor one of the named colours, if ``sample_rate`` is not
        positive, or if the band is empty: ``min_freq`` above ``max_freq``
        or above the Nyquist frequency. A misspelt colour would otherwise
        render as silence or as an unrelated slope, a slope of NaN as NaN,
        and an empty band as silence.

    Notes
    -----
    The noise is synthesized with components with random phases, with the
    moduli that are related to the decibels/octave, and with a frequency
    resolution of fs / nsamples = fs / (fs*d) = 1/d Hz

    The components are the multiples of that resolution from ``min_freq``
    to ``max_freq``, both included. The band used to be rounded down at
    both ends, which let in a component below ``min_freq`` whenever it
    fell between two, and always left out one at ``max_freq``. A noise too
    short to hold any component in the band is silence.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)


    Examples
    --------
    >>> hiss = noise("white", duration=0.5)     # flat, 0 dB per octave
    >>> len(hiss)
    22050
    >>> darker = noise("brown", duration=0.5)   # -6 dB per octave
    >>> tilted = noise(-9, duration=0.5)        # or any slope you name
    """
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
    # Checked before the length, so that a noise of no samples still
    # says what was wrong with the ones it was asked for.
    if not math.isfinite(prog):
        raise ValueError(
            f"noise_type must be a finite number of decibels per octave; "
            f"got {noise_type}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive; got {sample_rate}")
    if min_freq > max_freq:
        raise ValueError(
            f"min_freq ({min_freq} Hz) is above max_freq ({max_freq} Hz), "
            "so the band between them is empty")
    if min_freq > sample_rate / 2:
        raise ValueError(
            f"min_freq ({min_freq} Hz) is above the Nyquist frequency "
            f"({sample_rate / 2} Hz), so the band holds nothing")
    if seed is not None and (
            not isinstance(seed, (int, np.integer)) or isinstance(seed, bool)
            or not 0 <= seed < 2**64):
        raise ValueError("seed must be a nonnegative uint64 integer")
    if number_of_samples:
        length = number_of_samples
    else:
        length = int(duration * sample_rate)
    if length < 1:
        # Zero samples is zero samples, which is what every routine
        # here answers a zero duration with. The sequence operations
        # treat one as the identity it is: horizontal_stack and mix pass
        # it through, adsr shapes it into itself. Refusing would make
        # every caller that computes durations filter them first. Where
        # an empty sound stops being meaningful is at the sinks, and
        # normalize_mono says so there.
        return np.array([])

    # A random phase for the constant and every component below the
    # Nyquist frequency: (N + 1) // 2 of them. length // 2 drew one too
    # few for an odd length, whose highest component, (N - 1) / 2, was
    # always silent; an even length draws as many as it did.
    positive = (length + 1) // 2
    coeffs = np.zeros(length, dtype=complex)
    uniform = (np.random.uniform if seed is None
               else np.random.default_rng(seed).uniform)
    coeffs[:positive] = np.exp(1j * uniform(0, 2 * np.pi, positive))
    if length % 2 == 0:
        coeffs[length // 2] = 1.

    freq_res = sample_rate / length
    first_coeff, last_coeff = _band(min_freq, max_freq, length, sample_rate)
    first_coeff = max(1, first_coeff)
    coeffs[:first_coeff] = 0
    coeffs[last_coeff:] = 0

    factor = 10. ** (prog / 20.)
    freq_i = np.arange(coeffs.shape[0]) * freq_res
    denom = max(min_freq, freq_res)
    freqs = freq_i[first_coeff:last_coeff]
    attenuation_factors = factor ** (np.log2(freqs / denom))
    coeffs[first_coeff:last_coeff] *= attenuation_factors

    # A real signal has X[N - k] = conj(X[k]). There are (N - 1) // 2
    # such pairs whatever the parity, and one expression places them all:
    # the even case additionally has a Nyquist bin, which is set to a real
    # value above and is its own conjugate.
    #
    # The odd branch used to write the same conjugates one position early
    # and leave the last bin at zero, so the spectrum was not Hermitian
    # and the inverse transform came back complex. `.real` below then
    # discarded an imaginary part worth 8.7% of the signal, silently: an
    # odd-length noise was not the noise its spectrum described.
    paired = (length - 1) // 2
    coeffs[length - paired:] = np.conj(coeffs[1:paired + 1][::-1])

    noise_vector = np.fft.ifft(coeffs).real
    return music.core.normalize_mono(noise_vector)


def _band(low, high, length, sample_rate):
    """The coefficients from `low` to `high` Hz, both ends included.

    Returns the first index and one past the last, as a slice takes them.
    A bin within a billionth of an edge counts as on it, so that an edge
    the resolution divides exactly is not lost to rounding in the
    division.
    """
    first = math.ceil(round(low * length / sample_rate, 9))
    last = math.floor(round(high * length / sample_rate, 9)) + 1
    return first, last


def gaussian_noise(mean: float = 1, std: float = 0.5, duration: float = 2,
                   sample_rate: int = 44100) -> NDArray[np.float64]:
    """Synth gaussian noise

    Parameters
    ----------
    mean : int, optional
        The centre of the band the noise occupies, in units of 3000 Hz,
        by default 1 -- so 3000 Hz.
    std : float, optional
        The width of that band, in the same units, by default 0.5 -- so
        1500 Hz wide, running from 2250 to 3750 Hz.

        These name a band rather than a distribution, despite the
        routine's name and theirs. What is Gaussian here is the shape of
        the samples that come out, which is what a sum of many
        random-phase partials tends to; the samples' own mean and
        standard deviation are set by the normalization at the end and
        have nothing to do with these two. A band reaching below 0 Hz is
        read as the positive part of itself.
    duration : int, optional
        How long in seconds will the noise be, by default 2
    sample_rate : int, optional
        The sample rate to use, by default 44100

    Returns
    -------
    array
        An array for the gaussian noise

    Raises
    ------
    ValueError
        If ``std`` is not positive, or ``mean`` and ``std`` name a band
        that holds no frequency at the resolution the length gives,
        between 0 Hz and the Nyquist frequency. A band of no width zeroed
        every coefficient, and normalizing an all-zero spectrum divides
        by its own zero range: the caller got an array of NaN behind a
        warning. A duration too short to hold a sample is not refused --
        it renders nothing, as everything here does. Also if
        ``sample_rate`` is not positive.

    Notes
    -----
    The band includes both of its ends, as :func:`noise`'s does, and
    stops at the Nyquist frequency. It used to be zeroed after the
    spectrum was mirrored, so a band reaching past the Nyquist frequency
    kept some mirrored components: those frequencies came out at twice
    the level of the rest.

    Examples
    --------
    >>> grains = gaussian_noise(mean=1, std=0.5, duration=0.5)
    >>> len(grains)
    22050
    >>> gaussian_noise(duration=0.3).shape       # a fractional duration
    (13230,)
    """

    # int(): the length indexes arrays and sets a sample count, so a
    # fractional duration raised TypeError out of np.random.uniform
    # rather than rendering the half second it was asked for.
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive; got {sample_rate}")
    if std <= 0:
        # With both ends of the band included, a band of no width on a
        # component would hold that one, and render a sine.
        raise ValueError(
            f"std is the width of the band and must be positive; got {std}. "
            "A band of no width holds no frequency to draw a noise from")
    length = int(duration * sample_rate)
    if length < 1:
        # As in `noise` above: a zero duration renders nothing, and
        # normalize_mono is where an empty sound is refused.
        return np.array([])
    freq_res = sample_rate / float(length)
    coeffs = np.exp(1j * np.random.uniform(0, 2 * np.pi, length))
    f1 = (mean - std / 2) * 3000
    f2 = (mean + std / 2) * 3000
    # A band reaching below 0 Hz is the positive part of itself: mean=1
    # with std=3 asks for -1500 to 7500 Hz, and the sensible reading is 0
    # to 7500. Without the clamp the negative index reached
    # np.zeros(-1500), which is "negative dimensions are not allowed" --
    # numpy giving up on a line that was trying to do something else. A
    # band lying entirely below zero has nothing left after the clamp and
    # is refused below.
    # As in `noise` above: (N - 1) // 2 conjugate pairs whatever the
    # parity, the last of them the highest component below the Nyquist
    # frequency, where the band stops. Slicing at length // 2 assumed an
    # even length, so an odd one -- half a second at 22,050 Hz, say --
    # raised "could not broadcast input array from shape (5511,) into
    # shape (5512,)".
    paired = (length - 1) // 2
    first_coeff, last_coeff = _band(f1, f2, length, sample_rate)
    # Never the constant, and never the Nyquist component of an even
    # length, which has no partner to take a random phase with.
    first_coeff = max(1, first_coeff)
    last_coeff = min(last_coeff, paired + 1)
    if last_coeff <= first_coeff:
        # Every coefficient is about to be zeroed, and normalizing an
        # all-zero spectrum divides by its own zero range: the caller got
        # an array of NaN behind a warning. A band of no width holds no
        # frequency to draw from.
        raise ValueError(
            f"mean={mean} and std={std} give a band from {f1:.1f} Hz to "
            f"{f2:.1f} Hz, which holds no frequency at a resolution of "
            f"{freq_res:.1f} Hz. Widen std, or lengthen the noise")
    coeffs[:first_coeff] = 0
    coeffs[last_coeff:] = 0
    coeffs[length - paired:] = np.conj(coeffs[1:paired + 1][::-1])

    # The spectrum is Hermitian, so the transform is real to rounding.
    # It used to be scaled onto [-1, 1] here as well, which the
    # normalization undoes: it takes out the mean and divides by the peak.
    noise_vector = np.real(np.fft.ifft(coeffs))
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
        An array with no sound. A duration too short to hold a sample,
        negative ones included, gives an empty one, as :func:`noise` and
        :func:`note` do; a negative one used to raise numpy's "negative
        dimensions are not allowed".

    Examples
    --------
    >>> gap = silence(duration=0.25)
    >>> len(gap), float(abs(gap).max())
    (11025, 0.0)
    >>> phrase = horizontal_stack(note(220, 0.2), silence(0.1), note(330, 0.2))
    """

    return np.zeros(max(0, int(duration * sample_rate)))
