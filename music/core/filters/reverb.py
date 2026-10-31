"""Simple reverberation filters and impulse response generation."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from ..synths.noises import _random, noise
from ...utils import as_sonic_vector
from ..._rate import resolve_rate, sample_count


def reverb(duration: float = 1.9, first_phase_duration: float = 0.15,
           decay: float = -50, noise_type: str | float = "brown",
           sonic_vector: ArrayLike = 0,
           sample_rate: int | None = None,
           seed: int | None = None) -> NDArray[np.float64]:
    r"""
    Apply an artificial reverberation or return the impulse response.

    Parameters
    ----------
    duration : scalar
        The total duration of the reverberation in seconds.
    first_phase_duration : scalar
        The duration of the first phase of the reverberation in seconds.
    decay : scalar
        The total decay of the last incidence in decibels.
    noise_type : string or scalar
        The colour of the noise the second period is made of, passed to
        :func:`noise` as its ``noise_type``.
    sonic_vector : array_like
        An optional one dimensional array for the reverberation to be applied.
    sample_rate : scalar or None
        The sampling frequency.
        None, the default, is the rate :func:`music.get_sample_rate` returns.
    seed : integer or None
        A seed for the incidences and the noise, as :func:`noise` takes
        one: the same seed draws the same room again, and leaves NumPy's
        global random state alone. None, the default, draws from that
        state, as this always did.

    Returns
    -------
    result : numpy.ndarray
        An array with the impulse response of the reverberation. If
        sonic_vector is specified, the reverberation applied to sonic_vector.

    Raises
    ------
    ValueError
        If ``first_phase_duration`` is negative or longer than ``duration``,
        ``sample_rate`` is not positive, or ``duration`` is shorter than one
        sample: the response begins with the direct sound, so it has at
        least that.

        If ``sonic_vector`` is not one-dimensional: numpy said "object too
        deep for desired array". Reverberate each channel, or mix them
        down first.

        If ``seed`` is not a whole number in [0, 2**64).

    Notes
    -----
    The impulse response has the article's two periods [1]. Over the
    first, of :math:`\Lambda_1` samples, the sound comes back at sample
    :math:`i` with the probability :math:`p_i = (i / \Lambda_1)^2`
    (``eq:p1rev``); over the rest of the :math:`\Lambda_R`, it comes back
    as `noise_type` noise :math:`r^b_i` (``eq:p2rev``). Both decay by
    `decay` decibels, :math:`V_{dB}`, across the whole response:

    .. math::

        r_i = 10^{\frac{V_{dB}}{20} \frac{i}{\Lambda_R - 1}} \times
        \begin{cases}
            1 \text{ with probability } p_i, \text{ else } 0,
            & i < \Lambda_1, \\
            r^b_i, & \Lambda_1 \le i < \Lambda_R,
        \end{cases}

    with :math:`r_0 = 1`, the direct sound (``eq:rev``). Given a
    `sonic_vector`, it returns the sound convolved with :math:`r`, which
    is longer than the sound by :math:`\Lambda_R - 1` samples. Unlike
    the article's, the reflections are not filtered by the surfaces they
    meet, so every frequency decays alike. For a stereo room, reverberate
    each channel with a different `seed`.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)


    Examples
    --------
    >>> impulse = reverb(duration=1.9, first_phase_duration=0.15, decay=-50)
    >>> round(float(impulse[0]), 3)     # the direct sound
    1.0
    >>> wet = reverb(sonic_vector=note(440, 1), duration=0.5)
    >>> len(wet) > 44100                # longer than the note by its tail
    True
    """
    sample_rate = resolve_rate(sample_rate)
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive; got {sample_rate}")
    draw = _random(seed)
    samples = as_sonic_vector(sonic_vector)
    if samples is not None and samples.ndim != 1:
        raise ValueError(
            f"reverb reverberates a mono sound; got an array of shape "
            f"{samples.shape}. Reverberate each channel, or mix them down "
            f"first")
    if first_phase_duration < 0:
        raise ValueError(
            "first_phase_duration must be non-negative; "
            f"got {first_phase_duration}")
    lambda_r = sample_count(duration, sample_rate)
    lambda1 = sample_count(first_phase_duration, sample_rate)
    if lambda1 > lambda_r:
        # ii[:lambda1] is only lambda_r long, so p and the incidences
        # below came out different lengths and numpy reported a
        # broadcast failure naming two sample counts, which says
        # nothing about the two durations that caused it.
        raise ValueError(
            f"first_phase_duration ({first_phase_duration} s) cannot "
            f"exceed duration ({duration} s): the first period of the "
            "reverberation is part of it, not longer than it"
        )
    if lambda_r < 1:
        raise ValueError(
            f"duration ({duration} s) must span at least one sample at "
            f"{sample_rate} Hz: the response begins with the direct sound")
    # Sound reincidence probability in the first period:
    ii = np.arange(lambda_r)
    p = (ii[:lambda1] / lambda1) ** 2.
    # incidences:
    r1_ = draw.random(lambda1) < p
    # A one-sample response is the direct sound alone, with no span for
    # the decay to divide.
    a = 10. ** ((decay / 20) * (ii / max(lambda_r - 1, 1)))
    # Eq. 76 First period of reverberation:
    r1 = r1_ * a[:lambda1]  # first incidences

    # Eq. 77 Second period of reverberation:
    # The noise needs the rate as well as the band: given only
    # max_freq=sample_rate / 2, it built that band at 44.1 kHz, so at
    # 8 kHz the tail held 95% of its energy below 680 Hz.
    # A response that is all first period has an empty second one, which
    # `noise` renders as no samples, drawing nothing from the random
    # state. While it read zero samples as "not given" it returned two
    # seconds there, and the multiplication below failed to broadcast.
    # The band's floor is noise's own 15 Hz, held at the Nyquist frequency
    # for a rate too low to reach it: noise refuses a band upside down,
    # and at 20 Hz this one would be, for a floor the caller never set.
    tail = lambda_r - lambda1
    # A seeded room draws the noise's seed from its own generator, so the
    # two draws do not repeat each other's numbers.
    noise_ = noise(noise_type, min_freq=min(15, sample_rate / 2),
                   max_freq=sample_rate / 2, number_of_samples=tail,
                   sample_rate=sample_rate,
                   seed=None if seed is None else int(draw.integers(2**63)))
    r2 = noise_ * a[lambda1:lambda_r]

    # Eq. 78 Impulse response of the reverberation
    result = np.hstack((r1, r2))
    result[0] = 1.
    if samples is not None:
        return _convolved(samples, result)
    return result


def _convolved(sound, response):
    """`sound` convolved with `response`, through their spectra.

    The response lasts as long as the reverberation, 83,790 samples at
    the defaults, and ``np.convolve`` is always direct: it multiplied
    every sample of the sound by every sample of the response, so a
    two-second sound took a second and a half to reverberate, and a
    three-minute piece more than two minutes. Multiplying the spectra
    takes a hundredth of that second, and differs from the direct sum
    only in rounding, measured below one part in 1e14 of the peak. An
    empty sound has nothing to reverberate and gives nothing back, which
    ``np.convolve`` refused in its own words.
    """
    if sound.size == 0:
        return np.zeros(0)
    length = sound.size + response.size - 1
    size = 1 << (length - 1).bit_length()
    spectrum = np.fft.rfft(sound, size) * np.fft.rfft(response, size)
    return np.fft.irfft(spectrum, size)[:length]
