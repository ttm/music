import numpy as np
from ..synths import noise


def reverb(duration=1.9, first_phase_duration=0.15, decay=-50,
           noise_type="brown", sonic_vector=0, sample_rate=44100):
    """
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
        A string or scalar specifying the noise. Passed to
        noises(ntype=scalar).
    sonic_vector : array_like
        An optional one dimensional array for the reverberation to be applied.
    sample_rate : scalar
        The sampling frequency.

    Returns
    -------
    result : numpy.ndarray
        An array with the impulse response of the reverberation. If
        sonic_vector is specified, the reverberation applied to sonic_vector.

    Notes
    -----
    This is a simple artificial reverberation with a progressive loudness
    decay of the reincidences of the sound and with two periods: the first
    consists of scattered reincidences, the second period reincidences is
    modeled by a noise.

    Comparing with the description in [1], the frequency bands are ignored.

    One might want to run this function twice to obtain a stereo reverberation.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    lambda_r = int(duration * sample_rate)
    lambda1 = int(first_phase_duration * sample_rate)
    # Sound reincidence probability in the first period:
    ii = np.arange(lambda_r)
    p = (ii[:lambda1] / lambda1) ** 2.
    # incidences:
    r1_ = np.random.random(lambda1) < p
    a = 10. ** ((decay / 20) * (ii / (lambda_r - 1)))
    # Eq. 76 First period of reverberation:
    r1 = r1_ * a[:lambda1]  # first incidences

    # Eq. 77 Second period of reverberation:
    noise_ = noise(noise_type, max_freq=sample_rate / 2,
                   number_of_samples=lambda_r - lambda1)
    r2 = noise_ * a[lambda1:lambda_r]

    # Eq. 78 Impulse response of the reverberation
    result = np.hstack((r1, r2))
    result[0] = 1.
    if type(sonic_vector) in (np.ndarray, list):
        return np.convolve(sonic_vector, result)
    else:
        return result
