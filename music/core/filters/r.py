import numpy as n
from music.utils import convolve
from music.core.synths import noises

def R(d=1.9, d1=0.15, decay=-50, stat="brown", sonic_vector=0, fs=44100):
    """
    Apply an artificial reverberation or return the impulse response.

    Parameters
    ----------
    d : scalar
        The total duration of the reverberation in seconds.
    d1 : scalar
        The duration of the first phase of the reverberation
        in seconds.
    decay : scalar
        The total decay of the last incidence in decibels.
    stat : string or scalar
        A string or scalar specifying the noise.
        Passed to noises(ntype=scalar).
    sonic_vector : array_like
        An optional one dimensional array for the reverberation to
        be applied.
    fs : scalar
        The sampling frequency.

    Returns
    -------
    s : numpy.ndarray
        An array if the impulse response of the reverberation
        (if sonic_vector is not specified),
        or with the reverberation applied to sonic_vector.

    Notes
    -----
    This is a simple artificial reverberation with a progressive
    loudness decay of the reincidences of the sound and with
    two periods: the first consists of scattered reincidences,
    the second period reincidences is modeled by a noise.

    Comparing with the description in [1], the frequency bands
    are ignored.

    One might want to run this function twice to obtain
    a stereo reverberation.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound."
    arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    Lambda = int(d*fs)
    Lambda1 =  int(d1*fs)
    # Sound reincidence probability probability in the first period:
    ii = n.arange(Lambda)
    P = (ii[:Lambda1]/Lambda1)**2.
    # incidences:
    R1_ = n.random.random(Lambda1) < P
    A = 10.**( (decay1/20)*(ii/(Lambda-1)) )
    ### Eq. 76 First period of reverberation:
    R1 = R1_*A[:Lambda1]  # first incidences

    ### Eq. 77 Second period of reverberation:
    noise = noises(ntype, fmax=fs/2, nsamples=Lambda-Lambda1)
    R2 = noise*A[Lambda1:Lambda]
    ### Eq. 78 Impulse response of the reverberation
    R = n.hstack((R1,R2))
    R[0] = 1.
    if type(sonic_vector) in (n.ndarray, list):
        return convolve(sonic_vector, R)
    else:
        return R
