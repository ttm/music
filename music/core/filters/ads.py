import numpy as n
from .ad import AD

def ADS(d=2, A=20, D=20, S=-5, R=50, trans="exp", alpha=1,
        dB=-80, to_zero=1, nsamples=0, sonic_vector=0, fs=44100):
    """
    A shorthand to make an ADSR envelope for a stereo sound.

    See ADSR() for more information.

    """
    if type(sonic_vector) in (n.ndarray, list):
        sonic_vector1 = sonic_vector[0]
        sonic_vector2 = sonic_vector[1]
    else:
        sonic_vector1 = 0
        sonic_vector2 = 0
    s1 = AD(d=d, A=A, D=D, S=S, R=R, trans=trans, alpha=alpha,
        dB=dB, to_zero=to_zero, nsamples=nsamples, sonic_vector=sonic_vector1, fs=fs)
    s2 = AD(d=d, A=A, D=D, S=S, R=R, trans=trans, alpha=alpha,
        dB=dB, to_zero=to_zero, nsamples=nsamples, sonic_vector=sonic_vector2, fs=fs)
    s = n.vstack(( s1, s2 ))
    return s
