import numpy as n
# from . import J
from .f import F

def CF(s1, s2, dur=500, method='lin', fs=44100):
    """
    Cross fade in dur milisseconds.

    """
    ns = int(dur*fs/1000)
    if len(s1.shape) != len(s2.shape):
        print('enter s1 and s2 with the same shape')
    if len(s1.shape) == 2:
        s1_ = CF(s1[0], s2[0], dur, method, fs)
        s2_ = CF(s1[1], s2[1], dur, method, fs)
        s = n.array( (s1_, s2_) )
        return s
    s1[-ns:] *= F(nsamples=ns, method=method, fs=fs)
    s2[:ns] *= F(nsamples=ns, method=method, fs=fs, out=False)
    s = J(s1, s2, d = -dur/1000)
    return s
