import numpy as n
from ...utils import H

def sequenceOfStretches(x, s=[1,4,8,12], fs=44100):
    """
    Makes a sequence of squeezes of the fragment in x.

    Parameters
    ----------
    x : array_like
        The samples made to repeat as original or squeezed.
        Assumed to be in the form (channels, samples),
        i.e. x[1][120] is the 120th sample of the second channel.
    s : list of numbers
        Durations in seconds for each repeat of x.

    Examples
    --------
    >>> asound = H(*[V(f=i, fv=j) for i, j in zip([220,440,330,440,330],
                                                  [.5,15,6,5,30])])
    >>> s = sequenceOfStretches(asound)
    >>> s = sequenceOfStretches(asound,s=[.2,.3]*10+[.1,.2,.3,.4]*8+[.5,1.5,.5,1.,5.,.5,.25,.25,.5, 1., .5]*2)
    >>> W(s, 'stretches.wav')
    Notes
    -----
    This function is useful to render musical sequences given any material.

    """
    x = n.array(x)

    s_ = s*fs
    if len(x.shape) == 1:
        l = x.shape[0]
        stereo = False
    else:
        l = x.shape[1]
        stereo = True
    ns = l/fs
    ns_ = [ns/i for i in s]
    # x[::ns] (mono) or x[:, ::ns] stereo is the sound in one second
    # for any duration s[i], use ns_ = ns//s[i]
    # x[n.arange(0, len(x), ns_[i])]
    sound = []
    for ss in s:
        if ns/ss >= 1:
            indexes = n.arange(0, l, ns/ss).round().astype(n.int64)
        else:
            indexes = n.arange(0, l-1, ns/ss).round().astype(n.int64)
        if stereo:
            segment = x[:, indexes]
        else:
            segment = x[indexes]
        sound.append(segment)
    sound_ = H(*sound)
    return sound_
