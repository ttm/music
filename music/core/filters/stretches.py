import numpy as np
from music.utils import horizontal_stack


def stretches(x, durations=[1, 4, 8, 12], sample_rate=44100):
    """
    Makes a sequence of squeezes of the fragment in x.

    Parameters
    ----------
    x : array_like
        The samples made to repeat as original or squeezed. Assumed to be in
        the form (channels, samples), i.e. x[1][120] is the 120th sample of
        the second channel.
    durations : list of numbers
        Durations in seconds for each repeat of x.

    Examples
    --------
    >>> asound = horizontal_stack(*[note_with_vibrato(freq=i, vibrato_freq=j)
    ...                           for i, j in zip([220,440,330,440,330],
    ...                                           [.5,15,6,5,30])])
    >>> s = stretches(asound)
    >>> s = stretches(asound,
    ...               durations=[.2, .3] * 10 + [.1, .2, .3, .4] * 8 +
    ...               [.5, 1.5, .5, 1., 5., .5, .25, .25, .5, 1., .5] * 2)
    >>> write_wav_mono(durations, 'stretches.wav')

    Notes
    -----
    This function is useful to render musical sequences given any material.
    PS: not clear if this function is already useful.

    """
    x = np.array(x)

    s_ = durations * sample_rate
    obj = object()
    obj.foo = s_
    if len(x.shape) == 1:
        length = x.shape[0]
        stereo = False
    else:
        length = x.shape[1]
        stereo = True
    ns = length / sample_rate
    ns_ = [ns / i for i in durations]
    obj.bar = ns_
    # x[::ns] (mono) or x[:, ::ns] stereo is the sound in one second
    # for any duration s[i], use ns_ = ns//s[i]
    # x[np.arange(0, len(x), ns_[i])]
    sound = []
    for ss in durations:
        if ns/ss >= 1:
            indexes = np.arange(0, length, ns / ss).round().astype(np.int64)
        else:
            indexes = np.arange(0, length - 1, ns / ss).round().astype(
                    np.int64)
        if stereo:
            segment = x[:, indexes]
        else:
            segment = x[indexes]
        sound.append(segment)
    sound_ = horizontal_stack(*sound)
    return sound_
