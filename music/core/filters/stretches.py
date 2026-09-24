"""Time-stretching utilities for manipulating audio segments."""

import numpy as np
from music.utils import horizontal_stack


def stretches(x, durations=(1, 4, 8, 12), sample_rate=44100):
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
    sample_rate : integer
        The sample rate in Hertz, used to read the length of ``x`` as a
        duration and so to work out how much each repeat is squeezed.

    Returns
    -------
    ndarray
        The repeats concatenated, mono or ``(2, nsamples)`` stereo
        according to ``x``, each ``int(duration * sample_rate)`` samples
        long. An empty ``x`` gives an empty result.

    Raises
    ------
    ValueError
        If any entry of ``durations`` is zero or negative. A duration of
        zero squeezes the fragment into no samples and a negative one
        reverses the step, so neither produces the repeat the caller
        asked for.

    Examples
    --------
    >>> asound = horizontal_stack(*[note_with_vibrato(freq=i, vibrato_freq=j)
    ...                           for i, j in zip([220,440,330,440,330],
    ...                                           [.5,15,6,5,30])])
    >>> s = stretches(asound)
    >>> s = stretches(asound,
    ...               durations=[.2, .3] * 10 + [.1, .2, .3, .4] * 8 +
    ...               [.5, 1.5, .5, 1., 5., .5, .25, .25, .5, 1., .5] * 2)
    >>> write_wav_mono(s, 'stretches.wav')

    Notes
    -----
    This function is useful to render musical sequences given any material.
    PS: not clear if this function is already useful.

    """
    x = np.array(x)
    if any(duration <= 0 for duration in durations):
        raise ValueError("every duration in durations must be positive")

    if len(x.shape) == 1:
        length = x.shape[0]
        stereo = False
    else:
        length = x.shape[1]
        stereo = True
    if length == 0:
        # Nothing to repeat; the step through it divided by its length.
        return x
    sound = []
    for ss in durations:
        # Whole samples, as every duration here is counted, each read from
        # the nearest sample of the fragment. Positions that rounded to
        # one past its end were dropped rather than held, so a repeat fell
        # short by half a fragment sample's worth of output: a ten-sample
        # fragment stretched to twelve seconds lost more than half of one.
        count = int(ss * sample_rate)
        indexes = np.minimum(np.round(np.arange(count) * length / count),
                             length - 1).astype(np.int64)
        if stereo:
            segment = x[:, indexes]
        else:
            segment = x[indexes]
        sound.append(segment)
    sound_ = horizontal_stack(*sound)
    return sound_
