"""Time-stretching utilities for manipulating audio segments."""

import numpy as np
from music.utils import horizontal_stack


def stretches(x, durations=(1, 4, 8, 12), sample_rate=44100):
    """
    Makes a sequence of squeezes of the fragment in x.

    Parameters
    ----------
    x : array_like
        The mono samples to repeat, or two-channel stereo samples in the
        form ``(2, samples)``; for stereo, ``x[1][120]`` is the 120th sample
        of the second channel.
    durations : iterable of numbers
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
        asked for. Also if ``sample_rate`` is not positive, or ``x`` is not
        mono or two-channel stereo.

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
    if x.ndim not in (1, 2) or (x.ndim == 2 and x.shape[0] != 2):
        raise ValueError(
            "stretches accepts mono or stereo audio; expected a one-"
            f"dimensional array or shape (2, samples), got {x.shape}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive; got {sample_rate}")
    durations = tuple(durations)
    if any(duration <= 0 for duration in durations):
        raise ValueError("every duration in durations must be positive")

    if x.ndim == 1:
        length = x.shape[0]
        stereo = False
    else:
        length = x.shape[1]
        stereo = True
    if not durations:
        return x[:, :0] if stereo else x[:0]
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
