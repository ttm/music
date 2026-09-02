"""Utility functions shared across the package."""

import logging
import warnings
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

LAMBDA_TILDE = 1024 * 16

#: The waveforms :func:`waveform_table` knows how to build.
WAVEFORMS = ("sine", "sawtooth", "square", "triangle")


def waveform_table(kind: str,
                   size: int = LAMBDA_TILDE) -> NDArray[np.float64]:
    """One period of a primary waveform, as a lookup table.

    Each table is written directly as a function of the phase, so it is
    exact at any size and holds exactly `size` samples. Building them by
    halves instead — the way this package used to — costs a sample at odd
    sizes and puts the sawtooth and triangle slightly out of true.

    Parameters
    ----------
    kind : {'sine', 'sawtooth', 'square', 'triangle'}
        Which waveform to build.
    size : int, optional
        The number of samples in the period. Defaults to LAMBDA_TILDE.

    Returns
    -------
    ndarray
        `size` samples spanning one period, in [-1, 1].

    Raises
    ------
    ValueError
        If `kind` is not one of WAVEFORMS, or `size` is not positive.

    Examples
    --------
    >>> waveform_table('square', 4)
    array([-1., -1.,  1.,  1.])
    >>> waveform_table('triangle', 4)
    array([-1.,  0.,  1.,  0.])
    >>> waveform_table('sawtooth', 4)
    array([-1. , -0.5,  0. ,  0.5])
    """
    if size <= 0:
        raise ValueError(f"size must be positive; got {size}")
    phase = np.arange(size) / size
    if kind == "sine":
        return np.sin(2 * np.pi * phase)
    if kind == "sawtooth":
        return 2 * phase - 1
    if kind == "square":
        return np.where(phase < .5, -1.0, 1.0)
    if kind == "triangle":
        return np.where(phase <= .5, -1 + 4 * phase, 3 - 4 * phase)
    raise ValueError(
        f"unknown waveform {kind!r}; expected one of {list(WAVEFORMS)}"
    )


WAVEFORM_SINE = waveform_table("sine")
WAVEFORM_SAWTOOTH = waveform_table("sawtooth")
WAVEFORM_SQUARE = waveform_table("square")
WAVEFORM_TRIANGULAR = waveform_table("triangle")


def as_sonic_vector(sonic_vector: Any) -> NDArray[np.float64] | None:
    """Normalize a ``sonic_vector`` argument to an array, or None.

    Many routines in this package take an optional ``sonic_vector`` and fall
    back to synthesizing a bare envelope when none is given.  They used to
    detect that with ``type(sonic_vector) in (np.ndarray, list)``, which
    silently ignored tuples and ndarray subclasses: the input was discarded
    and the caller got an envelope of the default duration instead of an
    error.

    Parameters
    ----------
    sonic_vector : any
        Samples to use, or a scalar sentinel (historically ``0``) or None
        meaning that no samples were supplied.

    Returns
    -------
    ndarray or None
        The samples as an array, or None when none were supplied.

    Examples
    --------
    >>> as_sonic_vector(0) is None
    True
    >>> as_sonic_vector(None) is None
    True
    >>> as_sonic_vector((0.1, 0.2)).shape
    (2,)
    """
    if sonic_vector is None:
        return None
    array = np.asarray(sonic_vector, dtype=np.float64)
    if array.ndim == 0:
        # The legacy scalar sentinel: no samples were supplied.
        return None
    return array


def horizontal_stack(*arrays: ArrayLike) -> NDArray[np.float64]:
    """Creates a horizontal stack of arrays while preserving bidimensional
       data.

    This function takes multiple arrays as input and stacks them horizontally.
    If any of the input arrays are bidimensional (have two dimensions), the
    function ensures that they are treated as stereo data by duplicating mono
    channels to both left and right channels.

    Parameters
    ----------
    *arrays : array_like
        The arrays to be stacked horizontally. Can be one-dimensional (mono)
        or two-dimensional (stereo).

    Returns
    -------
    ndarray
        A numpy array representing the horizontal stack of input arrays.

    Examples
    --------
    >>> mono_array1 = np.array([0.1, 0.2, 0.3])
    >>> mono_array2 = np.array([0.4, 0.5, 0.6])
    >>> stereo_array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> stacked_array = create_horizontal_stack(mono_array1, stereo_array,
    >>>                                         mono_array2)
    >>> stacked_array.shape
    (2, 9)
    """
    # Initialize a flag to indicate whether stereo data is present
    stereo_present = False

    # Convert input arrays to numpy arrays
    arrays_np: list[np.ndarray] = [np.array(arr) for arr in arrays]

    # Check if any of the input arrays are bidimensional (stereo)
    for arr in arrays_np:
        if len(arr.shape) == 2:
            stereo_present = True
            break

    # If stereo data is present, ensure that mono channels are duplicated to
    # both left and right channels
    if stereo_present:
        for i, arr in enumerate(arrays_np):
            if len(arr.shape) == 1:
                arrays_np[i] = np.array((arr, arr))

    # Return the horizontal stack of arrays
    return np.hstack(arrays_np)


H = horizontal_stack


def db_to_amp(db_difference: float) -> float:
    """Converts a difference in decibels to a difference in amplitude.

    This function takes a difference in decibels as input and returns the
    corresponding difference in amplitude.

    Parameters
    ----------
    db_difference : float
        The difference in decibels to be converted.

    Returns
    -------
    float
        The difference in amplitude corresponding to the input difference
        in decibels.

    Examples
    --------
    >>> db_to_amp(6)
    2.0
    >>> db_to_amp(-6)
    0.5
    """
    return 10. ** (db_difference / 20.)


def amp_to_db(amplitude_difference: float) -> float:
    """Converts a difference in amplitude to a difference in decibels.

    This function takes a difference in amplitude as input and returns the
    corresponding difference in decibels.

    Parameters
    ----------
    amplitude_difference : float
        The difference in amplitude to be converted.

    Returns
    -------
    float
        The difference in decibels corresponding to the input difference
        in amplitude.

    Examples
    --------
    >>> amp_to_db(2.0)
    6.0
    >>> amp_to_db(0.5)
    -6.0
    """
    return 20. * np.log10(amplitude_difference)


def hz_to_midi(hertz_value: float) -> np.float64:
    """Converts a frequency in Hertz to a MIDI note number.

    This function takes a frequency value in Hertz as input and returns
    the corresponding MIDI note number.

    Parameters
    ----------
    hertz_value : float
        The frequency value in Hertz to be converted.

    Returns
    -------
    numpy.float64
        The MIDI note number corresponding to the input frequency value.

    Examples
    --------
    >>> hz_to_midi(440)
    69.0
    >>> hz_to_midi(880)
    81.0
    """
    safe_hz = np.clip(hertz_value, np.finfo(float).eps, None)
    return 69 + 12 * np.log2(safe_hz / 440)


def midi_to_hz(midi_value: float) -> float:
    """Converts a MIDI note number to the corresponding frequency in Hertz.

    This function takes a MIDI note number as input and returns the
    corresponding frequency value in Hertz.

    Parameters
    ----------
    midi_value : float
        The MIDI note number to be converted.

    Returns
    -------
    float
        The frequency value in Hertz corresponding to the input MIDI not
        number.

    Examples
    --------
    >>> midi_to_hz(69)
    440.0
    >>> midi_to_hz(81)
    880.0
    """
    return 440 * 2 ** ((midi_value - 69) / 12.)


def midi_to_hz_interval(midi_interval: float) -> float:
    """Converts a MIDI interval to the corresponding frequency interval in
       Hertz.

    This function takes a MIDI interval (measured in semitones) as input and
    returns the corresponding frequency interval in Hertz.

    Parameters
    ----------
    midi_interval : float
        The MIDI interval (measured in semitones) to be converted.

    Returns
    -------
    float
        The frequency interval in Hertz corresponding to the input MIDI
        interval.

    Examples
    --------
    >>> midi_to_hz_interval(12)
    2.0
    >>> midi_to_hz_interval(-12)
    0.5
    """
    return 2 ** (midi_interval / 12)


def pitch_to_freq(
    start_freq: float = 220.0,
    semitones: tuple[int, ...] = (0, 7, 7, 4, 7, 0),
) -> list[float]:
    """Generates a list of frequencies based on a list of semitones and a
    starting frequency.

    This function calculates a list of frequencies based on the given list of
    semitones and a starting frequency. Each semitone value represents the
    number of semitones above or below the starting frequency.

    Parameters
    ----------
    start_freq : float, optional
        The starting frequency in Hertz, by default 220.
    semitones : list, optional
        The list of semitone offsets relative to the starting frequency,
        by default [0, 7, 7, 4, 7, 0].

    Returns
    -------
    list
        A list of frequencies calculated from the given semitones and starting
        frequency.

    Examples
    --------
    >>> pitch_to_freq()  # Default semitones [0, 7, 7, 4, 7, 0]
    [220.0, 493.8833012561241, 493.8833012561241, 329.62755691286986,
     493.8833012561241, 220.0]
    >>> pitch_to_freq(start_freq=440, semitones=[0, 12, 12, 12])
    [440.0, 880.0, 880.0, 880.0]
    """
    return [start_freq * 2 ** (i / 12) for i in semitones]


def mix(first_sonic_vector: np.ndarray,
        second_sonic_vector: np.ndarray) -> np.ndarray:
    """Mixes two sonic vectors.

    This function mixes two sonic vectors of different lengths. It creates a
    new sonic vector by summing the samples of the input sonic vectors. If one
    of the input sonic vectors is shorter than the other, it is padded with
    zeros to match the length of the longer sonic vector before mixing.

    Parameters
    ----------
    first_sonic_vector : ndarray
        The first sonic vector.
    second_sonic_vector : ndarray
        The second sonic vector.

    Returns
    -------
    ndarray
        A mixed sonic vector containing the sum of the input sonic vectors.

    See Also
    --------
    mix_many : the same sum over a list of sounds of any lengths, with
               per-sound offsets and a choice of aligning their starts
               or their ends.
    """
    l1 = len(first_sonic_vector)
    l2 = len(second_sonic_vector)
    if l1 < l2:
        sound = np.zeros(l2)
        sound += second_sonic_vector
        sound[:l1] += first_sonic_vector
    else:
        sound = np.zeros(l1)
        sound += first_sonic_vector
        sound[:l2] += second_sonic_vector
    return sound


def mix_stereo(
    first_sonic_vector: np.ndarray,
    second_sonic_vector: np.ndarray | None = None,
    end: bool = False,
) -> np.ndarray:
    """Mixes two stereo sonic vectors.

    This function mixes two stereo sonic vectors. If only one sonic vector is
    provided, it is duplicated to create a stereo mix. Optionally, the shorter
    sonic vector can be padded with zeros to match the length of the longer
    sonic vector before mixing.

    Parameters
    ----------
    first_sonic_vector : ndarray
        The first stereo sonic vector to mix.
    second_sonic_vector : ndarray, optional
        The second stereo sonic vector to mix, by default None. If not
        provided, the first sonic vector is duplicated to create a stereo mix.
    end : bool, optional
        A flag indicating whether to append the second sonic vector at the end
        of the first sonic vector (if False) or at the beginning (if True), by
        default False.

    Returns
    -------
    ndarray
        A stereo sonic vector containing the mix of the input sonic vectors.

    Notes
    -----
    If `second_sonic_vector` is not provided, the `end` parameter is ignored.

    """
    # ndim, not len: a two-sample mono vector also has len() == 2, and was
    # taken for a stereo pair whose channels were then indexed as scalars.
    first_sonic_vector = np.asarray(first_sonic_vector, dtype=np.float64)
    if first_sonic_vector.ndim != 2:
        first_sonic_vector = np.array((first_sonic_vector, first_sonic_vector))
    if second_sonic_vector is None:
        second_sonic_vector = first_sonic_vector
    else:
        second_sonic_vector = np.asarray(second_sonic_vector,
                                         dtype=np.float64)
        if second_sonic_vector.ndim != 2:
            second_sonic_vector = np.array((second_sonic_vector,
                                            second_sonic_vector))

    if len(first_sonic_vector[0]) > len(second_sonic_vector[0]):
        if not end:
            l2_ = horizontal_stack(second_sonic_vector,
                                   np.zeros((2, len(first_sonic_vector[0]) -
                                             len(second_sonic_vector[0]))))
        else:
            l2_ = horizontal_stack(np.zeros((2, len(first_sonic_vector[0]) -
                                             len(second_sonic_vector[0]))),
                                   second_sonic_vector)
        l1_ = first_sonic_vector
    else:
        if not end:
            l1_ = horizontal_stack(first_sonic_vector,
                                   np.zeros((2, len(second_sonic_vector[0]) -
                                             len(first_sonic_vector[0]))))
        else:
            l1_ = horizontal_stack(np.zeros((2, len(second_sonic_vector[0]) -
                                             len(first_sonic_vector[0]))),
                                   first_sonic_vector)
        l2_ = second_sonic_vector
    return l1_ + l2_


def resolve_stereo(afunction, argdict, stereo_vars=('sonic_vector',)):
    """Resolve stereo arguments for a function.

    Parameters
    ----------
    afunction : function
        The function to apply the resolved arguments to.
    argdict : dict
        The dictionary of arguments to resolve.
    stereo_vars : list, optional
        List of variable names that represent stereo data, by default
        ['sonic_vector']

    Returns
    -------
    numpy.ndarray
        Stereo output of the function.
    """
    ag1 = argdict.copy()
    ag2 = argdict.copy()
    for v in stereo_vars:
        argdict[v] = convert_to_stereo(argdict[v])
        sv1 = argdict[v][0]
        sv2 = argdict[v][1]
        ag1[v] = sv1
        ag2[v] = sv2

    sv1_ = afunction(**ag1)
    sv2_ = afunction(**ag2)
    s = np.array((sv1_, sv2_))
    return s


def convert_to_stereo(sound_vector: ArrayLike) -> NDArray[np.float64]:
    """Converts a sound vector to stereo format.

    Converts a mono or multi-channel sound vector into stereo format. If the
    input vector is mono, it duplicates the signal to both left and right
    channels. If the input vector has more than two channels, it keeps only
    the first two channels (left and right) and sums the rest to both left and
    right channels.

    Parameters
    ----------
    sound_vector : array_like
        The input sound vector to be converted to stereo format. Can be a
        one-dimensional array (mono) or a two-dimensional array (stereo or
        multi-channel).

    Returns
    -------
    stereo_sound : ndarray
        A two-dimensional numpy array representing the sound vector in stereo
        format. The first row corresponds to the left channel, and the second
        row corresponds to the right channel.

    Examples
    --------
    >>> mono_vector = np.array([0.1, 0.2, 0.3, 0.4])
    >>> stereo_vector = convert_to_stereo(mono_vector)
    >>> stereo_vector.shape
    (2, 4)
    """
    # Convert the input sound vector to a numpy array
    sound_array = np.array(sound_vector)

    # Check the shape of the input array
    if len(sound_array.shape) == 1:
        # If the input vector is mono, duplicate it for both left and right
        # channels.
        stereo_sound = np.array((sound_array, sound_array))
    elif sound_array.shape[0] > 2:
        # If the input vector has more than two channels, keep only the first
        # two (left and right) and sum the rest to both left and right channels
        warnings.warn(
            'Keeping first two channels in left and right. '
            'The rest will be added to both left and right.')
        stereo_sound = np.array((sound_array[0], sound_array[1]))
        for channel in sound_array[2:]:
            stereo_sound += channel
    else:
        # If the input vector is already stereo, return it without any
        # modifications
        stereo_sound = sound_array

    return stereo_sound


def _integrate_phase(increments, length, block=16384):
    """Accumulate per-sample table shifts without accumulating drift.

    A wavetable lookup needs the running total of the shift between one
    sample and the next. ``np.cumsum`` gives it, but it accumulates into
    a total that keeps growing, so each addition loses low bits against
    a larger and larger running value. The error grows with the length
    of the render and it grows in one direction -- it is drift, not
    noise. Against the exact phase for a 200 Hz carrier and a 16384
    entry table, it reaches 0.48 table entries at ten minutes and 32 at
    an hour, which is enough to change the entry that gets looked up.

    Two things fix it. The running total is folded into one table period
    as it goes, so it never grows past ``length`` and never loses those
    bits; and it is carried between blocks through ``ndarray.sum``,
    whose pairwise summation is far more accurate than a sequential one.
    Inside a block the accumulation is still ``np.cumsum``, over at most
    ``block`` values, where the error stays near the machine epsilon.

    The result is the phase already folded into ``[0, length)``, which
    truncates to the same index the caller's own ``% length`` would
    have produced.

    Parameters
    ----------
    increments : array_like
        The shift in table entries between one sample and the next.
    length : int
        The number of entries in the table, the period to fold into.
    block : int
        How many samples to accumulate before folding and carrying.

    Returns
    -------
    ndarray
        The accumulated phase at each sample, in ``[0, length)``.

    See Also
    --------
    music.note_with_vibrato : one of the routines that integrates a
                              varying frequency this way.

    """
    increments = np.asarray(increments, dtype=np.float64)
    if increments.size <= block:
        return np.cumsum(increments) % length

    phase = np.empty(increments.shape, dtype=np.float64)
    carry = 0.0
    for start in range(0, increments.size, block):
        chunk = increments[start:start + block]
        phase[start:start + block] = (carry + np.cumsum(chunk)) % length
        carry = (carry + chunk.sum()) % length
    return phase


def mix_with_offset(
    first_sonic_vector: ArrayLike,
    second_sonic_vector: ArrayLike,
    duration: float = 0,
    number_of_samples: int = 0,
    sample_rate: int = 44100,
) -> NDArray[np.float64]:
    """Mix two sonic vector by placing the beginning of the second one
    a specified number of seconds after the first one.

    Parameters
    ----------
    first_sonic_vector : numeric array
        A sequence of PCM samples.
    second_sonic_vector : numeric array
        Another sequence of PCM samples.
    duration : numeric
        The offset of the second sound in seconds: how far after the
        start of the first the second begins. Negative starts the second
        sound that many seconds before the first one ends.
    number_of_samples : int
        The offset in samples, taken instead of ``duration`` when given.
    sample_rate : int
        The sample rate in Hertz.

    Returns
    -------
    ndarray
        The two sounds summed at that offset, long enough to hold both.

    Notes
    -----
    A negative ``duration`` must satisfy
    ``-duration * sample_rate < first_sonic_vector.shape[-1]``: the second
    sound cannot begin before the first one does.

    The description of ``duration`` above was truncated mid-sentence, and
    its last line sat at column zero, so numpydoc read "s1 ends." as a
    parameter of this function and rendered it as one.

    See Also
    --------
    mix_many_with_offsets : the same, for any number of sounds, each
                            offset from the mix built so far.
    mix_many : a list of sounds aligned at their starts or their ends.

    """
    first_sonic_vector = np.array(first_sonic_vector)
    second_sonic_vector = np.array(second_sonic_vector)
    if 2 in [len(first_sonic_vector.shape), len(second_sonic_vector.shape)]:
        # The parameters were renamed from s1/s2 at some point but these
        # names were not, so the stereo path raised KeyError.
        return resolve_stereo(mix_with_offset, locals(),
                              ['first_sonic_vector', 'second_sonic_vector'])
    dur = duration

    if not number_of_samples:  # sample in s1 where s2[0] is added
        ns = dur * sample_rate
    else:
        ns = number_of_samples

    if ns >= 0:
        nst = ns + len(second_sonic_vector)
    else:
        nst = len(first_sonic_vector) + len(second_sonic_vector) + ns

    if nst < len(first_sonic_vector):
        nst = len(first_sonic_vector)

    s = np.zeros(int(nst))
    s[:len(first_sonic_vector)] += first_sonic_vector
    logging.debug(
        's.shape %s s1.shape %s s2.shape %s ns %s nst %s',
        s.shape,
        first_sonic_vector.shape,
        second_sonic_vector.shape,
        ns,
        nst)
    ns_int = int(ns)
    if ns >= 0:
        s[ns_int: ns_int + len(second_sonic_vector)] += second_sonic_vector
        # s[-len(s2):] += s2
    else:
        start = int(len(first_sonic_vector) + ns)
        end = int(len(first_sonic_vector) + ns + len(second_sonic_vector))
        s[start:end] += second_sonic_vector
    return s


def mix_many_with_offsets(*args: ArrayLike) -> NDArray[np.float64]:
    """Mix any number of sonic vectors, each at its own offset.

    Where :func:`mix_with_offset` takes two sounds and one offset, this
    takes as many as it is given: each sound is mixed into the result
    built so far, delayed by the offset that follows it.

    Parameters
    ----------
    *args : sonic vectors, each optionally followed by a scalar
        A sequence of sonic vectors, each a sequence of PCM samples, or
        a sequence alternating the sonic vectors and their offsets in
        seconds. A vector with no scalar after it is mixed at offset 0.

    Returns
    -------
    ndarray
        Every sound summed at its offset. An empty argument list gives
        an empty array.

    Raises
    ------
    ValueError
        If a positional argument that should be a sonic vector is a bare
        number. Two consecutive offsets almost always means a sound was
        left out, and mixing silence at a wrong offset would hide it.

    See Also
    --------
    mix_with_offset : two sounds and a single offset.
    mix_many : a list of sounds aligned at their starts or their ends.

    """
    i = 0
    s: NDArray[np.float64] = np.array([])
    while i < len(args):
        a = args[i]  # new array
        # np.ndim rather than an exact type check, which rejected tuples and
        # ndarray subclasses although the parameters are array_like.
        if np.ndim(a) == 0:
            raise ValueError(
                "expected a sequence of numbers at position "
                f"{i}, got {a!r}")
        if len(args) > i + 1:
            off: Any = args[i + 1]
            if np.isscalar(off):
                offset = float(cast(float, off))
                i += 2
            else:
                offset = 0.0
                i += 1
        else:
            offset = 0.0
            i += 1
        s = mix_with_offset(s, a, duration=offset)
    return s


#: The name this had before it was renamed for saying, in the name, how it
#: differs from ``mix_with_offset``. Kept bound so existing callers work.
mix_with_offset_ = mix_many_with_offsets


def pan_transitions(p=((1, 1), (1, 0), (0, 1), (1, 1)), d=(2, 2, 2),
                    method=('lin', 'circ', 'exp'), sample_rate=44100,
                    sonic_vector=None):
    """Applies pan transitions to a sonic vector.

    Parameters
    ----------
    p : list of tuples, optional
        List of pan positions, where each tuple represents the amplitude
        envelope of each channel, by default [(1,1),(1,0),(0,1),(1,1)]
    d : list, optional
        List of durations for each transition, by default [2,2,2]
    method : list, optional
        List of pan transition methods, by default ['lin','circ','exp']
    sample_rate : int, optional
        Sample rate of the audio, by default 44100
    sonic_vector : ndarray, optional
        Input sonic vector, by default None

    Returns
    -------
    ndarray
        Stereo audio signal with pan transitions applied.

    Notes
    -----
    Each pan transition i starts and ends amplitude envelope
    of channel c in p[i][c] and p[i+1][c].

    Consider only one of such fades
    to understand the pan transition methods::

        'lin' fades linearly in and out:
            x*k_i+y*(1-k_i)
            or
            s1_i*x_i +s2_i*(1-x_i) = (s1-s2)*x_i + s_2
        'circ' keeps amplitude one using
            cos(x)**2 + sin(y)**2 = 1
        'exp' makes the cross_fade using exponentials.

    'exp' entails linear loudness variation for each channel,
    but total loudness is not preserved because
    final amplitude's ambit is not preserved.
    'lin' and 'circ', on the other hand, preserve total loudness
    but does not provide a linear variation of loudness for
    each sound on the cross-fade.

    For now, each channel's signal are kept from mixing.
    One immediate possibility is to maintain the expected
    tessiture of the sample amplitudes.
    Say p = [.5,1,0,.5] ~ [(1,1),(1,0),(0,1),(1,1)].
    Then pi,pj = .5,1 might be performed as::

        s1 = s1*.5 -> 0
        s2 = s1*.5 -> (s1+s2)*.5

    Or through sinusoids and expotentials

    Make fast and slow fades and parameter transitions
    using weber-fechner and steven's laws.
    E.g.::

        pitch_trans = [pitch0*X**(i/Y) for i in range(12)]
        pitch_trans = [pitch0 + X*i**Y for i in range(12)]

    Examples
    --------
    >>> p = [(0, 1), (1, 0), (0, 1), (1, 1)]
    >>> d = [2, 2, 2]
    >>> method = ['lin', 'circ', 'exp']
    >>> sonic_vector = np.random.rand(2, 44100 * 6)  # Random stereo signal
    >>> result = pan_transitions(p, d, method, sonic_vector=sonic_vector)
    """

    pp_ = p[0]
    t0_ = []
    t1_ = []
    for i, pp in enumerate(p[1:]):
        # t0 = pp[0] - pp_[0]
        # t1 = pp[1] - pp_[1]
        di = d[i] * sample_rate
        di_ = np.arange(di) / di
        t0 = pp_[0] * (1 - di_) + pp[0] * di_
        t1 = pp_[1] * (1 - di_) + pp[1] * di_
        t0_.append(t0)
        t1_.append(t1)
    t0__ = horizontal_stack(*t0_)
    t1__ = horizontal_stack(*t1_)
    t = np.array((t0__, t1__))
    if sonic_vector is not None:
        sonic_vector = convert_to_stereo(sonic_vector)
        return mix_with_offset(sonic_vector, t)
    return t


def mix_many(sonic_vectors, end=False, offset=0, sample_rate=44100):
    """Mix sonic vectors of arbitrary lengths.

    The operation consists in summing sample by sample [1].
    This function helps when the sonic_vectors are not
    of the same size.

    Parameters
    ----------
    sonic_vectors : list of sonic_arrays
        The sonic vectors to be summed.
    end : boolean
        If True, sync the final samples.
        If False (default) sync the initial samples.
    offset : list of scalars
        A list of the offsets for each sonic vectors
        in seconds.
    sample_rate : integer
        The sample rate. Only used if offset is supplied.

    Returns
    -------
    S : ndarray
        A numpy array where each value is a PCM sample of
        the resulting sound.

    Examples
    --------
    >>> W(mix_many(sonic_vectors=[np.vstack(), N()]))  # writes a WAV
                                                   # with nodes

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    sonic_vectors = [np.asarray(s) for s in sonic_vectors]

    if offset:
        for i, off in enumerate(offset):
            if off:
                pad = np.zeros(int(off * sample_rate))
                sonic_vectors[i] = np.hstack((pad, sonic_vectors[i]))

    max_len = max(len(s) for s in sonic_vectors)
    aligned = []
    for s in sonic_vectors:
        pad_len = max_len - len(s)
        if pad_len:
            pad = np.zeros(pad_len)
            if end:
                s = np.hstack((pad, s))
            else:
                s = np.hstack((s, pad))
        aligned.append(s)

    return np.sum(aligned, axis=0)


#: The name this had while it was the second mixer rather than the general
#: one. Kept bound so existing callers work.
mix2 = mix_many


def profile(adict):
    """
    Summarize a namespace of variables. **Not implemented.**

    Parameters
    ----------
    adict : dict
        The namespace to describe, mapping names to values -- typically
        ``locals()`` or ``vars()`` from a piece of synthesis code.

    Returns
    -------
    dict
        The structure described below, once this is written.

    Raises
    ------
    NotImplementedError
        Always. The body of this function has been commented out since
        it was written, so every call returned ``None`` while the
        docstring described a dictionary. Raising says the same thing
        the silence did, where a caller can hear it; the design below is
        kept because it is the specification, not a description.

    Notes
    -----
    Should return a dictionary with the following structure:
      d['type']['scalar'] should return all the names of scalar variables
      as strings.
      scalar: all names in numeric, string, float, integer,
      collections: all names in dict, list, set, ndarray

      d['analyses']['ndarray'] should return a general analysis of the
      ndarrays, including size in seconds of each considering fs.
      Mean and mean square values to have an idea of what is there.
      RMS values in different scales and the overal RMS standard deviation
      on a scale is helpful in grasping disconttinuities.
      The overal RMS mean of a scale is a hint of whether the variable
      is meant to be used (or usable as) PCM samples or parametrization.
      E.g.

        * Large arrays, i.e. with many elements, are usable as PCM samples.
          If the mean is zero, and they are bound to [-1,1] or to some power
          of 2, specially [-2**15, 2**15-1], it is probably PCM samples
          synthesized or sampled or derivatives.
          If it has more than one or two dimensions where the many samples
          are, it might be a collection of audio samples with the sample size

        * Arrays with an offset (abs(mean) << 0) and small number of elements
          are good candidates for parametrization.
          They might be used for repetition, yielding a clear rhythm.
          They might also be used to derive more ellaborate patterns,
          such as by using the values of more then one arrays,
          and using them simultaneously, often creating patterns
          because of the different sizes of each array.

        * Values in the order of hundreds and thousands are
          candidates for frequency.
          Values within zero and 150 are candidates for decibels,
          and for absolute pitch or pitch interval through MIDI notes
          and semitones count, respectively.
          If the values are integers of very close to them,
          or have many consecutive values deviating less then
          10, it is more likely to be related to pitches.
          If the consecutive values deviate by tens to about a hundred,
          it is kin to decibels notation.

    """
    raise NotImplementedError(
        "profile() was never implemented: its body is the commented-out "
        "sketch below. It is exported and documented, and it does not "
        "work.")
    # for key in adict:
    #     avar = adict[key]
    #     if type(sonic_vector) == np.ndarray:
    #     elif type(sonic_vector) == list:
    #     elif np.isscalar(avar):
    #     else:
    #         print('unrecognized type, implement dealing with it')


def rhythm_to_durations(durations=(4, 2, 2, 4, 1, 1, 1, 1, 2, 2, 4),
                        freqs=None, duration=.25, bpm=None,
                        total_duration=None):
    """Returns durations from rhythmic patterns.

    Parameters
    ----------
    durations : interable of scalars
        The relative durations of each item (e.g. note).
    freqs : iterable of scalars
        The number of the entry's duration that fits into the pulse.
        If supplied, durations is ignored.
    duration : scalar
        A basic duration (e.g. for the pulse) in seconds.
    bpm : scalar
        The number of beats per second.
        If supplied, duration is ignored.
    total_duration: scalar
        The total duration of the sequence in seconds.
        If supplied, both BPM and duration are ignored.

    Returns
    -------
    durs : List of durations in seconds.

    Examples
    --------
    >>> dt = [4, 2, 2, 4, 1,1,1,1, 2, 2, 4]
    >>> durs0 = rhythm_to_durations(dt, duration=.25)
    >>> df = [4, 8, 8, 4, 16, 16, 16, 16, 8, 8, 4]
    >>> durs0_ = rhythm_to_durations(freqs=df, duration=4)
    >>> dtut = [4,2,2, [8, 1,1,1], 4, [4, 1,1,.5,.5], 3,1, 3,1, 4]
    >>> durs1 = rhythm_to_durations(dtut)
    >>> dtuf2 = [4,8,8, [2, 3,3,3], 4, [4, 3,3,6,6], 16/3, 16, 16/3, 16, 4]
    >>> durs1_ = rhythm_to_durations(freqs=dtut2, duration=4)

    Notes
    -----
    The durations parameter is considered to be in a temporal notation
    for durations/rhythm: each entry is a relative duration to
    be multiplied by the base duration given through duration,
    BPM or total_duration::

        durs = [i*duration for i in durations]

    The frequencies parameter is considered to be in a
    frequential notation: each entry is the number of the
    entry that fits a same duration (also given through duration,
    BPM or total_duration)::

        durs = [duration/i for i in freqs]

    The examples above yield (two by two) the same sequences of durations
    by using duration=0.25 when in temporal notation or
    duration=4 when in frequency notation.

    To facilitate the description of rhythms (e.g. for tuplets),
    some set of durations might be an iterable inside durations
    or frequencies. In this case::

        ### if mode is temporal:
            total_dur = cell[0]*duration
            # durations are proportional to remaining values:
            d_ = [i/sum(cell[1:]) for i in cell[1:]]
            durs = [i*total_dur for i in d_]
        ### if mode is frequential:
            total_dur = duration/cell[0]
            # durations are inversely proportional to remaining values:
            d_ = [i/sum(cell[1:]) for i in cell[1:]]
            durs = [i*total_dur for i in d_]

    An example for achieving the same sequence of durations through
    temporal or frequential notation and with cells for tuplets
    is the last two sequences of the examples.

    It might be a good idea to incorporate also this notation::

        d2 = [1, 4, 1, 4]  # [quarter note + 4 sixteenth notes] x 2

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if not bpm and not total_duration:
        dur = duration
    elif bpm:
        dur = bpm / 60
    else:
        dur = None
    durs = []
    if freqs:
        if not dur:  # obtain from total_dur
            durs_ = [1 / i if not isinstance(i, (list, tuple, np.ndarray))
                     else 1 / i[0] for i in freqs]
            dur = total_duration / sum(durs_)
        for d in freqs:
            if isinstance(d, (list, tuple, np.ndarray)):
                t_ = dur / d[0]  # total timespan
                # relative durations from the frequency
                d_ = [1 / i for i in d[1:]]
                # normalize d_ to sum to t_
                d__ = [t_ * i / sum(d_) for i in d_]
                # durs = [t_*i/sum(d[1:]) for i in d[1:]]
                durs.extend(d__)
            else:
                durs.append(dur / d)
    else:
        if not dur:  # obtain from total_dur
            durs_ = [i if not isinstance(i, (list, tuple, np.ndarray))
                     else i[0] for i in durations]
            dur = total_duration / sum(durs_)
        for d in durations:
            if isinstance(d, (list, tuple, np.ndarray)):
                t_ = d[0] * dur  # total timespan
                # relative durations for the potential tuplet
                d_ = [i / sum(d[1:]) for i in d[1:]]
                # normalize d_ to fit t_
                d__ = [i * t_ for i in d_]
                # durs = [t_*i for i in d[1:]]
                durs.extend(d__)
            else:
                durs.append(d * dur)
    return durs
