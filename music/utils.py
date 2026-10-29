"""Utility functions shared across the package."""

import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._deprecation import alias
from ._rate import resolve_rate
from ._warn import warn

LAMBDA_TILDE = 1024 * 16


def cache_root() -> Path:
    """The platform's per-user cache directory.

    Two features here need somewhere to keep a large external resource
    that is not shipped with the package -- the eCantorix engine and the
    KEMAR impulse responses -- and neither belongs inside `site-packages`,
    which an upgrade replaces and a system install may not let you write
    to. This is the one place that decides where such a thing goes.
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home())
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")

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
        No arrays stack to no samples.

    Examples
    --------
    >>> mono_array1 = np.array([0.1, 0.2, 0.3])
    >>> mono_array2 = np.array([0.4, 0.5, 0.6])
    >>> stereo_array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> stacked_array = horizontal_stack(mono_array1, stereo_array,
    ...                                  mono_array2)
    >>> stacked_array.shape
    (2, 9)
    """
    if not arrays:
        # Nothing to stack is nothing, as a sound of no samples is; numpy
        # said it needed at least one array to concatenate.
        return np.zeros(0)

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
    1.9952623149688795
    >>> db_to_amp(-6)
    0.5011872336272722
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
        in amplitude. A ratio of zero is minus infinity decibels.

    Raises
    ------
    ValueError
        If the ratio is negative, which no number of decibels is. It came
        back as NaN behind numpy's warning, and zero as minus infinity
        behind another.

    Examples
    --------
    >>> float(amp_to_db(2.0))
    6.020599913279624
    >>> float(amp_to_db(0.5))
    -6.020599913279624
    >>> float(amp_to_db(0))
    -inf
    """
    if np.any(np.asarray(amplitude_difference) < 0):
        raise ValueError(
            f"amplitude_difference is a ratio of amplitudes, which is never "
            f"negative; got {amplitude_difference}")
    with np.errstate(divide='ignore'):
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
        A frequency of zero or below, which has no pitch, is taken as the
        smallest positive float, a pitch near -680, far below hearing:
        an array of frequencies with silent frames at 0 Hz converts
        without numpy's warning about a logarithm of zero.

    Examples
    --------
    >>> float(hz_to_midi(440))
    69.0
    >>> float(hz_to_midi(880))
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
        The frequency value in Hertz corresponding to the input MIDI note
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
    >>> pitch_to_freq(semitones=(0, 7, 12))  # a fifth, then an octave
    [220.0, 329.6275569128699, 440.0]
    >>> len(pitch_to_freq())  # the default is (0, 7, 7, 4, 7, 0)
    6
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

    Raises
    ------
    ValueError
        If either sound is not one-dimensional. This pads to the longer of
        the two, and ``len()`` of a stereo array counts its channels
        rather than its samples, so a stereo argument reached numpy as a
        two-element length and failed there with a message about
        broadcasting. Mix the channels separately, or use
        :func:`mix_stereo`.

    See Also
    --------
    mix_stereo : the same sum for two-channel sounds.
    mix_many : the same sum over a list of sounds of any lengths, with
               per-sound offsets and a choice of aligning their starts
               or their ends.

    Examples
    --------
    >>> both = mix(note(freq=220, duration=1), note(freq=330, duration=1))
    >>> len(both) == len(note(freq=220, duration=1))
    True
    >>> float(abs(both).max()) > 1.0   # two notes sum past full scale
    True
    """
    for name, vector in (("first_sonic_vector", first_sonic_vector),
                         ("second_sonic_vector", second_sonic_vector)):
        if np.ndim(vector) != 1:
            raise ValueError(
                f"{name} has {np.ndim(vector)} dimensions; mix takes mono "
                "sounds, since it pads to the longer of the two and a "
                "stereo array's length is its channel count. Mix the "
                "channels separately, or use mix_stereo")
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

    This function mixes two stereo sonic vectors. A mono one is put in both
    channels first, and the shorter is padded with zeros to the length of
    the longer.

    Parameters
    ----------
    first_sonic_vector : ndarray
        The first stereo sonic vector to mix.
    second_sonic_vector : ndarray, optional
        The second stereo sonic vector to mix, by default None: nothing,
        so the first comes back in stereo as it was.
    end : bool, optional
        If False, both sounds start together and the shorter is padded at
        the end. If True, both sounds end together and the shorter is padded
        at the beginning. Defaults to False.

    Returns
    -------
    ndarray
        A stereo sonic vector containing the mix of the input sonic vectors.

    Notes
    -----
    If `second_sonic_vector` is not provided, the `end` parameter is ignored.

    Given one sound, this returned it mixed with itself, at twice its
    amplitude: the default was the empty list, padded to the first
    sound's length, until a cleanup in 2024 made it None and put the first
    sound in its place.

    Examples
    --------
    >>> left = horizontal_stack(note(220, 0.5), silence(0.5))
    >>> right = horizontal_stack(silence(0.5), note(330, 0.5))
    >>> together = mix_stereo(convert_to_stereo(left),
    ...                       convert_to_stereo(right))
    >>> together.shape
    (2, 44100)
    """
    # ndim, not len: a two-sample mono vector also has len() == 2, and was
    # taken for a stereo pair whose channels were then indexed as scalars.
    first_sonic_vector = np.asarray(first_sonic_vector, dtype=np.float64)
    if first_sonic_vector.ndim != 2:
        first_sonic_vector = np.array((first_sonic_vector, first_sonic_vector))
    if second_sonic_vector is None:
        second_sonic_vector = np.zeros((2, 0))
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

    Examples
    --------
    >>> stereo = convert_to_stereo(note(440, 0.1))
    >>> # `loud` takes one channel; this runs it on each of the two.
    >>> both = resolve_stereo(loud, {"sonic_vector": stereo, "trans_dev": 6})
    >>> both.shape[0]
    2
    """
    ag1 = argdict.copy()
    ag2 = argdict.copy()
    for v in stereo_vars:
        stereo = convert_to_stereo(argdict[v])
        ag1[v] = stereo[0]
        ag2[v] = stereo[1]

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

    Raises
    ------
    ValueError
        If the sound has no dimensions or more than two: a number failed
        as an ``IndexError``, and a third axis was taken for channels.

    Examples
    --------
    >>> mono_vector = np.array([0.1, 0.2, 0.3, 0.4])
    >>> stereo_vector = convert_to_stereo(mono_vector)
    >>> stereo_vector.shape
    (2, 4)
    """
    # Keep channel sums in floating-point PCM even when the input is integer
    # data; adding extra channels in an integer dtype can overflow.
    sound_array = np.array(sound_vector, dtype=np.float64)
    if sound_array.ndim not in (1, 2):
        raise ValueError(
            f"convert_to_stereo takes a mono sound, or one sound's "
            f"channels; got an array of shape {sound_array.shape}")

    # Check the shape of the input array
    if len(sound_array.shape) == 1:
        # If the input vector is mono, duplicate it for both left and right
        # channels.
        stereo_sound = np.array((sound_array, sound_array))
    elif sound_array.shape[0] == 1:
        # Mono written as one row is mono too. It was returned as it was,
        # a single channel from a routine that promises two.
        stereo_sound = np.vstack((sound_array, sound_array))
    elif sound_array.shape[0] > 2:
        # If the input vector has more than two channels, keep only the first
        # two (left and right) and sum the rest to both left and right channels
        warn(
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


def _transition_is_linear(method: str, name: str = 'method') -> bool:
    """Whether `method` names a linear transition rather than an exponential.

    Every routine with a linear-or-exponential transition takes the same
    four names. They used to read them three ways: `loud` by exact name,
    `fade` by substring, so ``"explicit"`` was exponential, and the
    glissandi by comparison with ``"exp"`` alone, so ``"exponential"``
    swept linearly and an unknown name was linear without a word.

    Raises
    ------
    ValueError
        If `method` is none of ``'lin'``, ``'linear'``, ``'exp'`` and
        ``'exponential'``. `name` is the parameter the caller knows it by.
    """
    if method in ('lin', 'linear'):
        return True
    if method in ('exp', 'exponential'):
        return False
    raise ValueError(
        f"{name} must be 'lin'/'linear' or 'exp'/'exponential'; "
        f"got {method!r}")


def _signed_power(values, index, name):
    """Raise an oscillatory pattern to a distortion index, keeping its sign.

    The article's distortion index applies to ``(i/(Lambda-1))**alpha``, a
    ramp that runs from 0 to 1 and is never negative. This package and the
    MASS reference also carry one on the *oscillatory patterns*, whose
    values are signed, and there a plain ``x ** alpha`` has no real value
    wherever ``x`` is negative. A fractional index therefore produced NaN
    for half of every cycle, and an even one rectified the pattern into
    something that only ever pushed one way.

    Raising the magnitude and restoring the sign gives the reference's own
    value wherever the reference produced a number at all for a whole odd
    index, since ``sign(x)|x|**3`` is ``x**3``, and bends the oscillation
    symmetrically rather than folding it for every other index.

    An index of exactly 1 returns the values themselves rather than
    computing ``x ** 1``, so the undistorted path -- which is every case
    in ``RECONCILIATION.md`` -- is untouched, bit for bit, with no
    dependence on how a particular NumPy build rounds ``power``.

    See DISCREPANCIES.md for what the article does and does not say here.

    A negative index is refused: the oscillation passes through zero,
    where a negative power has no finite value, and it came back as
    infinity behind a warning. `name` is the parameter that gave it, for
    the refusal to name.
    """
    if index < 0:
        raise ValueError(
            f"{name} is a distortion index, which bends an oscillation "
            f"through zero, where a negative power has no finite value; got "
            f"{index}")
    if index == 1:
        return values
    return np.sign(values) * np.abs(values) ** index


def _bent(progress, alpha):
    """`progress`, a ramp from 0 to 1, raised to the index `alpha`.

    The article's ``(i/(Lambda-1))**alpha``, which makes a transition
    slower or faster. The ramp starts at zero, where a negative power has
    no finite value: every glissando and every exponential transition
    given a negative `alpha` came back holding infinity behind a warning,
    so it is refused.
    """
    if alpha < 0:
        raise ValueError(
            f"alpha bends a transition from zero, where a negative power "
            f"has no finite value; got {alpha}")
    return progress ** alpha


def _require_ears(zeta):
    """Refuse a negative distance between the ears.

    Zero is a head with coincident ears, which hear a source at the same
    moment: `localize` and `localize_linear` render the two channels
    alike. Below zero, the interaural delay came out negative and reached
    numpy as an array of negative length.
    """
    if zeta < 0:
        raise ValueError(
            f"zeta is the distance between the ears, in meters, and cannot "
            f"be negative; got {zeta}")


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
    number_of_samples: int | None = None,
    sample_rate: int | None = None,
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
    number_of_samples : int or None
        The offset in samples, taken instead of ``duration`` when given.
        Zero is an offset of zero samples; None, the default, reads
        ``duration`` instead.
    sample_rate : int or None
        The sample rate in Hertz. None, the default, is the rate
        :func:`music.get_sample_rate` returns.

    Returns
    -------
    ndarray
        The two sounds summed at that offset, long enough to hold both.

    See Also
    --------
    mix_many_with_offsets : the same, for any number of sounds, each
                            offset from the mix built so far.
    mix_many : a list of sounds aligned at their starts or their ends.

    Notes
    -----
    A negative ``duration`` must satisfy
    ``-duration * sample_rate < first_sonic_vector.shape[-1]``: the second
    sound cannot begin before the first one does.

    The description of ``duration`` above was truncated mid-sentence, and
    its last line sat at column zero, so numpydoc read "s1 ends." as a
    parameter of this function and rendered it as one.

    Examples
    --------
    >>> late = mix_with_offset(note(220, 1), note(330, 1), duration=0.5)
    >>> len(late) / 44100      # the second note starts half a second in
    1.5
    """
    sample_rate = resolve_rate(sample_rate)
    first_sonic_vector = np.array(first_sonic_vector)
    second_sonic_vector = np.array(second_sonic_vector)
    if 2 in [len(first_sonic_vector.shape), len(second_sonic_vector.shape)]:
        # The parameters were renamed from s1/s2 at some point but these
        # names were not, so the stereo path raised KeyError.
        return resolve_stereo(mix_with_offset, locals(),
                              ['first_sonic_vector', 'second_sonic_vector'])
    dur = duration

    if number_of_samples is None:  # sample in s1 where s2[0] is added
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
        The seconds are counted at :func:`music.get_sample_rate`'s rate.

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


    Examples
    --------
    >>> chord = mix_many_with_offsets(note(220, 1), 0.25, note(277, 1),
    ...                               0.25, note(330, 1))
    >>> len(chord) / 44100     # each offset is from the mix so far
    1.25
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
#: differs from ``mix_with_offset``. It still works, and warns.
mix_with_offset_ = alias(mix_many_with_offsets, 'mix_with_offset_')


def pan_transitions(p=((1, 1), (1, 0), (0, 1), (1, 1)), d=(2, 2, 2),
                    method='lin', sample_rate=None, sonic_vector=None):
    """Applies pan transitions to a sonic vector.

    Parameters
    ----------
    p : list of tuples, optional
        List of pan positions, where each tuple represents the amplitude
        envelope of each channel, by default [(1,1),(1,0),(0,1),(1,1)]
    d : list, optional
        The duration of each transition in seconds, by default [2,2,2].
        Durations past the last transition are not read.
    method : str or list, optional
        How each transition moves: 'lin', 'circ' or 'exp', described
        below. One name is the law of every transition, and a list holds
        one for each, by default 'lin'. Names past the last transition
        are not read.
    sample_rate : int or None, optional
        Sample rate of the audio. None, the default, is the rate
        :func:`music.get_sample_rate` returns.
    sonic_vector : ndarray, optional
        A mono or stereo sound to pan. Without one, the two envelopes
        themselves are returned.

    Returns
    -------
    ndarray
        The ``(2, n)`` panning envelopes, or ``sonic_vector`` scaled by
        them: as long as the sound, holding the last position past the
        end of the transitions.

    Raises
    ------
    ValueError
        If there are fewer durations or methods than transitions, or a
        method is not one of the three. A short list used to reach an
        ``IndexError`` from inside the routine.

        If a transition by 'circ' or 'exp' starts or ends at a negative
        amplitude: they are laws of a channel's power and loudness, which
        have no sign.

    Notes
    -----
    Each pan transition i takes the amplitude of channel c from p[i][c]
    toward p[i+1][c], which the next transition starts at, by the law
    method[i]. Over progress x from 0 toward 1, from amplitude a to b::

        'lin', linear in amplitude:
            a*(1 - x) + b*x
        'circ', linear in power along a quarter circle:
            sqrt((a*cos(pi*x/2))**2 + (b*sin(pi*x/2))**2)
        'exp', linear in decibels:
            a*(b/a)**x

    'exp' to or from silence, where decibels run out, is :func:`fade`'s
    exponential fade: 80 dB of it, then a linear one percent to silence.
    Across (1, 0) to (0, 1), 'lin' keeps the sum of the channels'
    amplitudes and lets their power dip 3 dB halfway; 'circ' keeps their
    power, since cos(x)**2 + sin(x)**2 = 1; 'exp' changes each channel's
    loudness at a steady rate and keeps neither. Until 2.0.0 ``method``
    was not read, and every transition was 'lin'.

    For now, each channel's signal are kept from mixing.
    One immediate possibility is to maintain the expected
    tessitura of the sample amplitudes.
    Say p = [.5,1,0,.5] ~ [(1,1),(1,0),(0,1),(1,1)].
    Then pi,pj = .5,1 might be performed as::

        s1 = s1*.5 -> 0
        s2 = s1*.5 -> (s1+s2)*.5

    Or through sinusoids and exponentials

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

    sample_rate = resolve_rate(sample_rate)
    transitions = len(p) - 1
    laws = [method] * transitions if isinstance(method, str) else method
    for name, given in (("durations", d), ("methods", laws)):
        if len(given) < transitions:
            raise ValueError(
                f"pan_transitions takes one of its {name} for each "
                f"transition between the {len(p)} positions, {transitions}; "
                f"got {len(given)}")
    for i, law in enumerate(laws[:transitions]):
        if law not in ("lin", "circ", "exp"):
            raise ValueError(f"a pan transition is 'lin', 'circ' or 'exp'; "
                             f"got {law!r}")
        if law != "lin" and min(*p[i], *p[i + 1]) < 0:
            raise ValueError(
                f"'{law}' is a law of a channel's power and loudness, which "
                f"have no sign, so it cannot move between {tuple(p[i])} and "
                f"{tuple(p[i + 1])}")
    pp_ = p[0]
    t0_ = []
    t1_ = []
    for i, pp in enumerate(p[1:]):
        # Whole samples, as every other routine counts a duration: the
        # count was np.arange of the product, which rounds a fraction up,
        # so 1.1 s of panning was a sample longer than 1.1 s of note.
        di = int(d[i] * sample_rate)
        di_ = np.arange(di) / di
        t0_.append(_pan_leg(pp_[0], pp[0], di_, laws[i]))
        t1_.append(_pan_leg(pp_[1], pp[1], di_, laws[i]))
        # Each leg starts where the one before it ended. Without this the
        # loop interpolated every leg from p[0], so the third leg of the
        # defaults ran (1,1) -> (1,1) instead of (0,1) -> (1,1): a channel
        # that should have risen from silence sat at full amplitude, and
        # the leg before it dropped to zero, leaving a full-scale step at
        # the join. tests/test_artifacts.py is what found it.
        pp_ = pp
    t0__ = horizontal_stack(*t0_)
    t1__ = horizontal_stack(*t1_)
    t = np.array((t0__, t1__))
    if sonic_vector is not None:
        # An envelope scales the sound. This mixed the two, adding the
        # ramps to it: the sound came back unpanned under a full-scale
        # offset, and panning silence returned the envelopes themselves.
        sonic_vector = convert_to_stereo(sonic_vector)
        length = sonic_vector.shape[1]
        if length > t.shape[1]:
            # The destination, which the ramps approach but never reach.
            final = np.asarray(p[-1], dtype=np.float64).reshape(2, 1)
            t = np.hstack((t, np.repeat(final, length - t.shape[1], axis=1)))
        return sonic_vector * t[:, :length]
    return t


def _pan_leg(start, end, progress, law):
    """One channel's amplitude from `start` toward `end`, by `law`.

    See :func:`pan_transitions`, whose Notes give the three laws.
    """
    if law == "lin":
        return start * (1 - progress) + end * progress
    if law == "circ":
        angle = progress * np.pi / 2
        return np.sqrt((start * np.cos(angle)) ** 2
                       + (end * np.sin(angle)) ** 2)
    if start and end:
        return start * (end / start) ** progress
    if start == end:
        return np.zeros(len(progress))
    # Imported here: the fades are built on this module.
    from .core.filters.fade import fade
    ramp = fade(fade_out=not end, number_of_samples=len(progress))
    return (start or end) * ramp


def mix_many(sonic_vectors, end=False, offset=0, sample_rate=None):
    """Mix sonic vectors of arbitrary lengths.

    The operation consists in summing sample by sample [1].
    This function helps when the sonic_vectors are not
    of the same size.

    Parameters
    ----------
    sonic_vectors : list of sonic_arrays
        The sonic vectors to be summed, mono or stereo. If any is stereo,
        each mono one is put in both channels.
    end : boolean
        If True, sync the final samples.
        If False (default) sync the initial samples.
    offset : list of scalars
        A list of the offsets for each sonic vectors
        in seconds.
    sample_rate : integer or None
        The sample rate. Only used if offset is supplied. None, the
        default, is the rate :func:`music.get_sample_rate` returns.

    Returns
    -------
    S : ndarray
        A numpy array where each value is a PCM sample of
        the resulting sound. No sounds mix to no samples.

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    Examples
    --------
    >>> mixed = mix_many([note(220, 2), note(440, 1)])
    >>> write_wav_mono(mixed)  # writes a WAV of the two notes summed

    """
    sample_rate = resolve_rate(sample_rate)
    sonic_vectors = [np.asarray(s, dtype=np.float64) for s in sonic_vectors]
    if not sonic_vectors:
        # Python's max() said its argument was empty.
        return np.zeros(0)
    # Each sound's length is its last axis. len() counted a stereo sound's
    # channels, so stereo sounds of different lengths, or one with a mono
    # sound, failed in numpy, and two of one length mixed by accident.
    if any(s.ndim == 2 for s in sonic_vectors):
        sonic_vectors = [convert_to_stereo(s) for s in sonic_vectors]

    def padded(sound, samples, before):
        pad = np.zeros(sound.shape[:-1] + (samples,))
        return np.concatenate((pad, sound) if before else (sound, pad),
                              axis=-1)

    if offset:
        for i, off in enumerate(offset):
            if off:
                sonic_vectors[i] = padded(sonic_vectors[i],
                                          int(off * sample_rate), True)

    max_len = max(s.shape[-1] for s in sonic_vectors)
    return np.sum([padded(s, max_len - s.shape[-1], end)
                   for s in sonic_vectors], axis=0)


#: The name this had while it was the second mixer rather than the general
#: one. It still works, and warns.
mix2 = alias(mix_many, 'mix2')


def _describe_array(array, sample_rate, blocks=16):
    """Measure one array: the figures :func:`profile` reports for it."""
    values = np.asarray(array)
    flat = values.reshape(-1)
    samples = values.shape[-1] if values.ndim > 1 else flat.size
    description: dict = {
        'shape': values.shape,
        'samples': int(samples),
        'seconds': float(samples) / sample_rate,
        'dtype': str(values.dtype),
    }
    real_numeric = (np.issubdtype(values.dtype, np.number) and
                    not np.issubdtype(values.dtype, np.complexfloating))
    if flat.size == 0 or not real_numeric:
        description['numeric'] = False
        return description

    finite = flat[np.isfinite(flat)] if np.issubdtype(
        values.dtype, np.floating) else flat
    description['numeric'] = True
    if finite.size == 0:
        description['finite'] = False
        return description
    description['finite'] = True
    # Squaring in the source dtype overflowed for integer PCM and float32
    # arrays. These measurements describe amplitudes, so do the arithmetic
    # in float64; complex arrays have no ordered real minimum or maximum.
    finite = finite.astype(np.float64, copy=False)
    description['mean'] = float(np.mean(finite))
    description['mean_square'] = float(np.mean(np.square(finite)))
    description['rms'] = float(np.sqrt(description['mean_square']))
    description['minimum'] = float(np.min(finite))
    description['maximum'] = float(np.max(finite))

    # The RMS of successive blocks: its spread is the hint of discontinuity
    # the specification asks for, and its mean is the level over the whole.
    block = max(1, finite.size // blocks)
    usable = finite[:block * max(1, finite.size // block)]
    per_block = np.sqrt(np.mean(
        np.square(usable.reshape(-1, block).astype(np.float64)), axis=1))
    description['block_rms_mean'] = float(np.mean(per_block))
    description['block_rms_std'] = float(np.std(per_block))
    return description


def _guess_role(description, pcm_samples=1000):
    """Read a measured array as PCM, as parametrisation, or as neither.

    Inference, not measurement.  The rules are the ones the specification
    in :func:`profile` states, and each guess is returned with the reason
    that produced it so a caller can disagree with it.
    """
    guesses: list = []
    if not description.get('numeric') or not description.get('finite'):
        return guesses

    samples = description['samples']
    mean, low, high = (description['mean'], description['minimum'],
                       description['maximum'])
    bound = max(abs(low), abs(high))
    centred = abs(mean) < 0.05 * max(bound, 1e-12)
    within_unity = bound <= 1.0
    power_of_two = bound > 1.0 and abs(
        bound - 2 ** round(np.log2(bound))) <= 1

    if samples >= pcm_samples and centred and (within_unity or power_of_two):
        guesses.append(('pcm samples',
                        f'{samples} elements, mean {mean:.3g} near zero, '
                        f'bounded by {bound:.6g}'))
    elif samples < pcm_samples and not centred:
        guesses.append(('parametrisation',
                        f'only {samples} elements and an offset mean of '
                        f'{mean:.3g}'))

    if bound >= 150:
        guesses.append(('frequencies in Hz',
                        f'values reach {bound:.6g}, in the hundreds or above'))
    elif 0 <= low and high <= 150:
        near_integer = bool(np.isclose(
            np.round([low, high, mean]), [low, high, mean], atol=1e-6).all())
        steps = description.get('block_rms_std', 0.0)
        if near_integer or steps < 10:
            guesses.append(('MIDI pitches or semitone intervals',
                            'values in [0, 150] that are integers or move '
                            'in steps under 10'))
        else:
            guesses.append(('decibels',
                            'values in [0, 150] moving in steps of tens'))
    return guesses


def profile(adict, sample_rate=None):
    """Summarize a namespace of variables.

    Sorts the names by what they hold, measures every array in it, and
    reads each array as PCM samples or as parametrisation. Written for
    looking at a piece of synthesis code mid-flight: hand it ``locals()``
    and it says which of the names are sounds, how long each is, and which
    of the numbers look like frequencies, decibels or pitches.

    Parameters
    ----------
    adict : dict
        The namespace to describe, mapping names to values -- typically
        ``locals()`` or ``vars()`` from a piece of synthesis code.
    sample_rate : scalar or None
        The sample rate the durations are reckoned at. None, the default,
        is the rate :func:`music.get_sample_rate` returns.

    Returns
    -------
        dict
        ``d['type']`` sorts the names: ``'scalar'`` for numbers and
        strings, ``'collections'`` for dicts, lists, sets and arrays, and
        ``'other'`` for everything else. ``d['analyses']['ndarray']`` maps
        each array name to its measurements -- ``shape``, ``samples`` (the
        last-axis length for multichannel audio), ``seconds`` at
        `sample_rate`, ``mean``, ``mean_square``, ``rms``,
        ``minimum``, ``maximum``, and the mean and standard deviation of
        the RMS taken block by block, whose spread is what discontinuity
        shows up in. Empty and non-real-numeric arrays are marked as
        unmeasurable; arrays with no finite values are marked as non-finite.
        ``d['guesses']`` maps the same names to readings of those
        measurements, each a ``(reading, reason)`` pair.

    See Also
    --------
    amp_to_db : the conversion the decibel reading above is about.
    hz_to_midi : the conversion the pitch reading is about.

    Notes
    -----
    The measurements are measurements; the guesses are guesses, which is
    why they are kept apart. A guess carries the reason that produced it
    so that a caller can disagree with it:

    * A large array whose mean sits at zero and whose values are bounded
      by 1 or by a power of two is read as PCM samples.
    * A short array with an offset mean is read as parametrisation --
      values meant to be used rather than heard, often driving a rhythm.
    * Values reaching into the hundreds are read as frequencies in Hz.
    * Values within [0, 150] are read as MIDI pitches or semitone
      intervals when they are integers or move in steps under 10, and as
      decibels when they move in steps of tens.

    This function used to raise ``NotImplementedError``: its body had been
    a commented-out sketch since it was written, and the docstring above
    was a specification rather than a description. It is now the second of
    these.

    Examples
    --------
    >>> sound = np.sin(2 * np.pi * 440 * np.arange(88200) / 44100)
    >>> summary = profile({'s': sound, 'freqs': np.array([220., 440.])})
    >>> sorted(summary['type']['collections'])
    ['freqs', 's']
    >>> summary['analyses']['ndarray']['s']['seconds']
    2.0
    >>> [reading for reading, _reason in summary['guesses']['s']]
    ['pcm samples']
    """
    sample_rate = resolve_rate(sample_rate)
    types: dict = {'scalar': [], 'collections': [], 'other': []}
    analyses: dict = {'ndarray': {}}
    guesses: dict = {}

    for name in adict:
        value = adict[name]
        if isinstance(value, np.ndarray):
            types['collections'].append(name)
            description = _describe_array(value, sample_rate)
            analyses['ndarray'][name] = description
            guesses[name] = _guess_role(description)
        elif isinstance(value, (dict, list, set, tuple)):
            types['collections'].append(name)
        elif isinstance(value, (str, bytes)) or np.isscalar(value):
            types['scalar'].append(name)
        else:
            types['other'].append(name)

    return {'type': types, 'analyses': analyses, 'guesses': guesses}


def rhythm_to_durations(durations=(4, 2, 2, 4, 1, 1, 1, 1, 2, 2, 4),
                        freqs=None, duration=.25, bpm=None,
                        total_duration=None):
    """Returns durations from rhythmic patterns.

    Parameters
    ----------
    durations : iterable of scalars
        The relative durations of each item (e.g. note).
    freqs : iterable of scalars
        The number of the entry's duration that fits into the pulse.
        If supplied, durations is ignored.
    duration : scalar
        A basic duration (e.g. for the pulse) in seconds.
    bpm : scalar
        Tempo in beats per minute. One beat lasts ``60 / bpm`` seconds.
        If supplied, duration is ignored.
    total_duration: scalar
        The total duration of the sequence in seconds.
        If supplied, both BPM and duration are ignored.

    Returns
    -------
    durs : List of durations in seconds.

    Raises
    ------
    ValueError
        If nothing given fixes a duration. A ``duration`` of zero reads as
        "not supplied" here, the same as ``bpm`` and ``total_duration``
        being absent, and the routine then divided ``None`` by a sum.

        If an entry of ``durations`` is negative, or of ``freqs`` is not
        positive, or a cell for a tuplet has no parts or parts that come
        to nothing. A frequency of zero, and parts summing to zero, failed
        dividing by zero; a negative duration came back negative; and a
        cell of its span alone was dropped, its time with it.

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

    Examples
    --------
    >>> dt = [4, 2, 2, 4, 1,1,1,1, 2, 2, 4]
    >>> durs0 = rhythm_to_durations(dt, duration=.25)
    >>> df = [4, 8, 8, 4, 16, 16, 16, 16, 8, 8, 4]
    >>> durs0_ = rhythm_to_durations(freqs=df, duration=4)
    >>> dtut = [4,2,2, [8, 1,1,1], 4, [4, 1,1,.5,.5], 3,1, 3,1, 4]
    >>> durs1 = rhythm_to_durations(dtut)
    >>> dtuf2 = [4,8,8, [2, 3,3,3], 4, [4, 3,3,6,6], 16/3, 16, 16/3, 16, 4]
    >>> durs1_ = rhythm_to_durations(freqs=dtuf2, duration=4)

    """
    durations = tuple(durations)
    name, entries = (("freqs", tuple(freqs)) if freqs is not None
                     else ("durations", durations))
    for entry in entries:
        cell = (list(entry) if isinstance(entry, (list, tuple, np.ndarray))
                else None)
        values = cell if cell is not None else [entry]
        if cell is not None and len(cell) < 2:
            raise ValueError(
                f"a tuplet in {name} is its span and the parts it is "
                f"divided into; got {cell}")
        if min(values) < 0 or (name == "freqs" and min(values) == 0):
            raise ValueError(
                f"{name} holds {entry}, and an entry of {name} must be "
                f"{'positive' if name == 'freqs' else 'zero or more'}")
        if cell is not None and sum(cell[1:]) == 0:
            raise ValueError(
                f"the parts of a tuplet in {name} come to nothing: {cell}")
    if total_duration is not None:
        if total_duration < 0:
            raise ValueError("total_duration must not be negative")
        dur = None
    elif bpm is not None:
        if bpm <= 0:
            raise ValueError("bpm must be positive")
        dur = 60 / bpm
    else:
        dur = duration
    if (dur is None or dur <= 0) and total_duration is None:
        raise ValueError(
            "nothing here gives a duration: pass a positive duration, a "
            f"bpm, or a total_duration; got duration={duration}, bpm={bpm}, "
            "total_duration=None")
    durs: list[float] = []
    if freqs is not None:
        freqs = tuple(freqs)
        if dur is None:  # obtain from total_dur
            durs_ = [1 / i if not isinstance(i, (list, tuple, np.ndarray))
                     else 1 / i[0] for i in freqs]
            if not durs_:
                return durs
            # Positive, each of them, so their sum is too.
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
        if dur is None:  # obtain from total_dur
            durs_ = [i if not isinstance(i, (list, tuple, np.ndarray))
                     else i[0] for i in durations]
            if not durs_:
                return durs
            if sum(durs_) == 0:
                raise ValueError(
                    "durations cannot distribute total_duration evenly")
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
