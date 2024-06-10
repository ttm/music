"""
This module is home of some basic functions used throughout the `music` package
which are not strictly related to the field of computational music.
"""

import numpy as np

LAMBDA_TILDE = 1024 * 16
WAVEFORM_SINE = np.sin(np.linspace(0, 2 * np.pi, LAMBDA_TILDE, endpoint=False))
WAVEFORM_SAWTOOTH = np.linspace(-1, 1, LAMBDA_TILDE)
WAVEFORM_SQUARE = np.hstack((np.ones(int(LAMBDA_TILDE / 2)) * -1,
                             np.ones(int(LAMBDA_TILDE / 2))))
triangular_tmp = np.linspace(-1, 1, LAMBDA_TILDE // 2, endpoint=False)
WAVEFORM_TRIANGULAR = np.hstack((triangular_tmp, triangular_tmp[::-1]))


def horizontal_stack(*arrays):
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
    arrays = [np.array(arr) for arr in arrays]

    # Check if any of the input arrays are bidimensional (stereo)
    for arr in arrays:
        if len(arr.shape) == 2:
            stereo_present = True
            break

    # If stereo data is present, ensure that mono channels are duplicated to
    # both left and right channels
    if stereo_present:
        for i, arr in enumerate(arrays):
            if len(arr.shape) == 1:
                arrays[i] = np.array((arr, arr))

    # Return the horizontal stack of arrays
    return np.hstack(arrays)


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
    return 69 + 12 * np.log2(hertz_value / 440)


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


def pitch_to_freq(start_freq: float = 220.,
                  semitones: list = [0, 7, 7, 4, 7, 0]) -> list:
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
    mix2 : A better mixer function that provides more control over the mixing
           process.
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


def mix_stereo(first_sonic_vector: np.ndarray,
               second_sonic_vector: np.ndarray = [],
               end: bool = False) -> np.ndarray:
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
    if len(first_sonic_vector) != 2:
        first_sonic_vector = np.array((first_sonic_vector, first_sonic_vector))
    if second_sonic_vector is None:
        second_sonic_vector = first_sonic_vector
    else:
        if len(second_sonic_vector) != 2:
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


def resolve_stereo(afunction, argdict, stereo_vars=['sonic_vector']):
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


def convert_to_stereo(sound_vector):
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
        print('Keeping first two channels in left and right. '
              'The rest will be added to both left and right.')
        stereo_sound = np.array((sound_array[0], sound_array[1]))
        for channel in sound_array[2:]:
            stereo_sound += channel
    else:
        # If the input vector is already stereo, return it without any
        # modifications
        stereo_sound = sound_array

    return stereo_sound


def mix_with_offset(first_sonic_vector, second_sonic_vector,
                    duration=0, number_of_samples=0, sample_rate=44100):
    """Mix two sonic vector by placing the beginning of the second one
    a specified number of seconds after the first one.

    Parameters
    ----------
    first_sonic_vector : numeric array
        A sequence of PCM samples.
    second_sonic_vector : numeric array
        Another sequence of PCM samples.
    duration : numeric
        The offset of the second sound, i.e. the displacement that
        the start of the second sound. (First sound has offset 0).
        Might be negative, denoting to start sound2 |d| seconds before s1 ends.
    number_of_samples : int
    sample_rate : int

    Notes
    -----
    if d<0, it should satisfy -d*fs < s1.shape[-1]

    TODO: enhance/recycle J_ and mix2 or delete them. TTM

    See Also
    --------
    (.functions).mix2 : a better mixer

    """
    first_sonic_vector = np.array(first_sonic_vector)
    second_sonic_vector = np.array(second_sonic_vector)
    if 2 in [len(first_sonic_vector.shape), len(second_sonic_vector.shape)]:
        return resolve_stereo(mix_with_offset, locals(), ['s1', 's2'])
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
    print('s.shape', 's1.shape', 's2.shape', 'ns', 'nst', s.shape,
          first_sonic_vector.shape, second_sonic_vector.shape, ns, nst)
    if ns >= 0:
        s[ns: ns + len(second_sonic_vector)] += second_sonic_vector
        # s[-len(s2):] += s2
    else:
        s[int(len(first_sonic_vector) + ns):
          int(len(first_sonic_vector) + ns +
              len(second_sonic_vector))] += second_sonic_vector
    return s


def mix_with_offset_(*args):
    """Mix sonic vectors with offsets.

    Parameters
    ----------
    J_ receives a sequence of sonic vectors,
    each a sequence of PCM samples.
    Or a sequence alternating the sonic vectors
    and their offsets.

    See Also
    --------
    (.functions).mix2 : a better mixer

    """
    # i = 0 # DEPRECATED
    # sounds = []
    # offsets = []
    # while i < len(args):
    #     print(i)
    #     the_sound = args[i]
    #     if len(args) < i+1:
    #         offset = args[i+1]
    #         if isinstance(offset, Number):
    #             i += 2
    #         else:
    #             offset = 0
    #             i += 1
    #     else:
    #         offset = 0
    #         i += 1
    #     sounds.append(the_sound)
    #     offsets.append(offset)
    # return np.zeros(args[0].shape[-1])
    # return mix2(sounds, False, offsets, 44100)
    i = 0  # DEPRECATED
    s = []
    while i < len(args):
        a = args[i]  # new array
        if type(a) not in (np.ndarray, list):
            print("Something odd happened,")
            print("skipping a value that should have been\
                  a sequence of numbers:", a)
            i += 1
            continue
        if len(args) > i + 1:
            offset = args[i + 1]  # potentialy duration
            if np.isscalar(offset):
                i += 2
            else:
                offset = 0
                i += 1
        else:
            offset = 0
            i += 1
        s = mix_with_offset(s, a, duration=offset)
    return s


def pan_transitions(p=[(1, 1), (1, 0), (0, 1), (1, 1)], d=[2, 2, 2],
                    method=['lin', 'circ', 'exp'], sample_rate=44100,
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
    to understand the pan transition methods:

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
    Then pi,pj = .5,1 might be performed as:
    s1 = s1*.5 -> 0
    s2 = s1*.5 -> (s1+s2)*.5
    Or through sinusoids and expotentials

    Make fast and slow fades and parameter transitions
    using weber-fechner and steven's laws.
    E.g. pitch_trans = [pitch0*X**(i/Y) for i in range(12)]
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
    if sonic_vector:
        sonic_vector = convert_to_stereo(sonic_vector)
        return mix_with_offset(sonic_vector, t)
    else:
        return t


# FIXME: malfunction
def mix2(sonic_vectors, end=False, offset=0, sample_rate=44100):
    """Mix sonic vectors. MALFUNCTION! TTM TODO

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
    >>> W(mix2(sonic_vectors=[np.vstack(), N()]))  # writes a WAV file
                                                   # with nodes

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
           representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if offset:
        for i, o in enumerate(offset):
            sonic_vectors[i] = horizontal_stack(np.zeros(o * sample_rate),
                                                sonic_vectors[i])

    amax = 0
    for s in sonic_vectors:
        amax = max(amax, len(s))
    for i in range(len(sonic_vectors)):
        if len(sonic_vectors[i]) < amax:
            if end:
                sonic_vectors[i] = np.hstack(
                    (np.zeros(amax - len(sonic_vectors[i])), sonic_vectors[i]))
            else:
                sonic_vectors[i] = horizontal_stack(
                    sonic_vectors[i], np.zeros(amax - len(sonic_vectors[i])))
    s = mix_with_offset_(*sonic_vectors)
    return s


def profile(adict):
    """
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
        If it has more than one or two dimensions where the many samples are,
        it might be a collection of audio samples with the sample size

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
    # for key in adict:
    #     avar = adict[key]
    #     if type(sonic_vector) == np.ndarray:
    #     elif type(sonic_vector) == list:
    #     elif np.isscalar(avar):
    #     else:
    #         print('unrecognized type, implement dealing with it')


def rhythm_to_durations(durations=[4, 2, 2, 4, 1, 1, 1, 1, 2, 2, 4],
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
    BPM or total_duration.
    >>> durs = [i*duration for i in durations]

    The frequencies parameter is considered to be in a
    frequential notation: each entry is the number of the
    entry that fits a same duration (also given through duration,
    BPM or total_duration).
    >>> durs = [duration/i for i in freqs]

    The examples above yield (two by two) the same sequences of durations
    by using duration=0.25 when in temporal notation or
    duration=4 when in frequency notation.

    To facilitate the description of rhythms (e.g. for tuplets),
    some set of durations might be an iterable inside durations
    or frequencies. In this case:
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

    It might be a good idea to incorporate also this notation:
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
