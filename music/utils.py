import numpy as n
from scipy.io import wavfile
# from .functions import *
from numbers import Number
from . import core

def H(*args):
    stereo = 0
    args = [n.array(a) for a in args]
    for a in args:
        if len(a.shape) == 2:
            stereo = 1
    if stereo:
        for i,a in enumerate(args):
            if len(a.shape) == 1:
                args[i] = n.array(( a, a ))
    return n.hstack(args)

def read(fname):
    s = wavfile.read(fname)
    if s[1].dtype != 'int16':
        print('implement non 16bit samples!')
        return
    fs_ = s[0]
    if len(s[1].shape) == 2:
        # l = s[1][:,0]/2**15
        # r = s[1][:,1]/2**15
        return n.array( s[1].T/2**15 )
    else:
        return s[1]/2**15

def V(*args):
    return n.vstack(args)
def db2Amp(db_difference):
    """Receives difference in decibels, returns amplitude proportion"""
    return 10.**(db_difference/20.)
def amp2Db(amp_difference):
    """Receives amplitude proportion, returns decibel difference"""
    return 20.*n.log10(amp_difference)
def hz2Midi(hz_val):
    """Receives Herz value and returns midi note value"""
    return 69+12*n.log2(hz_val/440)
def midi2Hz(midi_val):
    """Receives midi note value and returns corresponding Herz frequency"""
    #return 440*n.log2((69+midi_val)/69)
    return 440*2**((midi_val-69)/12.)
def midi2HzInterval(midi_interval):
    return 2**(midi_interval/12)
def p2f(f0=220.,semitones=[0,7,7,4,7,0]):
    return [f0*2**(i/12) for i in semitones]

def normalize(vector):
    vector=vector.astype(n.float64)
    v = vector
    v = -1+2*(vector-vector.min())/(vector.max()-vector.min())
    if len(v.shape) == 2:
        v[0] = v[0] - v[0].mean()
        v[1] = v[1] - v[1].mean()
    else:
        v = v - v.mean()
    return v
normalize_=normalize
def normalizeRows(vector):
    """Normalize each row of a bidimensional vector to [0,1]"""
    vector=vector.astype(n.float64)
    vector=((n.subtract(self.vector.T,self.vector.min(1)) / (self.vector.max(1)-self.vector.min(1))).T)
    return vector
def write(sonic_vector,filename="sound_music_name.wav", normalize=True,samplerate=44100):
    if normalize:
        sonic_vector=normalize_(sonic_vector)
    sonic_vector_ = n.int16(sonic_vector * float(2**15-1))
    wavfile.write(filename,samplerate, sonic_vector_) # escrita do som
def mix(self,list1,list2):
    """
    See Also
    --------
    (.functions).mix2 : a better mixer

    """
    l1=len(list1); l2=len(list2)
    if l1<l2:
        sound=n.zeros(l2)
        sound+=list2
        sound[:l1]+=list1
    else:
        sound=n.zeros(l1)
        sound+=list1
        sound[:l2]+=list2
    return sound
def mixS(l1, l2=[], end=False):
    if len(l1) != 2:
        l1 = n.array((l1, l1))
    if len(l2) != 2:
        l2 = n.array((l2, l2))
    if len(l1[0]) > len(l2[0]):
        if not end:
            l2_ = H( l2, n.zeros(( 2, len(l1[0])-len(l2[0]) )) )
        else:
            l2_ = H( n.zeros(( 2, len(l1[0])-len(l2[0]) )), l2 )
        l1_ = l1
    else:
        if not end:
            l1_ = H( l1, n.zeros(( 2, len(l2[0])-len(l1[0]) )) )
        else:
            l1_ = H( n.zeros(( 2, len(l2[0])-len(l1[0]) )), l1 )
        l2_ = l2
    return l1_+l2_

def CF(s1, s2, dur=500, method='lin', fs=44100):
    """
    Cross fade in dur milisseconds.

    """
    ns = int(dur*fs/1000)
    if len(s1.shape) != len(s2.shape):
        print('enter s1 and s2 with the same shape')
    if len(s1.shape) == 2:
        s1_ = CF(s1[0], s2[0], locals()['dur'],
                locals()['method'], locals()['fs'])
        s2_ = CF(s1[1], s2[1], locals()['dur'],
                locals()['method'], locals()['fs'])
        s = n.array( (s1_, s2_) )
        return s
    s1[-ns:] *= core.F(nsamples=ns, method=method, fs=fs)
    s2[:ns] *= core.F(nsamples=ns, method=method, fs=fs, out=False)
    s = J(s1, s2, d = -dur/1000)
    return s


def resolveStereo(afunction, argdict, stereovars=['sonic_vector']):
    ag1 = argdict.copy()
    ag2 = argdict.copy()
    for v in stereovars:
        argdict[v] = stereo(argdict[v])
        sv1 = argdict[v][0]
        sv2 = argdict[v][1]
        ag1[v] = sv1
        ag2[v] = sv2

    # sv1 = argdict['sonic_vector'][0]
    # sv2 = argdict['sonic_vector'][1]
    # ag1 = argdict.copy()
    # ag1['sonic_vector'] = sv1
    # ag2 = argdict.copy()
    # ag2['sonic_vector'] = sv2
    sv1_ = afunction(**ag1)
    sv2_ = afunction(**ag2)
    s = n.array( (sv1_, sv2_) )
    return s

def stereo(sonic_vector):
    s = n.array(sonic_vector)
    if len(s.shape) == 1:
        ss = n.array(( s, s ))
    elif s.shape[0] > 2:
        print('keeping first two channels in left and right. The rest will be added to both left and right')
        ss = n.array(( s[0], s[1] ))
        for sss in s[2:]:
            ss += sss
    else:
        ss = s
    return ss

def J(s1, s2, d=0, nsamples=0, fs=44100):
    """
    Mix s1 and s2 placing the beggining of s2 after s1 by dur seconds

    Parameters
    ----------
    s1 : numeric array
        A sequence of PCM samples.
    s2 : numeric array
        Another sequence of PCM samples.
    d : numeric
        The offset of the second sound, i.e. the displacement that
        the start of the second sound. (First sound has offset 0).
        Might be negative, denoting to start sound2 |d| seconds before s1 ends.

    Notes
    -----
    if d<0, it should satisfy -d*fs < s1.shape[-1]

    TODO: enhance/recycle J_ and mix2 or delete them. TTM 

    See Also
    --------
    (.functions).mix2 : a better mixer

    """
    s1 = n.array(s1)
    s2 = n.array(s2)
    if 2 in [len(s1.shape), len(s2.shape)]:
        return resolveStereo(J, locals(), ['s1', 's2'])
    dur = d

    if not nsamples:  # sample in s1 where s2[0] is added
        ns = dur*fs
    else:
        ns = nsamples

    if ns >= 0:
        nst = ns + len(s2)
    else:
        nst = len(s1) + len(s2) + ns 

    if nst < len(s1):
        nst = len(s1)

    s = n.zeros(int(nst))
    s[:len(s1)] += s1
    print('s.shape', 's1.shape', 's2.shape', 'ns', 'nst',
            s.shape, s1.shape, s2.shape, ns, nst)
    if ns >= 0:
        s[ns : ns+len(s2)] += s2
        # s[-len(s2):] += s2
    else:
        s[int(len(s1)+ns): int(len(s1)+ns+len(s2))] += s2
    return s

def J_(*args):
    """
    Mix sonic vectors with offsets.

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
    # return n.zeros(args[0].shape[-1])
    # return mix2(sounds, False, offsets, 44100)
    i = 0 # DEPRECATED
    s = []
    while i < len(args):
        a = args[i] # new array
        if type(a) not in (n.ndarray, list):
            print("Something odd happened,")
            print("skipping a value that should have heen a sequence of numbers:", a)
            i += 1
            continue
        if len(args) > i+1:
            offset = args[i+1] # potentialy duration
            if n.isscalar(offset):
                i += 2
            else:
                offset = 0
                i += 1
        else:
            offset = 0
            i += 1
        s = J(s, a, d=offset)
    return s

def panTransitions(p=[(1,1),(1,0),(0,1),(1,1)], d=[2,2,2],
        method=['lin','circ','exp'], fs=44100, sonic_vector=None):
    """
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
    """

    pp_ = p[0]
    t0_ = []
    t1_ = []
    for i, pp in enumerate(p[1:]):
        # t0 = pp[0] - pp_[0]
        # t1 = pp[1] - pp_[1]
        di = d[i]*fs
        di_ = n.arange(di)/di
        t0 = pp_[0]*(1-di_) + pp[0]*di_
        t1 = pp_[1]*(1-di_) + pp[1]*di_
        t0_.append(t0)
        t1_.append(t1)
    t0__ = H(*t0_)
    t1__ = H(*t1_)
    t = n.array(( t0__, t1__))
    if sonic_vector:
        sonic_vector = stereo(sonic_vector)
        return J(sonic_vector, t)
    else:
        return t


def mix2(sonic_vectors, end=False, offset=0, fs=44100):
    """
    Mix sonic vectors. MALFUNCTION! TTM TODO
    
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
    fs : integer
        The sample rate. Only used if offset is supplied.

    Returns
    -------
    S : ndarray
        A numpy array where each value is a PCM sample of
        the resulting sound.

    Examples
    --------
    >>> W(mix2(sonic_vectors=[V(), N()]))  # writes a WAV file with nodes

    Notes
    -----
    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if offset:
        for i, o in enumerate(offset):
            sonic_vectors[i] = H(n.zeros(o*fs), sonic_vectors[i] )
            
    amax = 0
    for s in sonic_vectors:
        amax = max(amax, len(s))
    for i in range(len(sonic_vectors)):
        if len(sonic_vectors[i]) < amax:
            if end:
                sonic_vectors[i] = n.hstack(( n.zeros(amax-len(sonic_vectors[i])), sonic_vectors[i] ))
            else:
                sonic_vectors[i] = H( sonic_vectors[i], n.zeros(amax-len(sonic_vectors[i])) )
    s = J_(*sonic_vectors)
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

      d['analyses']['ndarray'] should return a general analysis of the ndarrays,
      including size in seconds of each considering fs.
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
    #     if type(sonic_vector) == n.ndarray:
    #     elif type(sonic_vector) == list:
    #     elif n.isscalar(avar):
    #     else:
    #         print('unrecognized type, implement dealing with it')
    
V_ = V
