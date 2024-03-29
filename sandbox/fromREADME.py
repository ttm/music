import sys
keys = tuple(sys.modules.keys())
for key in keys:
    if "music" in key:
        del sys.modules[key]
sys.path.append("/")
import numpy as np
import music


### Basic usage
T = music.legacy.tables.Basic()


# 1) start a ѕynth
b = music.legacy.Being()

# 2) set its parameters using sequences to be iterated through
b.d_ = [1/2, 1/4, 1/4]  # durations in seconds
b.fv_ = [0, 1,5,15,150,1500,15000]  # vibrato frequency
b.nu_ = [5]  # vibrato depth in semitones (maximum deviation of pitch)
b.f_ = [220, 330]  # frequencies for the notes

# 3) render the wavfile
b.render(30, 'aMusicalSound.wav')  # render 30 notes iterating though the lists above

# 3b) Or the numpy arrays directly and use them to concatenate and/or mix sounds:
s1 = b.render(30)
b.f_ += [440]
b.fv_ = [1,2,3,4,5]
s2 = b.render(30)

# s1 then s2 then s1 and s2 at the same time, then at the same time but one in each LR channel,
# then s1 times s2 reversed, then s1+s2 but jumping 6 samples before using one:
s3 = music.utils.horizontal_stack(s1, s2, s1 + s2, (s1, s2),
       s1*s2[::-1],
       s1[::7] + s2[::7])
music.core.io.write_wav_stereo(s3, 'tempMusic.wav')

# X) Tweak with special sets of permutations derived from change ringing (campanology)
# or from finite group theory (algebra):
nel = 4
pe4 = music.structures.peals.PlainChanges.PlainChanges(nel)
b.perms = pe4.peal_direct
b.domain = [220*2**(i/12) for i in (0,3,6,9)]
b.curseq = 'f_'
b.f_ = []
nnotes = len(b.perms)*nel  # len(b.perms) == factorial(nel)
b.stay(nnotes)
b.nu_= [0]
b.d_ += [1/2]
s4 = b.render(nnotes)

b2 = music.legacy.Being()
b2.perms = pe4.peal_direct
b2.domain = b.domain[::-1]
b2.curseq = 'f_'
b2.f_ = []
nnotes = len(b.perms)*nel  # len(b.perms) == factorial(nel)
b2.stay(nnotes)
b2.nu_= [2,5,10,30,37]
b2.fv_ = [1,3,6,15,100,1000,10000]
b2.d_ = [1,1/6,1/6,1/6]
s42 = b2.render(nnotes)

i4 = music.structures.permutations.InterestingPermutations(4)
b2.perms = i4.rotations
b2.curseq = 'f_'
b2.f_ = []
b2.stay(nnotes)
s43 = b2.render(nnotes)

s43_ = music.core.filters.fade(sonic_vector=s43, duration=5, method='lin')

diff = s4.shape[0] - s42.shape[0]
s42_ = music.utils.horizontal_stack(s42, np.zeros(diff))
s_ = music.utils.horizontal_stack(s3, (s42_, s4), s43_)

music.core.io.write_wav_stereo(s_, 'geometric_music.wav')


##############
# Notice that you might relate a peal or any set of permutations
# to a sonic characteristic (frequency, duration, vibrato depth, vibrato frequency,
# attack duration, etc) through at least 3 methods:
# 1) initiate a Being(), set its permutations to the permutation sequence,
# its domain to the values to be permuted, and its curseq to
# the name of the Being sequence to be yielded by the permutation of the domain.
#
# 2) Achieve the sequence of values through peal.act() or just using permutation(domain)
# for all the permutations at hand.
# Then render the notes directly (e.g. using M.core.V_) or passing the sequence of values
# to a synth, such as Being()
#
# 3) Using IteratorSynth as explained below. (potentially deprecated)

pe3 = music.structures.peals.PlainChanges.PlainChanges(3)
music.structures.symmetry.print_peal(pe3.act(), [0])
freqs = sum(pe3.act([220,440,330]), [])

nnotes = len(freqs)

b = music.legacy.Being()
b.f_ = freqs
b.render(nnotes, 'theSound_campanology.wav')

### OR
b = music.legacy.Being()
b.domain = [220, 440, 330]
b.perms = pe3.peal_direct
b.f_ = []
b.curseq = 'f_'
b.stay(nnotes)
b.render(nnotes, 'theSound_campanology_.wav')


### OR (DEPRECATED, but still kept while not convinced to remove...)
isynth = music.legacy.IteratorSynth.IteratorSynth()
isynth.fundamental_frequency_sequence=freqs
isynth.tab_sequence = [T.sine, T.triangle, T.square, T.saw]

pcm_samples = music.utils.H(*[isynth.renderIterate() for i in range(len(freqs))])

music.core.io.write_wav_mono(pcm_samples, 'something.wav')

