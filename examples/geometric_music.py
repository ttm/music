import numpy as np
import music

# 1) start a ѕynth
being = music.legacy.Being()

# 2) set its parameters using sequences to be iterated through
being.d_ = [1/2, 1/4, 1/4]  # durations in seconds
being.fv_ = [0, 1,5,15,150,1500,15000]  # vibrato frequency
being.nu_ = [5]  # vibrato depth in semitones (maximum deviation of pitch)
being.f_ = [220, 330]  # frequencies for the notes

s1 = being.render(30)
being.f_ += [440]
being.fv_ = [1,2,3,4,5]
s2 = being.render(30)
s3 = music.utils.horizontal_stack(s1, s2, s1 + s2, (s1, s2),
       s1*s2[::-1],
       s1[::7] + s2[::7])

# X) Tweak with special sets of permutations derived from change ringing (campanology)
# or from finite group theory (algebra):
nel = 4
pe4 = music.structures.peals.PlainChanges.PlainChanges(nel)
being.perms = pe4.peal_direct
being.domain = [220*2**(i/12) for i in (0,3,6,9)]
being.curseq = 'f_'
being.f_ = []
nnotes = len(being.perms)*nel  # len(being.perms) == factorial(nel)
being.stay(nnotes)
being.nu_= [0]
being.d_ += [1/2]
s4 = being.render(nnotes)

b2 = music.legacy.Being()
b2.perms = pe4.peal_direct
b2.domain = being.domain[::-1]
b2.curseq = 'f_'
b2.f_ = []
nnotes = len(being.perms)*nel  # len(being.perms) == factorial(nel)
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
