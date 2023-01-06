import sys
keys = tuple(sys.modules.keys())
for key in keys:
    if "music" in key:
        del sys.modules[key]
from music.core import *
from music import *
print('imported')

# basic 5Hz monaural beat:

f = 220
d = 60 * 15  # 15 min
tab = S

##### basic beat:
# s1 = N(f, d, tab)
# s2 = N(f + 5, d, tab)
# W(s1 + s2, 'monaural-5Hz.wav')
# WS([s1, s2], 'binaural-5Hz.wav')
# 
# s1 = N(f, d, tab)
# s2 = N(f + 40, d, tab)
# W(s1 + s2, 'monaural-40Hz.wav')
# WS([s1, s2], 'binaural-40Hz.wav')  # [left, right]


##### crossfade to oscilate s1 and s2 in stereo:
# s1 = N(f, d, tab)
# s2 = N(f + 40, d, tab)
# linear = n.linspace(0, 1, s1.shape[0])
# linear_ = 1 - linear
# 
# left = s1 * linear + s2 * linear_
# right = s1 * linear_ + s2 * linear
# W(s1 + s2, 'monaural-40Hz_same.wav')
# WS([left, right], 'binaural-40Hz_crossfade.wav')

##### vibrato
## monoaural beat through a vibrato:
# s = V(d=60 * 15, fv=40, tab=S)
# W(s, 'monaural-40Hz_vibrato.wav')

## 40Hz beat with a 3Hz vibrato:
# fv = 3
# s1 = V(f, d, fv, tab=S)
# s2 = V(f + 40, d, fv, tab=S)
# W(s1 + s2, 'monaural-40Hz_with_3Hz_vibrato.wav')
# WS([s1, s2], 'binaural-40Hz_with_3Hz_vibrato.wav')
# 
## 40Hz beat with a 5Hz vibrato:
# s1 = V(f=f, d=60 * 15, fv=5, tab=S)
# s2 = V(f=f + 40, d=60 * 15, fv=5, tab=S)
# W(s1 + s2, 'monaural-40Hz_with_5Hz_vibrato.wav')
# WS([s1, s2], 'binaural-40Hz_with_5Hz_vibrato.wav')
# 
###### first martigli:
## 40Hz beat with a 20s vibrato period (1 / 20 = 0.05Hz vibrato):
## i.e.: a 20s martigli oscilatio _base_ with a 40Hz ([bi-mo]naural) entrainment
# mp0 = 20
# fv = mf0 = 1 / mp0
# s1 = V(f, d, fv, tab=S)
# s2 = V(f + 40, d, fv, tab=S)
# W(s1 + s2, 'monaural-40Hz_with_20s_vibrato.wav')
# WS([s1, s2], 'binaural-40Hz_with_20s_vibrato.wav')

###### tremolo
# monaural beat through a tremolo:
# s = V(d=d, tab=tab, fv=0, nu=2) * T(d, 40)
# W(s, 'monaural-40Hz_tremolo.wav')

###### mixing basic periodic cues:
# note220 = N(f, d, tab)
# natural = note220 + N(f + 40, d, tab)
# tremolo = note220 * T(d, 40)
# vibrato = V(d=60 * 15, fv=40, tab=S)
# W(natural + tremolo + vibrato, 'monaural-40Hz_3cues.wav')

# fe = 40  # frequency of entrainment
# f = 240  # multiple of 40
# note240 = N(f, d, tab)
# natural = note240 + N(f + fe, d, tab)
# tremolo = note240 * T(d, fe)
# vibrato = V(f, d, fv=fe, tab=S)
# W(natural + tremolo + vibrato, 'monaural-f0240_40Hz_3cues.wav')

# fe = 5  # frequency of entrainment
# f = 240  # multiple of 40
# note240 = N(f, d, tab)
# natural = note240 + N(f + fe, d, tab)
# tremolo = note240 * T(d, fe)
# vibrato = V(f, d, fv=fe, tab=S)
# W(natural + tremolo + vibrato, 'monaural-f0240_40Hz_3cues.wav')
# 
# fe = 5  # frequency of entrainment
# f = 240  # multiple of 40
# note240 = N(f, d, tab)
# natural = note240 + N(f + fe, d, tab)
# tremolo = note240 * T(d, fe)
# vibrato = V(f, d, fv=fe, tab=S)
# 
# ###### noise:
# fs = 44100
# ndur = 1 / (2 * fe)
# nsamples=int(fs * ndur)
# noise = noises(nsamples=nsamples, fmin=60)
# silence = n.zeros(nsamples)
# hit = H([noise, silence])
# min15 = 60 * 15
# nsamples_ = natural.shape[0]
# indexes = n.arange(nsamples_) % hit.shape[0]
# noise15min = hit[indexes]
# # 
# # W(noise15min, 'noiseHit5Hz.wav')
# W(natural + tremolo + vibrato + noise15min, 'monaural-f0240_5Hz_4cues_.wav')
# 
# 
# # adsr noise
# fe = 3
# fs = 44100
# ndur = 1 / fe
# nsamples = int(fs * ndur)
# noise = AD(nsamples=nsamples) * noises(nsamples=nsamples, fmin=100)
# min15 = 60 * 15
# nsamples_ = natural.shape[0]
# indexes = n.arange(nsamples_) % noise.shape[0]
# noise15min = noise[indexes]
# 
# W(noise15min, 'noiseHit3HzADSR_.wav')
# 
# 
# natural = note240 + N(f + fe, d, tab)
# tremolo = note240 * T(d, fe)
# vibrato = V(f, d, fv=fe, tab=S)
# W(natural + tremolo + vibrato + noise15min, 'monaural-f0240_3Hz_4cuesADSR_.wav')
#####  other ways to convey oscillations for entrainment stimulation:
# also FM() AM() noises() AD() loc() FIR() IIR() R() reverb PV_() PV()

##### symetry
# peals, permutations, symmetric scales, permutation of parameters and sound themselves

# 4 basic techiques: symetry, Martigli, binaural, Martigli-binaural

##### combination of the techniques ##########################

# symmetry + Martigli
# symmetry + binaural
# symmetry + Martigli-binaural
# Martigli + binaural
# Martigli + Martigli-binaural
# binaural + Martigli-binaural
# symmetry + binaural + Martigli-binaural

# pan, timbre, volume, Envelope, glissandi...

#### exploration of each tech ####

# beat glissandi
# s2 = N(f, d, tab)
# s1 = P(f + 0.2, f + 60, d)
# W(s1 + s2, 'monaural-0.2-60Hz.wav')
# WS([s1, s2], 'binaural-0.2-60Hz.wav')

# martigli-beat with glissandi
martigli_depth = nu = 4  # semitones
mp0 = 20
mf0 = 1 / mp0
# s2 = V(f, d, mf0, nu, tab=tab)
# s1 = PV(f + 0.2, f + 60, d, mf0, nu)
# W(s1 + s2, 'monaural-0.2-60Hz-20sMartigli.wav')
# WS([s1, s2], 'binaural-0.2-60Hz-20sMartigli.wav')

##############
# martigli with a 2.5Hz vibrato
# s1 = VV(f, d, 2.5, mf0, tab=tab)
# W(s1, 'monaural-monofonic-Martigli+2.5vibSin.wav')
# 
# # martigli with a 5Hz vibrato
# s2 = VV(f * 2, d, 5, mf0, tab=tab)
# W(s1 + s2, 'monaural-Martiglis+2.5+5vibSin.wav')
# WS([s1, s2], 'binaural-Martiglis+2.5+5vibSin.wav')
# 
##############
# # martigli with a 0.5Hz vibrato
# s1 = VV(f, d, 0.5, mf0, tab=tab)
# W(s1, 'monaural-monofonic-Martigli+2.5vibSin.wav')

# martigli with a 0.25Hz vibrato
# s2 = VV(f, d, 0.25, mf0, tab=tab)
# W(s1 + s2, 'monaural-Martiglis+0.5+0.25vibSin.wav')
# WS([s1, s2], 'binaural-Martiglis+0.5+0.25vibSin.wav')

###########
# # martigli-beat-glissando with vibrato
# # 0.25Hz vibrato
# s1 = VV(f, d, 5, mf0, tab=tab)
# s2 = PVV(f, f + 40, d, 5, mf0, tab=tab)
# W(s1 + s2, 'monaural-Martiglis+0.5+0.25vibSinGliss0-40.wav')
# WS([s1, s2], 'binaural-Martiglis+0.5+0.25vibSinGliss0-40.wav')
# 
# # lowering vibrato depth
# s1 = VV(f, d, 5, mf0, nu1=0.2, tab=tab)
# s2 = PVV(f, f + 40, d, 5, mf0, nu1=0.2, tab=tab)
# W(s1 + s2, 'monaural-Martiglis+0.5+0.25vibSinGliss0-40Soft.wav')
# WS([s1, s2], 'binaural-Martiglis+0.5+0.25vibSinGliss0-40Soft.wav')


#### numerical interlude
# a = sin(w1) + sin(w2)
# b = 2 * sin([w1 + w2] / 2) * sin([w1 - w2] / 2)
# a = b
# Fourier(a) = Fourier(b) 
# have all freq w1, w2, w3 = (w1 + w2) / 2, w4 = (w1 - w2) / 2
# although they are just the sum of w1 and w2?

w1 = 220
w2 = 223
d = 2
fs = 44100
nsamples = fs * d
sample_indexes = n.arange(nsamples)
s1 = n.sin(w1 * 2 * n.pi * sample_indexes / fs)
s2 = n.sin(w2 * 2 * n.pi * sample_indexes / fs)
s12 = s1 + s2
W(s12, 'monaural-220-240-spectrum.wav')
WS([s1, s2], 'binaural-220-240-spectrum.wav')
# we can hear the 20Hz beat, but there is no fourier signature


p.plot(n.abs(n.fft.fft(s1)))
p.plot(n.abs(n.fft.fft(s2)))
p.show()
p.plot(n.abs(n.fft.fft(s12))) # zero spectral signature for 
p.show()

# beat mapping (w1, w2), each 2 frequency becomes the mean and deviation:
w3 = (w1 + w2) / 2
w4 = (w1 - w2) / 2
s3 = n.sin(w3 * 2 * n.pi * sample_indexes / fs)
s4 = n.sin(w4 * 2 * n.pi * sample_indexes / fs)
s34_ = 2 * s3 * s4
# a = b => s1 + s2 = 2 * s3 * s4

# beat mapping(w3, w4) becomes w1, w2

# in practice, do we hear both.
# the frequency and temporal resolution are inversely proportional.

# in the case above, the FFT has the 220 and 223 coefficients (w1, w2),
# we hear 221.5 and a 1.5 beat (w3, w4).
# in this case, the sensation w3, w4 is heard although the physical stimulus was w1 + w2.

# what happens if we use not 2 real frequencies, but 3?

# when our resolution between w1 & w2 gives up, we hear w3 and w4.


diff = 

p.plot(n.abs(n.fft.fft(s1)))
p.plot(n.abs(n.fft.fft(s2)))
p.show()
p.plot(n.abs(n.fft.fft(s2 + s1))) # zero spectral signature for 
p.show()





# martigli-binaural-sym,
#   central freq is note
#   glissandi in central martigli to change note
#   symmetry for the notes chosen

# f = 220
# d = 60 * 15  # 15 min
# tab = S
# mp0 = 20
# fv = mf0 = 1 / mp0
# fe = 40
# s1 = V(f, d, fv, tab=S)
# s2 = V(f + fe, d, fv, tab=S)
# W(s1 + s2, f'monaural-{fe}Hz_with_{mp0}s_vibrato.wav')
# WS([s1, s2], f'binaural-{fe}Hz_with_{mp0}s_vibrato.wav')
# 
# # symmetric scale
# symFreq = 5
# sym_ambit = 2  # an octave
# note_freqs = [f * sym_ambit ** (i / symFreq) for i in range(symFreq)] 
# 
# sym = structures.symmetry.PlainChanges(symFreq)
# seqs = sym.act()
# seqs_freq = [[note_freqs[i] for i in j] for j in seqs]
# compasses = len(seqs)
# notes = compasses * len(seqs[0])
# 
# nsamples_note = s1.shape[0] / notes

