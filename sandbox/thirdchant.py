import sys
keys=tuple(sys.modules.keys())
for key in keys:
    if "music" in key:
        del sys.modules[key]
import music as M
import numpy as n
#from percolation.rdf import c
H = M.utils.stereo_horizontal_stack

# durs  = 1,  2,   1,  2,   1, 2
# notes = 0,  2,   3,  2,   8,7,
# text  = 'wake up  out of your sleep'
# sonic_vector=M.singing.sing(text = text, notes=notes,
#         durs=durs,Q=120,lang='en',M="3/4")

# M.utils.write(sonic_vector,filename="some.wav",samplerate=44100)
#
# sonic_vector=M.singing.sing(text = "wa-a-a-a-a-a", notes=notes,
#         durs=durs,Q=120,lang='en',K="Cm")

sonic_vector = M.singing.sing(text = 'eu a-mo a ta-ís tei-xei-ra fa-bri', notes=(0, 5,5, 4, 2,7, 7,8,7, 4,0), durs=(2, 1,2, 1, 1,2, 1,1,1, 1,2), lang = 'pt',transpose=0, reference=60-36)
M.utils.write(sonic_vector,filename="thirdChant.wav",samplerate=44100)


peal = M.structures.symmetry.PlainChanges(3,1)
semi = 2**(1/12)
f0 = 220*2
notes = [f0, f0*semi**4, f0*semi**8]
sy = M.synths.CanonicalSynth()
note_vecs = []
for permutation in peal.peal_direct:
    notes_ = permutation(notes)
    print(notes_)
    for note in notes_:
        note_vector = sy.render(fundamental_frequency=note, duration=.2)
        note_vecs.append(note_vector)
peal_vec = n.hstack(note_vecs*4)
M.utils.write(peal_vec, filename='peal.wav', samplerate=44100)
