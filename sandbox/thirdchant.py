import sys
keys=tuple(sys.modules.keys())
for key in keys:
    if "music" in key:
        del sys.modules[key]
import music as M
from percolation.rdf import c
H = M.utils.H

durs  = 1,  1, 1, 1      ,1,1, 1, 1
notes = 0,  2,   4,  5,   7,9, 11, 12
text  = 'sin-ging my ass out for you yes'
sonic_vector=M.singing.sing(text = text, notes=notes,reference=60,
        durs=durs,Q=120,lang='en',M="4/4",transpose=0)
sonic_vector_=M.singing.sing(text = text, notes=notes[::-1],reference=60,
        durs=durs,Q=120,lang='en',M="4/4",transpose=0)
M.utils.write(H(sonic_vector,sonic_vector_)
    ,filename="some.wav",samplerate=44100)

durs  = 1,  1, 1, 1      ,1,1, 1, 1
notes = 0,  2,   3,  5,   7,9, 11, 12
notes_ = 12, 10, 8, 7, 5, 3, 2, 0
text  = 'sin-ging my ass out for you yes'
sonic_vector=M.singing.sing(text = text, notes=notes,reference=60,
        durs=durs,Q=120,lang='en',M="4/4",transpose=0)
sonic_vector_=M.singing.sing(text = text, notes=notes_,reference=60,
        durs=durs,Q=120,lang='en',M="4/4",transpose=0)
M.utils.write(H(sonic_vector,sonic_vector_)
    ,filename="some_.wav",samplerate=44100)

sonic_vector=M.singing.sing(text = text, notes=notes,reference=60,
        durs=durs,Q=120,lang='en',M="4/4",transpose=0)
sonic_vector_=M.singing.sing(text = text, notes=notes_,reference=60,
        durs=durs,Q=120,lang='en',M="4/4",transpose=0)
M.utils.write(H(sonic_vector,sonic_vector_)
    ,filename="some__.wav",samplerate=44100)

