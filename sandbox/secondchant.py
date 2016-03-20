import sys
keys=tuple(sys.modules.keys())
for key in keys:
    if "music" in key:
        del sys.modules[key]
import music as M
from percolation.rdf import c
H = M.utils.H

durs  = 1,  2,   1,  2,   1, 2 
notes = 0,  2,   3,  2,   8,7,
text  = 'wake up  out of your sleep'
sonic_vector=M.singing.sing(text = text, notes=notes,
        durs=durs,Q=120,lang='en',M="3/4")

M.utils.write(sonic_vector,filename="some.wav",samplerate=44100)

sonic_vector=M.singing.sing(text = "wa-a-a-a-a-a", notes=notes,
        durs=durs,Q=120,lang='en',K="Cm")
M.utils.write(sonic_vector,filename="scale.wav",samplerate=44100)
