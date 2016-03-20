import sys
keys=tuple(sys.modules.keys())
for key in keys:
    if "music" in key:
        del sys.modules[key]
import music as M
from percolation.rdf import c
H = M.utils.H

# sonic_vector=M.singing.sing(text = "pa-la-vras são muito le-gais")
# M.utils.write(sonic_vector,filename="achant.wav", normalize=False,samplerate=44100)

durs  = 1,  1,   1,  1,   1, 1,1, 1 
notes = 0,  0,   0,  0,   0, 0,0, 0
text  = 'pi-tchu-qui-nha, co-a-li-nha'
sonic_vector=M.singing.sing(text = text, notes=notes, durs=durs,Q=240)
sonic_vectorA=M.singing.sing(text = text, notes=notes,
        durs=durs,Q=240,reference=72)

notes = 0,  4,   7,  4,   12, 7,4, 0
sonic_vector_=M.singing.sing(text = text, notes=notes, durs=durs,Q=240)
notes = 0,  7,   4,  7,   12, 4,7, 0
sonic_vector_2=M.singing.sing(text = text, notes=notes, durs=durs,Q=240)

notes = 4,  5,   4,  7,   4, 12,7, 0
sonic_vectorA2=M.singing.sing(text = text, notes=notes,
        durs=durs,Q=240,reference=72)

notes = 0,  2,   4,  5,   7, 9,11, 12,\
        12,  11,   9,  7,   5, 4,2, 0
scale1 = M.singing.sing(text = text+" "+text, notes=notes,
        durs=durs*2,Q=240,reference=48)



sv = H (sonic_vector, sonic_vector, sonic_vector_, sonic_vector_2,
        sonic_vector+sonic_vectorA,sonic_vector+sonic_vectorA,
        sonic_vector_+sonic_vectorA2,sonic_vector_2+sonic_vectorA2,
        scale1
        )
M.utils.write(sv,filename="achant.wav",samplerate=44100)
M.utils.write(scale1,filename="scale.wav",samplerate=44100)
