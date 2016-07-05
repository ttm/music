import sys
keys=tuple(sys.modules.keys())
for key in keys:
    if "music" in key:
        del sys.modules[key]
import music as M
from percolation.rdf import c

# class to maintain state. Each call has an update method which loads new values from list.
# ex:
# M.synth.

# organize current CanonicalSynth into separate functions
# and write human-readable variable names for lookup process

# use CanonicalSynth to make other synths?
# E.g. ones in which settings change in each call.
class IteratorSynth(M.synths.CanonicalSynth):
    def renderIterate(self,**statevars):
        self.absorbState(**statevars)
        self.iterateElements()
        return self.render()
    def iterateElements(self):
        sequences=[var for var in dir(self) if var.endswith("_sequence")]
        state_vars=[i[:-9] for i in sequences]
        positions=[i+"_position" for i in sequences]
        for sequence,state_var,position in zip(sequences,state_vars,positions):
            if position not in dir(self):
                self.__dict__[position]=0
            self.__dict__[state_var]=self.__dict__[sequence][self.__dict__[position]]
            self.__dict__[position]+=1
            self.__dict__[position]%=len(self.__dict__[sequence])
isynth=IteratorSynth()
isynth.fundamental_frequency_sequence=[220,400,100,500]
sounds=[]
for i in range(300):
    sounds+=[isynth.renderIterate(duration=.2)]
# M.utils.write(M.H(*sounds),"./sandsounds/ra.wav")
M.utils.write(M.H(*sounds),"./ra.wav")
sys.exit()

semitom=2**(1/12)
fseq=[200*semitom**i for i in range(4)]
#fseq=[220,400,100,500]

def mirror(seq):
    return seq[::-1]
# def secondMirror(seq):

# mirrors
mi=mirror(fseq)
# mi2=secondMirror(fseq)
mi_mi2=mirror(mi2)
mi_full=fseq+mi+mi2+mi_mi2

# rotations
# ro=rotate(fseq)
# ro2=rotate(ro)
# ro3=rotate(ro2)
# ro_full=fseq+ro+ro2+ro3
#
# # swaps
# sp=swap(fseq,0,1)
# sp3=swap(fseq,0,3)
# sp4=swap(fseq,1,2)
# sp6=swap(fseq,2,3)
# sp_full=sp+sp3+sp4+sp6
#
# # swaps that are mirrors
# sp_mi=swap(fseq,0,2) # mirror
# sp_mi2=swap(fseq,1,3) # mirror
# mirrors_full=sp_mi+sp_mi2

isynth.fundamental_frequency_sequence=4*fseq+mi_full
sounds=[]
for i in range(300):
    sounds+=[isynth.renderIterate(duration=.2)]
M.utils.write(M.H(*sounds),"./sandsounds/ba.wav")

isynth.renderIterate()


