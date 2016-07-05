import sys
# keys=tuple(sys.modules.keys())
# for key in keys:
#     if "music" in key:
#         del sys.modules[key]
import music as M
# import numpy as n
from percolation.rdf import c

peal = M.structures.symmetry.PlainChanges(3, 1)
semi = 2**(1/12)
f0 = 110*2
notes = [f0, f0*semi**4, f0*semi**8]


class IteratorSynth(M.synths.CanonicalSynth):
    def renderIterate(self, **statevars):
        self.absorbState(**statevars)
        self.iterateElements()
        return self.render()

    def iterateElements(self):
        sequences = [var for var in dir(self) if var.endswith("_sequence")]
        state_vars = [i[:-9] for i in sequences]
        positions = [i+"_position" for i in sequences]
        for sequence, state_var, position in zip(sequences, state_vars, positions):
            if position not in dir(self):
                self.__dict__[position] = 0
            self.__dict__[state_var] = self.__dict__[sequence][self.__dict__[position]]
            self.__dict__[position] += 1
            self.__dict__[position] %= len(self.__dict__[sequence])
isynth = IteratorSynth()
isynth.fundamental_frequency_sequence = []
for perm in peal.peal_direct:
    isynth.fundamental_frequency_sequence.extend(perm(notes))
sounds = []
for i in range(30):
    sounds += [isynth.renderIterate(duration=.2)]
# M.utils.write(M.H(*sounds),"./sandsounds/ra.wav")
c('finished rendering peal')
M.utils.write(M.H(*sounds), "./apeal.wav")
c('finished writing peal')
peal = M.structures.symmetry.PlainChanges(4, 2)
notes = [f0, f0*semi**3, f0*semi**6, f0*semi**9]
isynth = IteratorSynth()
isynth.fundamental_frequency_sequence = []
for perm in peal.peal_direct:
    isynth.fundamental_frequency_sequence.extend(perm(notes))
sounds = []
for i in range(len(isynth.fundamental_frequency_sequence)):
    sounds += [isynth.renderIterate(duration=.2)]
c('finished rendering peal')
M.utils.write(M.H(*sounds), "./apeal4.wav")
c('finished writing peal')
sys.exit()
