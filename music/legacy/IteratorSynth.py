from .CanonicalSynth import CanonicalSynth


class IteratorSynth(CanonicalSynth):
    """A synth that iterates through arbitrary lists of variables


    Any variable used by the CanonicalSynth can be used.
    Just append the variable name with the token _sequence.

    Example:
    >>> isynth=M.IteratorSynth()
    >>> isynth.fundamental_frequency_sequence = [220, 400, 100, 500]
    >>> isynth.duration_sequence = [2, 1, 1.5]
    >>> isynth.vibrato_frequency_sequence = [3, 6.5, 10]
    >>> sounds=[]
    >>> for i in range(300):
            sounds += [isynth.renderIterate(tremolo_frequency=.2*i)]
    >>> M.utils.write(M.H(*sounds),"./example.wav")

    """

    def renderIterate(self,**statevars):
        self.absorbState(**statevars)
        self.iterateElements()
        return self.render()

#     def render(self):
#         sequences = [var for var in dir(self) if var.endswith("_sequence")]
#         lens = [len(self.__dict__[seq]) for seq in sequences]
#         iterations = max(lens)
#         sonic_vector = [self.renderIterate() for i in range(iterations)]
#         return sonic_vector

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
