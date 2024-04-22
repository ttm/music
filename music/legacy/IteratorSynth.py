from .CanonicalSynth import CanonicalSynth


class IteratorSynth(CanonicalSynth):
    """
    A synthesizer that iterates through arbitrary lists of variables.

    Inherits from CanonicalSynth.

    Attributes:
        No additional attributes.

    Example:
    >>> isynth = M.IteratorSynth()
    >>> isynth.fundamental_frequency_sequence = [220, 400, 100, 500]
    >>> isynth.duration_sequence = [2, 1, 1.5]
    >>> isynth.vibrato_frequency_sequence = [3, 6.5, 10]
    >>> sounds = []
    >>> for i in range(300):
            sounds += [isynth.renderIterate(tremolo_frequency=.2*i)]
    >>> import music.core.io
    >>> music.core.io.write_wav_mono(M.H(*sounds),"./example.wav")
    """

    def renderIterate(self, **statevars):
        """
        Renders a sound iteration with the given state variables.

        Parameters:
            **statevars: Arbitrary keyword arguments for state variables.

        Returns:
            list: A list representing the rendered sound.
        """
        self.absorbState(**statevars)
        self.iterateElements()
        return self.render()

    def iterateElements(self):
        """
        Iterates through the sequences of state variables.
        """
        sequences = [var for var in dir(self) if var.endswith("_sequence")]
        state_vars = [i[:-9] for i in sequences]
        positions = [i + "_position" for i in sequences]
        for sequence, state_var, position in zip(sequences, state_vars,
                                                 positions):
            if position not in dir(self):
                self.__dict__[position] = 0
            self.__dict__[state_var] = \
                self.__dict__[sequence][self.__dict__[position]]
            self.__dict__[position] += 1
            self.__dict__[position] %= len(self.__dict__[sequence])
