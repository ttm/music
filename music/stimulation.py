import config
import synths
import tables

class ContinuousComponent(CanonicalSynth):
    def __init__(self, **statevars):
        if statevars.template in ('e', 'extreme', 'extreme-fidelity'):
            statevars.tables = tables.Basic(2 ** 16)
        CanonicalSynth.__init__(self, **statevars)

    def mix(self, sound):
        ''' mix self with another sound.

        add sample-by sample resolving compatibiliies of:
        1) length of the sound;
        2) mono-stereo;
        3) numeric type (integer/binary, floating-point);
        4) sample-rate.
        '''
        self.operations.append(f'mixed: {sound.description()}')
        self.samples += sound.samples
        return self
    
    def description(self):
        return ', '.join(self.operations)
