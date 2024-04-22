import numpy as n

from music.legacy import tables
from music.utils import horizontal_stack
from music.core.io import write_wav_mono
from music.core.filters.adsr import adsr
from music.core.synths.notes import note_with_vibrato
from music.utils import WAVEFORM_TRIANGULAR, WAVEFORM_SINE

T = tables.Basic()
n_ = n


def V_(st=0, freq=220, duration=2., vibrato_freq=2., max_pitch_dev=2.,
       waveform_table=WAVEFORM_TRIANGULAR,
       vibrato_waveform_table=WAVEFORM_SINE):
    """A shorthand for generating a note with vibrato.

    Args:
        st (float): Semitones relative to the base frequency.
        freq (float): Base frequency of the note.
        duration (float): Duration of the note in seconds.
        vibrato_freq (float): Frequency of the vibrato.
        max_pitch_dev (float): Maximum pitch deviation.
        waveform_table (array): Table representing the waveform of the note.
        vibrato_waveform_table (array): Table representing the waveform of the
                                        vibrato.

    Returns:
        array: A note with vibrato.
    """
    f_ = freq * 2 ** (st / 12)
    return note_with_vibrato(freq=f_, duration=2., vibrato_freq=2.,
                             max_pitch_dev=2.,
                             waveform_table=WAVEFORM_TRIANGULAR,
                             vibrato_waveform_table=WAVEFORM_SINE)


def ADV(note_dict={}, adsr_dict={}):
    """Apply ADSR envelope to a note with vibrato.

    Args:
        note_dict (dict): Dictionary containing parameters for the note.
        adsr_dict (dict): Dictionary containing parameters for the ADSR
                          envelope.

    Returns:
        array: Note with applied ADSR envelope.
    """
    return adsr(sonic_vector=V_(**note_dict), **adsr_dict)


class Being:
    def __init__(self):
        rhythm = [1.]  # repetition of one second
        rhythm2 = [1/2, 1/2]  # repetition of one second
        rhythm3 = [1/3, 1/3, 1/3]  # repetition of one second
        rhythm4 = [1/4, 1/4, 1/3]  # repetition of one second

        # assume duration = 1 (be 1 second, minute or whatnot):
        rhythmic_spectrum = [[1. / i] * i for i in range(1, 300)]

        # pitch or frequency sequences (to be used at will)
        f = 110
        freqs = [220]
        freq_spectrum = [i*f for i in range(1, 300)]
        neg_spec = [f/i for i in range(2, 300)]

        freq_sym = [[f*2**((i*j)/12) for i in range(j)] for j in [2, 3, 4, 6]]
        freq_sym_ = [[f*2**((i*j)/12) for i in range(300)]
                     for j in [2, 3, 4, 6]]

        dia = [2, 2, 1, 2, 2, 2, 1]
        notes_diatonic = [[dia[(j+i) % 7] for i in range(7)] for j in range(7)]
        notes_diatonic_ = [sum(notes_diatonic[i]) for i in range(7)]
        freq_diatonic = [[f*2**((12 * i + notes_diatonic_[j])/12)
                          for i in range(30)] for j in range(7)]

        intensity_octaves = [[10**((i*10)/(j*20)) for i in range(300)]
                             for j in range(1, 20)]  # steps of 10db - 1/2 dB
        db0 = 10**(-120/20)
        intensity_spec = [[db0*i for i in j] for j in intensity_octaves]

        # diatonic noise, noises derived from the symmetric scales etc: one
        # sinusoid or other basic waveform in each note.
        # Synth on the freq domain to optimize and simplify the process

        # make music of the spheres using ellipses and relations recalling
        # gravity
        self.resources = locals()
        self.startBeing()

    def walk(self, n, method='straight'):
        """Walk a certain number of steps.

        Args:
            n (int): Number of steps.
            method (str): Method of walking.

        Returns:
            array: Sequence of steps.
        """
        if method == 'straight':
            # ** TTM
            sequence = [self.grid[self.pointer + i] for i in range(n)]
            self.pointer += n
        elif method == 'low-high':
            sequence = [self.grid[self.pointer + i % (self.seqsize + 1) + i //
                                  self.seqsize] for i in range(n*self.seqsize)]
        elif method == 'perm-walk':
            # restore walk from 02peal
            pass
        self.addSeq(sequence)

    def setPar(self, par='f'):
        """Set parameter to be developed in walks and stays.

        Args:
            par (str): Parameter to be set.

        Returns:
            None
        """
        if par == 'f':
            self.grid = self.fgrid
            self.pointer = self.fpointer

    def setSize(self, ss):
        """Set the size.

        Args:
            ss (int): Size to set.

        Returns:
            None
        """
        self.seqsize = ss

    def setPerms(self, perms):
        """Set permutations.

        Args:
            perms (list): List of permutations.

        Returns:
            None
        """
        self.perms = perms

    def stay(self, n, method='perm'):
        """Stay for a certain number of notes.

        Args:
            n (int): Number of notes.
            method (str): Method of staying.

        Returns:
            array: Sequence of stayed notes.
        """
        if method == 'straight':
            sequence = [self.grid[(self.pointer + i) % self.seqsize]
                        for i in range(n)]
        elif method == 'perm':
            # ** TTM
            sequence = []
            if not isinstance(self.domain, n_.ndarray):
                if not self.domain:
                    domain = self.grid[self.pointer: self.pointer +
                                       self.seqsize]
                else:
                    domain = n_.array(self.domain)
                    print("Implemented OK?? TTM")
            else:
                domain = self.domain
            # nel = self.perms[0].size  # should match self.seqsize ?
            count = 0
            while len(sequence) < n:
                perm = self.perms[count % len(self.perms)]
                seq = perm(domain)
                sequence.extend(seq)
                count += 1
            sequence = sequence[:n]
        self.addSeq(sequence)
        self.total_notes += n

    def addSeq(self, sequence):
        """Add sequence to the Being.

        Args:
            sequence (array): Sequence to add.

        Returns:
            None
        """
        if isinstance(self.__dict__[self.curseq], list):
            self.__dict__[self.curseq].extend(sequence)
        else:
            self.__dict__[self.curseq] = \
                horizontal_stack(self.__dict__[self.curseq], sequence)

    def render(self, nn, fn=False):
        """Render notes of the Being.

        Args:
            nn (int): Number of notes to render.
            fn (str or bool): File name to save the rendered notes.

        Returns:
            array or None: Rendered notes.
        """
        self.mkArray()
        ii = n.arange(nn)
        duration = self.d_[ii % len(self.d_)]*self.dscale
        freq = self.f_[ii % len(self.f_)]
        waveform_table = self.tab_[ii % len(self.tab_)]
        vibrato_freq = self.fv_[ii % len(self.fv_)]
        max_pitch_dev = self.nu_[ii % len(self.nu_)]
        A = self.A_[ii % len(self.A_)]
        D = self.D_[ii % len(self.D_)]
        S = self.S_[ii % len(self.S_)]
        R = self.R_[ii % len(self.R_)]
        notes = [ADV({'freq': ff, 'duration': dd, 'vibrato_freq': fvv,
                      'max_pitch_dev': nuu, 'waveform_table': tabb},
                     {'attack_duration': AA, 'decay_duration': DD,
                      'sustain_level': SS, 'release_duration': RR})
                 for ff, dd, fvv, nuu, tabb, AA, DD, SS, RR
                 in zip(freq, duration, vibrato_freq, max_pitch_dev,
                        waveform_table, A, D, S, R)]
        if fn:
            if not isinstance(fn, str):
                fn = 'abeing.wav'
            if fn[-4:] != '.wav':
                fn += '.wav'
            write_wav_mono(horizontal_stack(*notes), fn)
        else:
            return horizontal_stack(*notes)

    def startBeing(self):
        """Start the Being.

        Args:
            None

        Returns:
            None
        """
        self.dscale = 1
        self.d_ = [1]
        self.f_ = [220]
        self.fv_ = [3]
        self.nu_ = [1]
        self.tab_ = [T.triangle]
        self.A_ = [20]
        self.D_ = [20]
        self.S_ = [-5]
        self.R_ = [50]
        self.mkArray()
        self.total_notes = 0

    def mkArray(self):
        """Make array.

        Args:
            None

        Returns:
            None
        """
        self.d_ = n.array(self.d_)
        self.f_ = n.array(self.f_)
        self.fv_ = n.array(self.fv_)
        self.nu_ = n.array(self.nu_)
        self.tab_ = n.array(self.tab_)
        self.A_ = n.array(self.A_)
        self.D_ = n.array(self.D_)
        self.S_ = n.array(self.S_)
        self.R_ = n.array(self.R_)

    def howl(self):
        """Produce a howling sound.

        Args:
            None

        Returns:
            None
        """
        pass

    def freeze(self):
        """Freeze the Being.

        Args:
            None

        Returns:
            None
        """
        pass
