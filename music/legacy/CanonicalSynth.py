import numpy as n
import music as M


class CanonicalSynth:
    """
    Simple synthesizer for sound synthesis with vibrato, tremolo, and ADSR.

    All functions but absorbState return a sonic array.
    You can parametrize the synth in any function call.
    If you want to keep some set of states for specific
    calls, clone your CanonicalSynth or create a new instance.
    You can also pass arbitrary variables to use later on.

    Parameters
    ----------
    f : scalar
        The frequency of the note in Hertz.
    d : scalar
        The duration of the note in seconds.
    fv : scalar
        The frequency of the vibrato oscillations in Hertz.
    nu : scalar
        The maximum deviation of pitch in the vibrato in semitones.
    tab : array_like
        The table with the waveform to synthesize the sound.
    tabv : array_like
        The table with the waveform of the vibrato oscillatory pattern.

    Examples
    --------
    >>> cs =  CanonicalSynth()
    # TODO: develop example
    """

    def __init__(s, **statevars):
        """
        Initializes the synthesizer with given state variables.

        Parameters
        ----------
        **statevars : dict
            Arbitrary keyword arguments for state variables.
        """
        s.absorbState(**statevars)
        if "tables" not in dir(s):
            s.tables = M.legacy.tables.Basic()
        if "samplerate" not in dir(s):
            s.samplerate = 44100
        s.synthSetup()
        s.adsrSetup()

    def synthSetup(self, table=None, vibrato_table=None, tremolo_table=None,
                   vibrato_depth=.1, vibrato_frequency=2., tremolo_depth=3.,
                   tremolo_frequency=0.2, duration=2,
                   fundamental_frequency=220):
        """
        Setup synth engine. ADSR is configured separately.

        Parameters
        ----------
        table : array_like, optional
            The waveform table for sound synthesis, by default None.
        vibrato_table : array_like, optional
            The waveform table for vibrato oscillatory pattern, by default
            None.
        tremolo_table : array_like, optional
            The waveform table for tremolo oscillatory pattern, by default
            None.
        vibrato_depth : float, optional
            The depth of vibrato in semitones, by default 0.1.
        vibrato_frequency : float, optional
            The frequency of the vibrato oscillations in Hertz, by default 2.0.
        tremolo_depth : float, optional
            The depth of tremolo in decibels, by default 3.0.
        tremolo_frequency : float, optional
            The frequency of the tremolo oscillations in Hertz, by default 0.2.
        duration : float, optional
            The duration of the note in seconds, by default 2.
        fundamental_frequency : float, optional
            The fundamental frequency of the note in Hertz, by default 220.
        """
        if not table:
            table = self.tables.triangle
        if vibrato_depth and vibrato_frequency:
            vibrato = True
            if not vibrato_table:
                vibrato_table = self.tables.sine
        else:
            vibrato = False
        if tremolo_depth and tremolo_frequency:
            tremolo = True
            if not tremolo_table:
                tremolo_table = self.tables.sine
        else:
            tremolo = False
        locals_ = locals().copy()
        del locals_["self"]
        for i in locals_:
            exec("self.{}={}".format(i, i))

    def adsrSetup(self, A=100., D=40, S=-5., R=50, render_note=False,
                  adsr_method="absolute"):
        """
        Setup ADSR parameters.

        Parameters
        ----------
        A : float, optional
            Attack time in milliseconds, by default 100.
        D : int, optional
            Decay time in milliseconds, by default 40.
        S : float, optional
            Sustain level in decibels, by default -5.
        R : int, optional
            Release time in milliseconds, by default 50.
        render_note : bool, optional
            Whether to render the note immediately, by default False.
        adsr_method : str, optional
            The ADSR method, by default "absolute".
        """
        adsr_method = adsr_method  # implement relative and False
        a_S = 10 ** (S / 20.)
        Lambda_A = int(A * self.samplerate * 0.001)
        Lambda_D = int(D * self.samplerate * 0.001)
        Lambda_R = int(R * self.samplerate * 0.001)

        ii = n.arange(Lambda_A, dtype=n.float64)
        A_ = ii / (Lambda_A - 1)
        A_i = n.copy(A_)
        ii = n.arange(Lambda_A, Lambda_D + Lambda_A, dtype=n.float64)
        D = 1 - (1 - a_S) * ((ii - Lambda_A) / (Lambda_D - 1))
        D_i = n.copy(D)
        R = a_S * n.linspace(1, 0, Lambda_R)
        R_i = n.copy(R)
        locals_ = locals().copy()
        del locals_["self"]
        for i in locals_:
            exec("self.{}={}".format(i, i))

    def adsrApply(self, audio_vec):
        """
        Apply ADSR envelope to the audio vector.

        Parameters
        ----------
        audio_vec : array_like
            Input audio vector.

        Returns
        -------
        array_like
            Audio vector with applied ADSR envelope.
        """
        Lambda = len(audio_vec)
        S = n.ones(Lambda - self.Lambda_R - (self.Lambda_A + self.Lambda_D),
                   dtype=n.float64) * self.a_S
        envelope = n.hstack((self.A_i, self.D_i, S, self.R_i))
        return envelope * audio_vec

    def render(self, **statevars):
        """
        Render a note with given parameters.

        Parameters
        ----------
        **statevars : dict
            Arbitrary keyword arguments for state variables.

        Returns
        -------
        array_like
            Rendered audio vector.
        """
        self.absorbState(**statevars)
        tremolo_envelope = self.tremoloEnvelope()
        note = self.rawRender()
        note = note * tremolo_envelope
        note = self.adsrApply(note)
        return note

    def tremoloEnvelope(self, sonic_vector=None, **statevars):
        """
        Calculate the tremolo envelope.

        Parameters
        ----------
        sonic_vector : array_like, optional
            Input sonic vector, by default None.
        **statevars : dict
            Arbitrary keyword arguments for state variables.

        Returns
        -------
        array_like
            Tremolo envelope.
        """
        self.absorbState(**statevars)
        if sonic_vector:
            Lambda = len(sonic_vector)
        else:
            Lambda = n.floor(self.samplerate * self.duration)
        ii = n.arange(Lambda)
        Lt = len(self.tremolo_table)
        Gammaa_i = n.floor(ii * self.tremolo_frequency * Lt /
                           self.samplerate)
        Gammaa_i = n.array(Gammaa_i, n.int64)
        A_i = self.tremolo_table[Gammaa_i % Lt]
        A_i = 10. ** ((self.tremolo_depth / 20.) * A_i)
        if sonic_vector is not None:
            return A_i * sonic_vector
        else:
            return A_i

    def absorbState(s, **statevars):
        """
        Absorb state variables.

        Parameters
        ----------
        **statevars : dict
            Arbitrary keyword arguments for state variables.
        """
        for varname in statevars:
            s.__dict__[varname] = statevars[varname]

    def rawRender(self, **statevars):
        """
        Render the sound without applying ADSR.

        Parameters
        ----------
        **statevars : dict
            Arbitrary keyword arguments for state variables.

        Returns
        -------
        array_like
            Rendered audio vector.
        """
        self.absorbState(**statevars)
        Lambda = n.floor(self.samplerate * self.duration)
        ii = n.arange(Lambda)
        Lv = len(self.vibrato_table)
        Gammav_i = n.floor(ii * self.vibrato_frequency * Lv /
                           self.samplerate)
        Gammav_i = n.array(Gammav_i, n.int64)
        Tv_i = self.vibrato_table[Gammav_i % Lv]
        F_i = self.fundamental_frequency * (2. **
                                            (Tv_i * self.vibrato_depth / 12.))
        Lt = len(self.table)
        D_gamma_i = F_i * (Lt / self.samplerate)
        Gamma_i = n.cumsum(D_gamma_i)
        Gamma_i = n.floor(Gamma_i)
        Gamma_i = n.array(Gamma_i, dtype=n.int64)
        return self.table[Gamma_i % int(Lt)]

    def render2(self, **statevars):
        """
        Render the sound and apply ADSR.

        Parameters
        ----------
        **statevars : dict
            Arbitrary keyword arguments for state variables.

        Returns
        -------
        array_like
            Rendered audio vector with applied ADSR envelope.
        """
        self.absorbState(**statevars)
        Lambda = n.floor(self.samplerate * self.duration)
        ii = n.arange(Lambda)
        Lv = len(self.vibrato_table)
        Gammav_i = n.floor(ii * self.vibrato_frequency * Lv /
                           self.samplerate)
        Gammav_i = n.array(Gammav_i, n.int64)
        Tv_i = self.vibrato_table[Gammav_i % Lv]
        F_i = self.fundamental_frequency * (2. **
                                            (Tv_i * self.vibrato_depth / 12.))
        Lt = self.tables.size
        D_gamma_i = F_i * (Lt / self.samplerate)
        Gamma_i = n.cumsum(D_gamma_i)
        Gamma_i = n.floor(Gamma_i)
        Gamma_i = n.array(Gamma_i, dtype=n.int64)
        sound = self.table[Gamma_i % int(Lt)]
        sound = self.adsrApply(sound)
        return sound
