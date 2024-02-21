import numpy as n
import music as M

class CanonicalSynth:
    """Simple synth for sound synthesis with vibrato, tremolo and ADSR.
    
    All functions but absorbState returns a sonic array.
    You can parametrize the synth in any function call.
    If you want to keep some set of states for specific
    calls, clone your CanonicalSynth or create a new instance.
    You can also pass arbitrary variables to user latter on.
    
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
    """
    def __init__(s, **statevars):
        s.absorbState(**statevars)
        if "tables" not in dir(s):
            s.tables = M.tables.Basic()
        if "samplerate" not in dir(s):
            s.samplerate = 44100
        s.synthSetup()
        s.adsrSetup()

    def synthSetup(self,table=None,vibrato_table=None,tremolo_table=None,vibrato_depth=.1,vibrato_frequency=2.,tremolo_depth=3.,tremolo_frequency=0.2,duration=2,fundamental_frequency=220):
        """Setup synth engine. ADSR is configured seperately"""
        if not table:
            table=self.tables.triangle
        if vibrato_depth and vibrato_frequency:
            vibrato=True
            if not vibrato_table:
                vibrato_table=self.tables.sine
        else:
            vibrato=False
        if tremolo_depth and tremolo_frequency:
            tremolo=True
            if not tremolo_table:
                tremolo_table=self.tables.sine
        else:
            tremolo=False
        locals_=locals().copy(); del locals_["self"]
        for i in locals_:
            exec("self.{}={}".format(i,i))

    def adsrSetup(self,A=100.,D=40,S=-5.,R=50,render_note=False, adsr_method="absolute"):
        adsr_method=adsr_method # implement relative and False
        a_S=10**(S/20.)                       #
        Lambda_A=int(A*self.samplerate*0.001) #
        Lambda_D=int(D*self.samplerate*0.001) #
        Lambda_R=int(R*self.samplerate*0.001) #

        ii=n.arange(Lambda_A,dtype=n.float64)
        A_=ii/(Lambda_A-1)
        A_i=n.copy(A_) #
        ii=n.arange(Lambda_A,Lambda_D+Lambda_A,dtype=n.float64)
        D=1-(1-a_S)*(   ( ii-Lambda_A )/( Lambda_D-1) )
        D_i=n.copy(D) #
        #ii=n.arange(self.Lambda-self.Lambda_R,self.Lambda,dtype=n.float64)
        #R=self.a_S-self.a_S*((ii-(self.Lambda-self.Lambda_R))/(self.Lambda_R-1))
        R=a_S*(n.linspace(1,0,Lambda_R))
        R_i=n.copy(R) #
        locals_=locals().copy(); del locals_["self"]
        for i in locals_:
            exec("self.{}={}".format(i,i))
#        if render_note:
#            return self.render(d=.5)

    def adsrApply(self,audio_vec):
        Lambda=len(audio_vec)
        S=n.ones(Lambda-self.Lambda_R-(self.Lambda_A+
                   self.Lambda_D),dtype=n.float64)*self.a_S
        envelope=n.hstack((self.A_i, self.D_i, S, self.R_i))
        return envelope*audio_vec

    def render(self,**statevars):
        """Render a note with f0 Hertz and d seconds"""
        self.absorbState(**statevars)
        tremolo_envelope=self.tremoloEnvelope()
        note=self.rawRender()
        note=note*tremolo_envelope
        note=self.adsrApply(note)
        return note

    def tremoloEnvelope(self,sonic_vector=None,**statevars):
        self.absorbState(**statevars)
        if sonic_vector:
            Lambda=len(sonic_vector)
        else:
            Lambda=n.floor(self.samplerate*self.duration)
        ii=n.arange(Lambda)
        Lt=len(self.tremolo_table)
        Gammaa_i=n.floor(ii*self.tremolo_frequency*Lt/self.samplerate) # índices para a LUT
        Gammaa_i=n.array(Gammaa_i,n.int64)
        # variação da amplitude em cada amostra
        A_i=self.tremolo_table[Gammaa_i%Lt]
        A_i=10.**((self.tremolo_depth/20.)*A_i)
        if sonic_vector!=None:
            return A_i*sound
        else:
            return A_i

    def absorbState(s,**statevars):
        for varname in statevars:
            s.__dict__[varname]=statevars[varname]

    def rawRender(self,**statevars):
        self.absorbState(**statevars)
        Lambda=n.floor(self.samplerate*self.duration)
        ii=n.arange(Lambda)
        Lv=len(self.vibrato_table)

        Gammav_i=n.floor(ii*self.vibrato_frequency*Lv/self.samplerate) # índices para a LUT
        Gammav_i=n.array(Gammav_i,n.int64)
        # padrão de variação do vibrato para cada amostra
        Tv_i=self.vibrato_table[Gammav_i%Lv]

        # frequência em Hz em cada amostra
        F_i=self.fundamental_frequency*(   2.**(  Tv_i*self.vibrato_depth/12.  )   )
        # a movimentação na tabela por amostra
        Lt=len(self.table)
        D_gamma_i=F_i*(Lt/self.samplerate)
        Gamma_i=n.cumsum(D_gamma_i) # a movimentação na tabela total
        Gamma_i=n.floor( Gamma_i) # já os índices
        Gamma_i=n.array( Gamma_i, dtype=n.int64) # já os índices
        return self.table[Gamma_i%int(Lt)] # busca dos índices na tabela

    def render2(self,**statevars):
        self.absorbState(**statevars)
        Lambda=n.floor(self.samplerate*self.duration)
        ii=n.arange(Lambda)
        Lv=len(self.vibrato_table)

        Gammav_i=n.floor(ii*self.vibrato_frequency*Lv/self.samplerate) # índices para a LUT
        Gammav_i=n.array(Gammav_i,n.int64)
        # padrão de variação do vibrato para cada amostra
        Tv_i=self.vibrato_table[Gammav_i%Lv]

        # frequência em Hz em cada amostra
        F_i=self.fundamental_frequency*(   2.**(  Tv_i*self.vibrato_depth/12.  )   )
        # a movimentação na tabela por amostra
        Lt=self.tables.size
        D_gamma_i=F_i*(Lt/self.samplerate)
        Gamma_i=n.cumsum(D_gamma_i) # a movimentação na tabela total
        Gamma_i=n.floor( Gamma_i) # já os índices
        Gamma_i=n.array( Gamma_i, dtype=n.int64) # já os índices
        sound=self.table[Gamma_i%int(Lt)] # busca dos índices na tabela
        sound=self.adsrApply(sound)
        return sound
