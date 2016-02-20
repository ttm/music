import numpy as n
import music as M
#from . import tables
class CanonicSynth:
    def __init__(self,samplerate=44100,tables=None):
        self.samplerate=samplerate
        if tables:
            self.tables=tables
        else:
            self.tables=M.tables.Basic()
        self.synthSetup()
        self.adsrSetup()
    def synthSetup(self,table=None,vibrato_table=None,tremolo_table=None,vibrato_depth=.1,vibrato_frequency=2.,tremolo_depth=3.,tremolo_frequency=0.2):
        """Setup synth engine. ADSR is configured seperately"""
        if not table:
            table=self.tables.sine
        if vibrato_depth and vibrato_freqquency:
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
        for i in locals_:
            exec("self.{}={}".format(i,i))
    def adsrSetup(self,A=100.,D=40,S=-5.,R=50,render_note=False, adsr_method="absolute"):
        adsr_method=adsr_method # implement relative and False
        a_S=10**(S/20.)                       #
        Lambda_A=int(A*self.samplerate*0.001) #
        Lambda_D=int(D*self.samplerate*0.001) #
        Lambda_R=int(R*self.samplerate*0.001) #

        ii=n.arange(Lambda_A,dtype=n.float)
        A_=ii/(Lambda_A-1)
        A_i=n.copy(A_) #
        ii=n.arange(Lambda_A,Lambda_D+Lambda_A,dtype=n.float)
        D=1-(1-a_S)*(   ( ii-Lambda_A )/( Lambda_D-1) )
        D_i=n.copy(D) # 
        #ii=n.arange(self.Lambda-self.Lambda_R,self.Lambda,dtype=n.float)
        #R=self.a_S-self.a_S*((ii-(self.Lambda-self.Lambda_R))/(self.Lambda_R-1))
        R=a_S*(n.linspace(1,0,Lambda_R))
        R_i=n.copy(R) #
        for i in locals_:
            exec("self.{}={}".format(i,i))
#        if render_note:
#            return self.render(d=.5)

    def adsrApply(self,audio_vec):
        Lambda=len(audio_vec)
        S=n.ones(Lambda-self.Lambda_R-(self.Lambda_A+
                   self.Lambda_D),dtype=n.float)*self.a_S
        envelope=n.hstack((self.A_i, self.D_i, S, self.R_i))
        return envelope*audio_vec

    def render(self,fundamental_frequency=220.,duration=2.):
        """Render a note with f0 Hertz and d seconds"""
        tremolo_envelope=tremoloEnvelope(duration)
        note=self.rawRender(fundamental_frequency,duration)
        note=note*tremolo_envelope
        note=self.adsrApply(note)
        return self.note

    def tremoloEnvelope(self,duration=2,tremolo_frequency=None,tremolo_depth=None,tremolo_table=None,sonic_vector=None):
        if sinc_vector:
            Lambda=len(sound)
        else:
            Lambda=n.floor(self.samplerate*d)
        ii=n.arange(Lambda)
        Lt=len(tremolo_table)
        Gammaa_i=n.floor(ii*tremolo_frequency*Lt/self.samplerate) # índices para a LUT
        Gammaa_i=n.array(Gammaa_i,n.int)
        # variação da amplitude em cada amostra
        A_i=tremolo_table[Gammaa_i%Lt] 
        A_i=10.**((tremolo_depth/20.)*A_i)
        if sound!=None:
            return A_i*sound
        else:
            return A_i

    def rawRender(self,fundamental_frequency=None,duration=None,table=None,vibrato_frequency=None,vibrato_depth=None,vibrato_table=None):
        Lambda=n.floor(self.samplerate*duration)
        ii=n.arange(Lambda)
        Lv=len(vibrato_table)

        Gammav_i=n.floor(ii*vibrato_frequency*Lv/self.samplerate) # índices para a LUT
        Gammav_i=n.array(Gammav_i,n.int)
        # padrão de variação do vibrato para cada amostra
        Tv_i=tabv[Gammav_i%Lv] 

        # frequência em Hz em cada amostra
        F_i=fundamental_frequency*(   2.**(  Tv_i*nu/12.  )   ) 
        # a movimentação na tabela por amostra
        Lt=len(table)
        D_gamma_i=F_i*(Lt/self.samplerate)
        Gamma_i=n.cumsum(D_gamma_i) # a movimentação na tabela total
        Gamma_i=n.floor( Gamma_i) # já os índices
        Gamma_i=n.array( Gamma_i, dtype=n.int) # já os índices
        return table[Gamma_i%int(Lt)] # busca dos índices na tabela

    def render2(self,f=200,d=2.,tab=None,fv=2.,nu=2.,tabv=None):
        Lambda=n.floor(self.samplerate*d)
        ii=n.arange(Lambda)
        Lv=self.tam_vib_tab

        Gammav_i=n.floor(ii*fv*Lv/self.samplerate) # índices para a LUT
        Gammav_i=n.array(Gammav_i,n.int)
        # padrão de variação do vibrato para cada amostra
        Tv_i=tabv[Gammav_i%Lv] 

        # frequência em Hz em cada amostra
        F_i=f*(   2.**(  Tv_i*nu/12.  )   ) 
        # a movimentação na tabela por amostra
        Lt=self.tables.size
        D_gamma_i=F_i*(Lt/self.samplerate)
        Gamma_i=n.cumsum(D_gamma_i) # a movimentação na tabela total
        Gamma_i=n.floor( Gamma_i) # já os índices
        Gamma_i=n.array( Gamma_i, dtype=n.int) # já os índices
        sound=tab[Gamma_i%int(Lt)] # busca dos índices na tabela
        sound=self.adsrApply(sound)
        return sound
