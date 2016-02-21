import music as M, numpy as n

synth=M.synths.CanonicalSynth()

class TestSong2:
    def __init__(self):
        global M

        note=synth.render()
        M.utils.write(note) # saved to fooname.wav

        note=synth.rawRender()
        note2=synth.rawRender(duration=4.)
        # test: rawRender + applyAdsr
        note3=synth.rawRender(duration=4.,vibrato_frequency=4.)
        note4=synth.rawRender(duration=4.,vibrato_frequency=4.,vibrato_depth=8.)
        note5=synth.rawRender(duration=4.,vibrato_frequency=1.,vibrato_depth=8.)
        note6=synth.rawRender(duration=4.,vibrato_frequency=0.5,vibrato_depth=8.)
        notes=[note,note2,note3,note4,note5,note6]
        notes=[synth.adsrApply(i) for i in notes]
        vibratos=n.hstack(notes)
        M.utils.write(vibratos,"vibratos.wav") # saved to fooname.wav
        # tremoloEnvelope
        note6=synth.rawRender(duration=4.,vibrato_frequency=0.)
        note7=synth.rawRender(fundamental_frequency=440.,duration=4.,vibrato_frequency=0.)
        te=synth.tremoloEnvelope(duration=4.)
        te2=synth.tremoloEnvelope(duration=4.)
        te3=synth.tremoloEnvelope(tremolo_depth=20,duration=4.)
        te4=synth.tremoloEnvelope(duration=4.,tremolo_table=synth.tables.triangle)

        notes=[note6*te,note6*te2,note6*te3,note6*te4]
        notes=[synth.adsrApply(i) for i in notes]
        tremolos=n.hstack(notes)
        M.utils.write(tremolos,"tremolos.wav") # saved to fooname.wav

        # tremolog + envelope
        R=synth.rawRender
        T=synth.tremoloEnvelope
        A=synth.adsrApply
        notes=[T(sounduration=R(duration=4.)), # == T(duration=4.)*R(duration=4.)
        T(duration=4.)*R(duration=4.), # sould sound the same
        T(tre_freq=4.,duration=4.)*R(duration=4.),
        T(tre_freq=2.,duration=4.)*R(duration=4.,vibrato_frequency=4.),
        T(tre_freq=4.,duration=4.)*R(duration=4.,vibrato_frequency=4.),
        T(tre_freq=8.,duration=4.)*R(duration=4.,vibrato_frequency=4.),
        T(tre_freq=4.,duration=4.)*R(duration=4.,vibrato_frequency=8.)]
        notes=[synth.adsrApply(i) for i in notes]
        tremolos=n.hstack(notes)
        M.utils.write(tremolos,"TV.wav") # saved to fooname.wav

        f0=220.
        M_=M.utils.midi2HzInterval
        H=n.hstack
        R=synth.render2
        notes_=[T(tremolo_depth=2.,duration=4.)*R(fundamental_frequency=f0*M_(7),duration=4.,vibrato_frequency=4.)+
        T(tremolo_depth=2.,duration=4.)*R(        fundamental_frequency=f0,duration=4.,vibrato_frequency=4.),

        T(tremolo_depth=4.,duration=4.)*H(( R(fundamental_frequency=f0*M_(7),duration=2.,vibrato_frequency=4.), R(fundamental_frequency=f0*M_(7),duration=2.,vibrato_frequency=4.) )) +
        T(tremolo_depth=4.,duration=4.)*R(    fundamental_frequency=f0,duration=4.,     vibrato_frequency=4.),

        T(tremolo_depth=2.,duration=4.)*R(    fundamental_frequency=f0*M_(7),duration=4.,vibrato_frequency=4.)+
        T(tremolo_depth=2.,duration=4.)*R(    fundamental_frequency=f0,duration=4.,vibrato_frequency=4.),

        T(tremolo_depth=4.,duration=4.)*R(    fundamental_frequency=f0*M_(7),duration=4.,vibrato_frequency=2.)+
        T(tremolo_depth=4.,duration=4.)*R(    fundamental_frequency=f0,duration=4.,     vibrato_frequency=2.),

        T(tremolo_depth=6.,duration=4.)*R(    fundamental_frequency=f0*M_(7),duration=4.,vibrato_frequency=4.)+
        T(tremolo_depth=8.,duration=4.)*R(    fundamental_frequency=f0,duration=4.,     vibrato_frequency=2.),

        T(tremolo_depth=6.,duration=4.)*R(    fundamental_frequency=f0*M_(-7),duration=4.,vibrato_frequency=4.)+
        T(tremolo_depth=8.,duration=4.)*H(( R(fundamental_frequency=f0,duration=2.,     vibrato_frequency=2.),        R(fundamental_frequency=f0,duration=2.,     vibrato_frequency=4.) )),

        T(tremolo_depth=6.,duration=4.)*R(    fundamental_frequency=f0*M_(-7),duration=4.,vibrato_frequency=8.)+       
        T(tremolo_depth=8.,duration=4.)*H(( R(fundamental_frequency=f0,duration=2.,     vibrato_frequency=2.),        R(fundamental_frequency=f0,duration=2.,     vibrato_frequency=8.) )),

        T(tremolo_depth=8.,duration=4.)*R(    fundamental_frequency=f0*M_(-7),duration=4.,vibrato_frequency=8.)+       
        T(tremolo_depth=8.,duration=4.)*H(( R(fundamental_frequency=f0,duration=2.,     vibrato_frequency=2.),        R(fundamental_frequency=f0,duration=2.,     vibrato_frequency=6.) )),

        T(tremolo_depth=.5,duration=4.)*R(    fundamental_frequency=f0*M_(-7),duration=4.,vibrato_frequency=1.)+       
        T(tremolo_depth=.5,duration=4.)*H(( R(fundamental_frequency=f0,duration=2.,     vibrato_frequency=2.),        R(fundamental_frequency=f0,duration=2.,     vibrato_frequency=6.) )),

        T(tremolo_depth=1.,duration=4.)*R(    fundamental_frequency=f0*M_(-5),duration=4.,vibrato_frequency=1.)+       
        T(tremolo_depth=.5,duration=4.)*H(( R(fundamental_frequency=f0,duration=2.,     vibrato_frequency=2.),        R(fundamental_frequency=f0*M_(2),duration=2.,     vibrato_frequency=6.) )),

        T(tremolo_depth=2.,duration=4.)*R(    fundamental_frequency=f0*M_(-5),duration=4.,vibrato_frequency=1.)+       
        T(tremolo_depth=1.,duration=4.)*H(( R(fundamental_frequency=f0,duration=2.,     vibrato_frequency=2.),        R(fundamental_frequency=f0*M_(2.2),duration=2.,     vibrato_frequency=8.) )),

        T(tremolo_depth=2.,duration=4.)*R(    fundamental_frequency=f0*M_(5),duration=4.,vibrato_frequency=2.)+
        T(tremolo_depth=2.,duration=4.)*H(( R(fundamental_frequency=f0,duration=2.,     vibrato_frequency=2.),        R(fundamental_frequency=f0*M_(0.2),duration=2.,     vibrato_frequency=8.) )),

        T(tremolo_depth=2.,duration=4.)*R(    fundamental_frequency=f0*M_(5.2),duration=4.,vibrato_frequency=2.)+
        T(tremolo_depth=2.,duration=4.)*H(( R(fundamental_frequency=f0,duration=2.,     vibrato_frequency=2.),        R(fundamental_frequency=f0*M_(-.2),duration=2.,     vibrato_frequency=8.) )),

        T(tremolo_depth=2.,duration=4.)*R(    fundamental_frequency=f0*M_(7.2),duration=4.,vibrato_frequency=2.)+
        T(tremolo_depth=2.,duration=4.)*H(( R(fundamental_frequency=f0*M_(-1),duration=2.,     vibrato_frequency=2.),  R(fundamental_frequency=f0*M_(-.2),duration=2.,     vibrato_frequency=8.) )),

        T(tremolo_depth=2.,duration=4.)*R(    fundamental_frequency=f0*M_(12.),duration=4.,vibrato_frequency=2.)+
        T(tremolo_depth=2.,duration=4.)*H(( R(fundamental_frequency=f0*M_(0),duration=2.,     vibrato_frequency=2.),   R(fundamental_frequency=f0*M_(4.2),duration=2.,     vibrato_frequency=6.) )),

        T(tremolo_depth=2.,duration=4.)*R(    fundamental_frequency=f0*M_(12.),duration=4.,vibrato_frequency=2.)+
        T(tremolo_depth=2.,duration=4.)*H(( R(fundamental_frequency=f0*M_(-12),duration=2.,     vibrato_frequency=2.), R(fundamental_frequency=f0,duration=2.,     vibrato_frequency=4.) )),

        T(tremolo_depth=2.,duration=4.)*R(    fundamental_frequency=f0*M_(7.),duration=4.,vibrato_frequency=4.)+
        T(tremolo_depth=2.,duration=4.)*H(( R(fundamental_frequency=f0*M_(-12),duration=2.,     vibrato_frequency=2.), R(fundamental_frequency=f0,duration=2.,     vibrato_frequency=4.) )),
        ]

        #notes_=[synth.adsrApply(i) for i in notes_]
        #
        #
        #notes=[T(sounduration=R(duration=4.)), # == T(duration=4.)*R(duration=4.)
        #T(duration=4.)*R(duration=4.), # sould sound the same
        #T(tre_freq=4.,duration=4.)*R(duration=4.),
        #T(tre_freq=2.,duration=4.)*R(duration=4.,vibrato_frequency=4.),
        #T(tre_freq=4.,duration=4.)*R(duration=4.,vibrato_frequency=4.),
        #T(tre_freq=8.,duration=4.)*R(duration=4.,vibrato_frequency=4.),
        #T(tre_freq=4.,duration=4.)*R(duration=4.,vibrato_frequency=8.)][::-1]
        #
        #notes=[synth.adsrApply(i) for i in notes]
        #
        #vibrosong=H(notes_+notes)
        locals_=locals().copy(); del locals_["self"]
        for i in locals_:
            exec("self.{}={}".format(i,i))
    def render(self):
        A=synth.adsrApply
        H=n.hstack
        synth.adsrSetup(0,0,0,3000)
        vibrosong=A(H(self.notes_+[n.zeros(44100)]))
        M.utils.write(vibrosong,"vibrosong.wav")



