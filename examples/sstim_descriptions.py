"""Write stimuli as SSTIM descriptions, and render descriptions back.

SSTIM, the Sensory Stimulation Vocabulary (https://w3id.org/sstim),
describes a stimulus so that any tool can read it: its signals, its
channels, and how each channel renders each signal. This writes a stimulus
and a session as descriptions, renders them back, and renders a
description written by hand, as another tool would write it, in SSTIM's
terms alone.

Needs ``pip install 'music[sstim]'``. Checking a description against
SSTIM's own validator, ``validate_sstim(graph)``, also needs the network
the first time, and is left out here.
"""

from datetime import datetime, timezone

import numpy as np

import music
from music.stimulation.sstim import (
    SessionPlan, from_sstim_graph, inspect_sstim, render_sstim,
    to_sstim_graph,
)

BASE = "https://example.org/music/examples/"

# A stimulus is described as the call that renders it, every default
# written out, and renders back sample for sample.
beat = to_sstim_graph(music.binaural_beats, carrier_freq=220, beat_freq=7,
                      duration=8, base=BASE + "beat/")
beat.serialize("stim_binaural.ttl", format="turtle")
assert np.array_equal(
    render_sstim("stim_binaural.ttl"),
    music.binaural_beats(carrier_freq=220, beat_freq=7, duration=8))

# A session is a program of its phases, each an SSTIM stimulus; with a
# plan, it is also SSTIM's own SessionSpecification, which needs a whole
# number of seconds, a preset, a creation time and a master volume.
session = music.StimulationSession(sample_rate=8000, end_ramp=2.0)
session.add(music.binaural_beats, duration=40, ramp=1.0,
            carrier_freq=220, beat_freq=7, label="settle")
session.add(music.modulated_noise, duration=20, ramp=2.0, seed=1,
            noise_type="pink", modulation_freq=4, max_freq=4000,
            label="rest")
plan = SessionPlan(preset_iri="https://example.org/presets/settle-rest",
                   preset_label="Settle, then rest",
                   created_at=datetime.now(timezone.utc),
                   master_volume=0.5)
planned = to_sstim_graph(session, plan=plan, base=BASE + "session/")
planned.serialize("stim_session.ttl", format="turtle")
read = from_sstim_graph("stim_session.ttl")
assert read.plan == plan
assert np.array_equal(read.render(), session.render())

# Another tool's description, in SSTIM's terms alone: an amplitude
# modulation whose carrier, rate and length it states, but not its depth,
# which SSTIM has no term for.
elsewhere = """
@prefix s: <https://w3id.org/sstim#> .
@prefix v: <https://w3id.org/sstim/vocab#> .
@prefix e: <https://w3id.org/sstim/exposure#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

<urn:elsewhere:am> a s:StimulusSpecification ;
  rdfs:label "Seven hertz on a 220 Hz tone" ;
  s:stimulusRegime "determinate" ;
  s:hasSignal <urn:elsewhere:rate> ;
  s:hasStimulusChannel <urn:elsewhere:ears> .
<urn:elsewhere:rate> a s:StimulationSignal ;
  rdfs:label "7 Hz" ;
  s:hasSignalShape v:shapeSine ; s:hzMin 7.0 ; s:hzMax 7.0 .
<urn:elsewhere:ears> a e:StimulusChannel ;
  rdfs:label "Both ears" ;
  e:perceivedModality e:modalityAuditory ;
  e:deliveryMedium e:mediumAirConductedSound ;
  e:hasBodyPlacement e:placementEars ;
  s:channelDurationSeconds 8.0 ;
  s:hasSignalRendering <urn:elsewhere:am-rendering> .
<urn:elsewhere:am-rendering> a s:SignalRendering ;
  rdfs:label "Amplitude modulation" ;
  s:rendersSignal <urn:elsewhere:rate> ;
  s:rendersOntoParameter v:paramAmplitude ;
  s:hasRenderingMechanism v:mechanismAmplitudeModulation ;
  s:hasRenderingPresence v:presencePhysical ;
  s:renderingCarrierHz 220.0 .
"""
capability = inspect_sstim(elsewhere)
print(f"another tool's description is {capability.generator}, "
      f"and needs {', '.join(capability.needs)}")
music.write_wav_mono(
    render_sstim(elsewhere, sample_rate=44100,
                 controls={"modulation_depth": 0.8}),
    "stim_elsewhere.wav")

print("wrote a stimulus and a planned session as SSTIM, read both back, "
      "and rendered a description written elsewhere")
