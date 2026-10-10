"""Independent RDF producer and analytic beat consumer interoperability."""

import numpy as np
import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from music.stimulation.sstim_io import (
    MUSIC, SSTIM, SSTIM_EX, SSTIM_V, to_sstim_graph,
)
from music.stimulation.sstim_semantic import (
    SemanticBeat, inspect_sstim_beat, render_semantic_beat,
)

PROFILE = "zero-phase-equal-gain-beats-v1"


def authored_graph(stereo=True):
    """Author standard SSTIM Turtle without calling MUSIC's exporter."""
    def chan(name, placement, hz, mechanism, presence):
        return f"""
<urn:portable:{name}> a sstim-ex:StimulusChannel ;
    sstim-ex:perceivedModality sstim-ex:modalityAuditory ;
    sstim-ex:deliveryMedium sstim-ex:mediumAirConductedSound ;
    sstim-ex:hasBodyPlacement sstim-ex:{placement} ;
    sstim:channelDurationSeconds 1.0 ;
    sstim:hasSignalRendering <urn:portable:r-{name}> .
<urn:portable:r-{name}> a sstim:SignalRendering ;
    sstim:rendersSignal <urn:portable:signal> ;
    sstim:rendersOntoParameter sstim-v:paramAmplitude ;
    sstim:hasRenderingMechanism sstim-v:{mechanism} ;
    sstim:hasRenderingPresence sstim-v:{presence} ;
    sstim:renderingCarrierHz {hz} .
"""
    if stereo:
        body = (chan("left", "placementEarLeft", "195.0",
                     "mechanismBinauralBeat", "presencePerceptual")
                + chan("right", "placementEarRight", "205.0",
                       "mechanismBinauralBeat", "presencePerceptual"))
        channels = ("<urn:portable:left>, <urn:portable:right>")
    else:
        body = chan("mono", "placementEars", "200.0",
                    "mechanismMonauralBeat", "presencePhysical")
        channels = "<urn:portable:mono>"
    return Graph().parse(data=f"""
@prefix sstim: <https://w3id.org/sstim#> .
@prefix sstim-v: <https://w3id.org/sstim/vocab#> .
@prefix sstim-ex: <https://w3id.org/sstim/exposure#> .
<urn:portable:spec> a sstim:StimulusSpecification ;
    sstim:stimulusRegime "determinate" ;
    sstim:hasSignal <urn:portable:signal> ;
    sstim:hasStimulusChannel {channels} .
<urn:portable:signal> a sstim:StimulationSignal ;
    sstim:hasSignalShape sstim-v:shapeSine ;
    sstim:hzMin 10.0 ; sstim:hzMax 10.0 .
{body}
""", format="turtle")


def swap(g, subject, predicate, replacement):
    g.remove((subject, predicate, None))
    if replacement is not None:
        g.add((subject, predicate, replacement))


@pytest.mark.parametrize("stereo", [True, False])
def test_independently_authored_sstim_without_music_hints(stereo):
    g = authored_graph(stereo)
    assert not list(g.triples((None, MUSIC.generator, None)))
    spec = inspect_sstim_beat(g, sample_rate=48000)
    assert spec.carrier_freq == 200
    assert spec.beat_freq == 10
    assert spec.duration == 1
    assert spec.technique == ("binaural" if stereo else "monaural")
    out = render_semantic_beat(g, profile=PROFILE, sample_rate=48000)
    low = np.sin(2 * np.pi * 195 * np.arange(48000) / 48000)
    high = np.sin(2 * np.pi * 205 * np.arange(48000) / 48000)
    if stereo:
        np.testing.assert_allclose(out, np.vstack((low, high)), atol=1e-11)
    else:
        np.testing.assert_allclose(out, (low + high) / 2, atol=1e-11)


def test_music_export_can_be_read_without_music_specific_triplets():
    g = to_sstim_graph("binaural_beats",
                       parameters={"carrier_freq": 200, "beat_freq": 10},
                       duration=.1, sample_rate=48000)
    for subject, pred, obj in list(g):
        if str(pred).startswith(str(MUSIC)):
            g.remove((subject, pred, obj))
    x = inspect_sstim_beat(g, sample_rate=48000)
    assert x.technique == "binaural"
    assert render_semantic_beat(x, profile=PROFILE).shape == (2, 4800)


def test_unsupported_explicit_rendering_profile_and_technique():
    with pytest.raises(ValueError, match="explicit"):
        render_semantic_beat(authored_graph(), profile="guess")
    with pytest.raises(ValueError, match="unsupported semantic"):
        render_semantic_beat(
            SemanticBeat("other", 200, 10, .1, 48000), profile=PROFILE)
    with pytest.raises(ValueError, match="invalid sample count"):
        render_semantic_beat(
            SemanticBeat("binaural", 200, 10, 0, 48000), profile=PROFILE)


@pytest.mark.parametrize("stereo,change,match", [
    (True, "no-root", "one SSTIM"),
    (True, "regime", "determinate"),
    (True, "no-signal-type", "signal class"),
    (True, "shape", "sine"),
    (True, "beat-zero", "fixed positive"),
    (True, "beat-variable", "fixed positive"),
    (True, "no-channels", "one or two"),
    (True, "signal-target", "another signal"),
    (True, "no-channel-type", "channel class"),
    (True, "modality", "auditory"),
    (True, "medium", "air-conducted"),
    (True, "no-rendering-type", "rendering class"),
    (True, "parameter", "amplitude"),
    (True, "duplicate-placements", "duplicate"),
    (True, "different-duration", "durations differ"),
    (True, "bad-duration", "duration"),
    (True, "bad-mechanism", "binaural channel semantics"),
    (True, "bad-presence", "binaural channel semantics"),
    (True, "carrier-reversed", "carriers contradict"),
    (True, "carrier-mismatch", "carriers contradict"),
    (True, "carrier-nyquist", "carriers contradict"),
    (False, "bad-mechanism", "monaural channel semantics"),
    (False, "bad-presence", "monaural channel semantics"),
    (False, "bad-placement", "monaural channel semantics"),
    (False, "carrier-nyquist", "outside Nyquist"),
])
def test_semantic_profile_rejects_contradictions(stereo, change, match):
    g = authored_graph(stereo)
    root = URIRef("urn:portable:spec")
    signal = URIRef("urn:portable:signal")
    c0 = URIRef("urn:portable:left" if stereo else "urn:portable:mono")
    r0 = URIRef("urn:portable:r-left" if stereo else "urn:portable:r-mono")
    c1 = URIRef("urn:portable:right")
    r1 = URIRef("urn:portable:r-right")
    bad = URIRef("urn:portable:other")
    if change == "no-root":
        g.remove((root, RDF.type, SSTIM.StimulusSpecification))
    elif change == "regime":
        swap(g, root, SSTIM.stimulusRegime, Literal("stochastic"))
    elif change == "no-signal-type":
        g.remove((signal, RDF.type, SSTIM.StimulationSignal))
    elif change == "shape":
        swap(g, signal, SSTIM.hasSignalShape, SSTIM_V.shapeSquare)
    elif change == "beat-zero":
        swap(g, signal, SSTIM.hzMin, Literal(0))
    elif change == "beat-variable":
        swap(g, signal, SSTIM.hzMax, Literal(11))
    elif change == "no-channels":
        g.remove((root, SSTIM.hasStimulusChannel, None))
    elif change == "signal-target":
        swap(g, r0, SSTIM.rendersSignal, bad)
    elif change == "no-channel-type":
        g.remove((c0, RDF.type, SSTIM_EX.StimulusChannel))
    elif change == "modality":
        swap(g, c0, SSTIM_EX.perceivedModality, SSTIM_EX.modalityVisual)
    elif change == "medium":
        swap(g, c0, SSTIM_EX.deliveryMedium, bad)
    elif change == "no-rendering-type":
        g.remove((r0, RDF.type, SSTIM.SignalRendering))
    elif change == "parameter":
        swap(g, r0, SSTIM.rendersOntoParameter, SSTIM_V.paramFrequency)
    elif change == "duplicate-placements":
        swap(g, c1, SSTIM_EX.hasBodyPlacement,
             SSTIM_EX.placementEarLeft)
    elif change == "different-duration":
        swap(g, c1, SSTIM.channelDurationSeconds, Literal(2))
    elif change == "bad-duration":
        swap(g, c0, SSTIM.channelDurationSeconds, Literal(-1))
        if stereo:
            swap(g, c1, SSTIM.channelDurationSeconds, Literal(-1))
    elif change == "bad-mechanism":
        swap(g, r0, SSTIM.hasRenderingMechanism,
             SSTIM_V.mechanismFrequencyModulation)
    elif change == "bad-presence":
        swap(g, r0, SSTIM.hasRenderingPresence,
             SSTIM_V.presencePhysical if stereo else
             SSTIM_V.presencePerceptual)
    elif change == "bad-placement":
        swap(g, c0, SSTIM_EX.hasBodyPlacement, SSTIM_EX.placementEarLeft)
    elif change == "carrier-reversed":
        swap(g, r0, SSTIM.renderingCarrierHz, Literal(220))
    elif change == "carrier-mismatch":
        swap(g, r1, SSTIM.renderingCarrierHz, Literal(204))
    elif change == "carrier-nyquist":
        swap(g, r1 if stereo else r0, SSTIM.renderingCarrierHz,
             Literal(48000))
    with pytest.raises(ValueError, match=match):
        inspect_sstim_beat(g, sample_rate=48000)


def test_invalid_inspection_arguments_and_numeric_literals():
    with pytest.raises(ValueError, match="sample_rate"):
        inspect_sstim_beat(authored_graph(), sample_rate=True)
    g = authored_graph()
    signal = URIRef("urn:portable:signal")
    swap(g, signal, SSTIM.hzMin, Literal("invalid"))
    with pytest.raises(ValueError, match="numeric"):
        inspect_sstim_beat(g)
    swap(g, signal, SSTIM.hzMin, Literal("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        inspect_sstim_beat(g)
