"""An ordered MUSIC program cannot invent delivered SSTIM session events."""

import os

import numpy as np
import pytest

pytest.importorskip("rdflib")
from rdflib import Literal, URIRef  # noqa: E402
from rdflib.namespace import RDF, XSD  # noqa: E402

import music  # noqa: E402
from music.stimulation.sstim_program import (  # noqa: E402
    from_sstim_program_graph, render_sstim_program,
    to_sstim_program_graph,
)
from music.stimulation.sstim_io import (  # noqa: E402
    MUSIC, SSTIM, validate_sstim,
)


def mixed_program():
    session = music.StimulationSession(
        sample_rate=48000, end_ramp=.02, ramp_shape="equal_power")
    session.add(music.binaural_beats, duration=.1, gain=.4,
                beat_freq=9, label="beat")
    session.add(music.modulated_noise, duration=.1, ramp=.03,
                noise_type="pink", seed=17, modulation_freq=6,
                gain=.1, label="noise")
    session.add(music.spatial_motion, duration=.1, ramp=.02,
                carrier_freq=400, motion_rate=2., label="spatial")
    return session


def test_program_roundtrip_is_sample_exact_with_seeded_noise():
    session = mixed_program()
    graph = to_sstim_program_graph(
        session, base="https://example.org/music/program-one/")
    rebuilt = from_sstim_program_graph(graph)
    assert rebuilt.sample_rate == session.sample_rate
    assert rebuilt.end_ramp == session.end_ramp
    assert rebuilt.ramp_shape == session.ramp_shape
    assert [p.label for p in rebuilt.phases] == [
        "beat", "noise", "spatial"]
    np.testing.assert_array_equal(render_sstim_program(graph),
                                  session.render())
    assert render_sstim_program(graph).shape == (2, 14400)
    assert len(list(graph.subjects(
        RDF.type, SSTIM.StimulusSpecification))) == 3
    assert not list(graph.subjects(RDF.type, SSTIM.SessionInstance))


def test_program_serialization_and_linear_ramps():
    session = music.StimulationSession(
        sample_rate=44100, ramp_shape="linear", end_ramp=0.0)
    session.add(music.frequency_modulation, duration=.1, gain=.5,
                modulation_freq=8)
    session.add(music.isochronic_tones, duration=.1, ramp=.03,
                pulse_rate=11)
    graph = to_sstim_program_graph(session)
    out = render_sstim_program(graph.serialize(format="turtle"))
    np.testing.assert_array_equal(out, session.render())


def test_program_malformed_phase_order_rejected():
    graph = to_sstim_program_graph(mixed_program())
    part = next(graph.subjects(MUSIC.order, Literal(1)))
    graph.remove((part, MUSIC.order, None))
    graph.add((part, MUSIC.order, Literal(9, datatype=XSD.integer)))
    with pytest.raises(ValueError, match="phase order"):
        from_sstim_program_graph(graph)


def test_program_semantic_graph_tampering_rejected():
    graph = to_sstim_program_graph(mixed_program())
    root = next(graph.subjects(RDF.type, MUSIC.StimulationProgram))
    graph.add((root, MUSIC.rampShape, Literal("unknown")))
    with pytest.raises(ValueError, match="invalid program metadata"):
        from_sstim_program_graph(graph)


@pytest.mark.parametrize("action,match", [
    ("wrong-base", "base"),
    ("empty", "1 to 50"),
    ("wrong-shape", "ramp shape"),
    ("end-ramp", "end_ramp"),
    ("array-phase", "nonportable"),
    ("negative-gain", "duration, ramp, or gain"),
    ("negative-duration", "duration, ramp, or gain"),
    ("long-duration", "5 million"),
])
def test_refuse_unportable_sessions(action, match):
    s = music.StimulationSession(sample_rate=48000)
    base = "https://example.org/music/safe/"
    if action == "wrong-base":
        base = "https://w3id.org/sstim/test/"
    elif action == "wrong-shape":
        s.ramp_shape = "triangle"
    elif action == "end-ramp":
        s.end_ramp = float("inf")
    elif action == "array-phase":
        s.add(np.ones(200), gain=.2)
    elif action == "negative-gain":
        s.add(music.binaural_beats, duration=.1, gain=-.5)
    elif action == "negative-duration":
        s.add(music.binaural_beats, duration=.1)
        s.phases[0].duration = -1
    elif action == "long-duration":
        s.add(music.binaural_beats, duration=200)
    if action not in ("empty", "array-phase", "negative-gain",
                      "negative-duration", "long-duration"):
        s.add(music.binaural_beats, duration=.1)
    with pytest.raises(ValueError, match=match):
        to_sstim_program_graph(s, base=base)


def test_refuse_unrecognised_and_missing_program_roots():
    g = to_sstim_program_graph(mixed_program())
    root = next(g.subjects(RDF.type, MUSIC.StimulationProgram))
    g.remove((root, RDF.type, MUSIC.StimulationProgram))
    with pytest.raises(ValueError, match="one MUSIC"):
        from_sstim_program_graph(g)


def test_changed_phase_signal_is_rejected():
    g = to_sstim_program_graph(mixed_program())
    signal = next(g.subjects(RDF.type, SSTIM.StimulationSignal))
    g.add((signal, SSTIM.hasSignalShape, URIRef(
        "https://w3id.org/sstim/vocab#shapeSquare")))
    with pytest.raises(ValueError):
        from_sstim_program_graph(g)


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SSTIM_LIVE") != "1",
                    reason="pinned ontology check runs on Python 3.12")
def test_full_profile_accepts_program_constituent_stimuli():
    g = to_sstim_program_graph(mixed_program())
    result = validate_sstim(g, version="0.19.0")
    assert result.ok, str(result)


@pytest.mark.parametrize("problem,match", [
    ("unexpected-iri", "unexpected program IRI"),
    ("no-phases", "1 to 50"),
    ("too-many", "1 to 50"),
    ("wrong-stimulus-iri", "stimulus IRI mismatch"),
    ("missing-subgraph", "missing stimulus subgraph"),
    ("missing-generator", "missing executable generator"),
    ("unknown-generator", "unknown generator"),
    ("inconsistent-duration", "rate or duration mismatch"),
    ("extra-statement", "triples contradict"),
])
def test_reject_program_ambiguity(problem, match):
    graph = to_sstim_program_graph(mixed_program())
    root = next(graph.subjects(RDF.type, MUSIC.StimulationProgram))
    phase = next(graph.objects(root, MUSIC.hasPhase))
    stimulus = next(graph.objects(phase, MUSIC.phaseStimulus))
    if problem == "unexpected-iri":
        graph.remove((root, RDF.type, MUSIC.StimulationProgram))
        graph.add((URIRef("https://example.org/broken"),
                   RDF.type, MUSIC.StimulationProgram))
    elif problem == "no-phases":
        graph.remove((root, MUSIC.hasPhase, None))
    elif problem == "too-many":
        for index in range(51):
            graph.add((root, MUSIC.hasPhase,
                       URIRef(f"https://example.org/extra-{index}")))
    elif problem == "wrong-stimulus-iri":
        graph.remove((phase, MUSIC.phaseStimulus, stimulus))
        graph.add((phase, MUSIC.phaseStimulus,
                   URIRef("https://example.org/not-this-phase")))
    elif problem == "missing-subgraph":
        stem = str(stimulus)[:-len("specification")]
        for triple in list(graph):
            if str(triple[0]).startswith(stem):
                graph.remove(triple)
    elif problem in ("missing-generator", "unknown-generator"):
        graph.remove((stimulus, MUSIC.generator, None))
        if problem == "unknown-generator":
            graph.add((stimulus, MUSIC.generator, Literal("arbitrary")))
    elif problem == "inconsistent-duration":
        graph.remove((phase, MUSIC.phaseDurationSeconds, None))
        graph.add((phase, MUSIC.phaseDurationSeconds,
                   Literal(".7", datatype=XSD.decimal)))
    else:
        graph.add((root, MUSIC.unexpectedNote, Literal("not allowed")))
    with pytest.raises(ValueError, match=match):
        from_sstim_program_graph(graph)
