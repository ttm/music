"""Sessions as programs, and as SSTIM's planned SessionSpecification."""

from datetime import datetime, timedelta, timezone
import os

import numpy as np
import pytest

pytest.importorskip("rdflib")
from rdflib import Graph, Literal, URIRef  # noqa: E402
from rdflib.namespace import DCTERMS, RDF, RDFS, XSD  # noqa: E402

import music  # noqa: E402
from music.stimulation.sstim import (  # noqa: E402
    MUSIC, SSTIM, SSTIM_V, PlannedSession, SessionPlan, from_sstim_graph,
    render_sstim, to_sstim_graph, validate_sstim,
)

BASE = "https://example.org/music/session-test/"
CREATED = datetime(2026, 10, 11, tzinfo=timezone.utc)
PRESET = "https://example.org/real-preset/version-1"


def mixed():
    session = music.StimulationSession(
        sample_rate=48000, end_ramp=.02, ramp_shape="equal_power")
    session.add(music.binaural_beats, duration=.1, gain=.4, beat_freq=9,
                label="beat")
    session.add(music.modulated_noise, duration=.1, ramp=.03,
                noise_type="pink", seed=17, modulation_freq=6, gain=.1,
                label="noise")
    session.add(music.spatial_motion, duration=.1, ramp=.02,
                carrier_freq=400, motion_rate=2., label="spatial")
    return session


def minute(*, duration=60., sample_rate=8000):
    """A session a plan can be made of, which no test renders."""
    session = music.StimulationSession(sample_rate=sample_rate)
    session.add(music.binaural_beats, duration=duration, carrier_freq=200.,
                beat_freq=10., label="planned auditory stimulation")
    return session


def plan(**kwargs):
    fields = dict(preset_iri=PRESET, preset_label="Versioned test preset",
                  created_at=CREATED, master_volume=.2)
    fields.update(kwargs)
    return SessionPlan(**fields)


def planned(session=None, **kwargs):
    return to_sstim_graph(minute() if session is None else session,
                          plan=plan(**kwargs), base=BASE)


def program_root(graph):
    return next(graph.subjects(RDF.type, MUSIC.StimulationProgram))


def plan_root(graph):
    return next(graph.subjects(RDF.type, SSTIM.SessionSpecification))


def swap(graph, node, predicate, replacement):
    graph.remove((node, predicate, None))
    if replacement is not None:
        graph.add((node, predicate, replacement))


def test_a_program_renders_exactly_the_session_it_describes():
    session = mixed()
    graph = to_sstim_graph(session, base=BASE)
    read = from_sstim_graph(graph)
    assert (read.sample_rate, read.end_ramp, read.ramp_shape) == (
        48000, .02, "equal_power")
    assert [p.label for p in read.phases] == ["beat", "noise", "spatial"]
    np.testing.assert_array_equal(render_sstim(graph), session.render())
    assert render_sstim(graph).shape == (2, 14400)
    assert len(list(graph.subjects(
        RDF.type, SSTIM.StimulusSpecification))) == 3
    assert not list(graph.subjects(RDF.type, SSTIM.SessionInstance))


def test_a_program_of_whole_seconds_reads_back():
    session = music.StimulationSession(sample_rate=8000)
    session.add(music.modulated_noise, duration=1, noise_type="pink",
                seed=3, max_freq=4000)
    session.add(music.spatial_motion, duration=1, ramp=.1)
    graph = to_sstim_graph(session)
    np.testing.assert_array_equal(render_sstim(graph), session.render())


def test_a_program_with_linear_ramps_survives_turtle():
    session = music.StimulationSession(
        sample_rate=44100, ramp_shape="linear", end_ramp=0.)
    session.add(music.frequency_modulation, duration=.1, gain=.5,
                modulation_freq=8)
    session.add(music.isochronic_tones, duration=.1, ramp=.03, pulse_rate=11)
    text = to_sstim_graph(session).serialize(format="turtle")
    np.testing.assert_array_equal(render_sstim(text), session.render())


def test_a_long_session_is_described_without_a_cap():
    session = minute(duration=1800., sample_rate=48000)
    read = from_sstim_graph(to_sstim_graph(session))
    assert read.duration == 1800
    with pytest.raises(ValueError, match="more than max_samples"):
        render_sstim(to_sstim_graph(session))


@pytest.mark.parametrize("action,match", [
    ("wrong-base", "base"),
    ("empty", "from 1 to 50 phases"),
    ("too-many", "from 1 to 50 phases"),
    ("wrong-shape", "ramp shape"),
    ("end-ramp", "end_ramp"),
    ("array-phase", "has no description"),
    ("other-callable", "has no description"),
    ("negative-gain", "ramp and gain"),
    ("negative-duration", "ramp and gain"),
    ("bad-argument", "takes no argument"),
])
def test_a_session_that_cannot_be_described_is_refused(action, match):
    session = music.StimulationSession(sample_rate=48000)
    base = BASE
    if action == "wrong-base":
        base = "https://w3id.org/sstim/test/"
    elif action == "too-many":
        for _ in range(51):
            session.add(music.binaural_beats, duration=.1)
    elif action == "wrong-shape":
        session.ramp_shape = "triangle"
    elif action == "end-ramp":
        session.end_ramp = float("inf")
    elif action == "array-phase":
        session.add(np.ones(200), gain=.2)
    elif action == "other-callable":
        session.add(lambda **kw: music.binaural_beats(**kw), duration=.1)
    elif action == "negative-gain":
        session.add(music.binaural_beats, duration=.1, gain=-.5)
    elif action == "negative-duration":
        session.add(music.binaural_beats, duration=.1)
        session.phases[0].duration = -1
    elif action == "bad-argument":
        session.add(music.binaural_beats, duration=.1, volume=1)
    if action in ("wrong-base", "wrong-shape", "end-ramp"):
        session.add(music.binaural_beats, duration=.1)
    with pytest.raises(ValueError, match=match):
        to_sstim_graph(session, base=base)


def test_a_session_takes_no_stimulus_arguments():
    with pytest.raises(TypeError, match="takes no duration"):
        to_sstim_graph(mixed(), duration=1)
    with pytest.raises(TypeError, match="takes no duration"):
        to_sstim_graph(mixed(), beat_freq=3)


@pytest.mark.parametrize("problem,match", [
    ("two-roots", "exactly one MUSIC StimulationProgram"),
    ("renamed-root", "not named ...program"),
    ("two-ramp-shapes", "settings are unreadable"),
    ("no-phases", "from 1 to 50 phases"),
    ("too-many", "from 1 to 50 phases"),
    ("order", "numbered 1, 2, 3"),
    ("no-order", "expected exactly one .*order"),
    ("stimulus-iri", "names another phase's stimulus"),
    ("missing-stimulus", "exactly one SSTIM StimulusSpecification"),
    ("missing-generator", "expected exactly one .*generator"),
    ("unknown-generator", "not a stimulus a description can be written"),
    ("phase-duration", "another rate or length than the phase"),
    ("changed-signal", "not those the package writes"),
    ("extra-package-term", "unexpected .*unexpectedNote"),
])
def test_the_reader_refuses_a_program_it_did_not_write(problem, match):
    graph = to_sstim_graph(mixed(), base=BASE)
    root = program_root(graph)
    phase = URIRef(BASE + "phase-1")
    stimulus = URIRef(BASE + "phase-1/specification")
    if problem == "two-roots":
        graph.add((URIRef(BASE + "elsewhere"), RDF.type,
                   MUSIC.StimulationProgram))
    elif problem == "renamed-root":
        graph.remove((root, RDF.type, MUSIC.StimulationProgram))
        graph.add((URIRef(BASE + "broken"), RDF.type,
                   MUSIC.StimulationProgram))
    elif problem == "two-ramp-shapes":
        graph.add((root, MUSIC.rampShape, Literal("linear")))
    elif problem == "no-phases":
        graph.remove((root, MUSIC.hasPhase, None))
    elif problem == "too-many":
        for index in range(51):
            graph.add((root, MUSIC.hasPhase, URIRef(BASE + f"x-{index}")))
    elif problem == "order":
        swap(graph, phase, MUSIC.order, Literal(9, datatype=XSD.integer))
    elif problem == "no-order":
        swap(graph, phase, MUSIC.order, None)
    elif problem == "stimulus-iri":
        swap(graph, phase, MUSIC.phaseStimulus,
             URIRef(BASE + "phase-2/specification"))
    elif problem == "missing-stimulus":
        for triple in list(graph):
            if str(triple[0]).startswith(BASE + "phase-1/"):
                graph.remove(triple)
    elif problem in ("missing-generator", "unknown-generator"):
        swap(graph, stimulus, MUSIC.generator,
             None if problem == "missing-generator" else Literal("eval"))
    elif problem == "phase-duration":
        swap(graph, phase, MUSIC.phaseDurationSeconds,
             Literal(".7", datatype=XSD.decimal))
    elif problem == "changed-signal":
        signal = URIRef(BASE + "phase-1/signal")
        graph.add((signal, SSTIM.hasSignalShape, SSTIM_V.shapeSquare))
    else:
        graph.add((root, MUSIC.unexpectedNote, Literal("not allowed")))
    with pytest.raises(ValueError, match=match):
        from_sstim_graph(graph)


def test_a_session_is_read_as_written_with_nothing_else():
    graph = to_sstim_graph(mixed())
    with pytest.raises(ValueError, match="no controls or profile"):
        from_sstim_graph(graph, controls={"gain": 1})
    with pytest.raises(ValueError, match="no controls or profile"):
        from_sstim_graph(graph, profile="zero-phase-equal-gain-beats-v1")
    with pytest.raises(ValueError, match="fixes its sample rate at 48000"):
        from_sstim_graph(graph, sample_rate=44100)
    with pytest.raises(ValueError, match="lasts 0.3 s"):
        from_sstim_graph(graph, duration=1)
    assert from_sstim_graph(graph, sample_rate=48000, duration=.3)
    with pytest.raises(ValueError, match="oversampling_factor is for"):
        render_sstim(graph, oversampling_factor=4)


def test_a_plan_is_sstims_own_session_specification():
    graph = planned()
    assert (None, RDF.type, SSTIM.SessionSpecification) in graph
    assert (URIRef(PRESET), RDF.type, SSTIM.Preset) in graph
    assert (None, RDF.type, SSTIM.SessionInstance) not in graph
    root = plan_root(graph)
    assert (root, SSTIM.hasReproducibilityLevel,
            SSTIM_V.reproEquivalentPresentation) in graph
    assert (root, SSTIM.durationSeconds,
            Literal(60, datatype=XSD.integer)) in graph
    assert len(str(graph.value(root, SSTIM.configurationDigest))) == 64
    assert (root, SSTIM.digestAlgorithm, Literal(
        "sha256-music-plan-json-v1", datatype=XSD.string)) in graph
    assert not any(str(s).startswith("https://w3id.org/sstim")
                   for s, _, _ in graph)
    read = from_sstim_graph(graph)
    assert isinstance(read, PlannedSession)
    assert read.plan == plan()
    assert read.session.duration == 60
    assert len(read.session.phases) == 1
    text = graph.serialize(format="turtle")
    assert from_sstim_graph(text).plan == plan()


def test_a_planned_session_is_described_again_as_it_was():
    read = from_sstim_graph(planned())
    assert set(to_sstim_graph(read, base=BASE)) == set(planned())
    with pytest.raises(TypeError, match="has its plan"):
        to_sstim_graph(read, plan=plan())


def test_a_plan_up_to_two_hours_is_written_and_read():
    graph = planned(minute(duration=7200., sample_rate=48000))
    assert from_sstim_graph(graph).session.duration == 7200
    with pytest.raises(ValueError, match="more than max_samples"):
        render_sstim(graph)


def test_a_plan_renders_its_session():
    session = minute(duration=60., sample_rate=8000)
    np.testing.assert_array_equal(
        render_sstim(planned(session)), session.render())


@pytest.mark.parametrize("fields,match", [
    ({"preset_iri": ""}, "http\\(s\\) IRI outside SSTIM's"),
    ({"preset_iri": 3}, "http\\(s\\) IRI outside SSTIM's"),
    ({"preset_iri": "https://w3id.org/sstim#Fake"}, "outside SSTIM's"),
    ({"preset_label": " "}, "must have a name"),
    ({"created_at": datetime(2026, 10, 11)}, "with its time zone"),
    ({"created_at": "2026-10-11"}, "with its time zone"),
    ({"master_volume": -1}, "master volume"),
    ({"master_volume": float("nan")}, "master volume"),
    ({"master_volume": True}, "master volume"),
    ({"master_volume": "1"}, "master volume"),
])
def test_a_plan_sstim_would_not_accept_is_refused(fields, match):
    with pytest.raises(ValueError, match=match):
        planned(**fields)


@pytest.mark.parametrize("seconds", [59., 60.5, 7201.])
def test_a_plan_is_a_whole_number_of_seconds_from_60_to_7200(seconds):
    with pytest.raises(ValueError, match="whole number of seconds"):
        planned(minute(duration=seconds))


def test_a_plan_must_be_a_session_plan():
    with pytest.raises(TypeError, match="SessionPlan"):
        to_sstim_graph(minute(), plan={"preset_iri": PRESET})
    with pytest.raises(TypeError, match="StimulationSession"):
        from music.stimulation.sstim._sessions import program_graph
        program_graph("not a session", BASE)


@pytest.mark.parametrize("mutation,match", [
    ("untyped-root", "statements are not those the package writes"),
    ("two-roots", "exactly one SSTIM SessionSpecification"),
    ("renamed-root", "not named ...session-specification"),
    ("other-program", "must name its own program"),
    ("no-program-link", "expected exactly one .*hasProgram"),
    ("preset-type", "must be an sstim:Preset"),
    ("preset-label", "settings are unreadable"),
    ("no-created", "expected exactly one .*created"),
    ("created-type", "must be an xsd:dateTime"),
    ("created-value", "settings are unreadable"),
    ("volume", "master volume"),
    ("duration-value", "settings are unreadable"),
    ("duration", "duration is not its program's"),
    ("reproducibility", "not those the package writes"),
    ("digest", "not those the package writes"),
    ("digest-algorithm", "not those the package writes"),
    ("extra-term", "unexpected .*unrecognizedControl"),
    ("program-gone", "exactly one MUSIC StimulationProgram"),
])
def test_the_reader_refuses_a_plan_it_did_not_write(mutation, match):
    graph = planned()
    root = plan_root(graph)
    preset = URIRef(PRESET)
    if mutation == "untyped-root":
        graph.remove((root, RDF.type, SSTIM.SessionSpecification))
    elif mutation == "two-roots":
        graph.add((URIRef("urn:other"), RDF.type,
                   SSTIM.SessionSpecification))
    elif mutation == "renamed-root":
        graph.remove((root, RDF.type, SSTIM.SessionSpecification))
        graph.add((URIRef(BASE + "elsewhere"), RDF.type,
                   SSTIM.SessionSpecification))
    elif mutation == "other-program":
        swap(graph, root, MUSIC.hasProgram, URIRef("urn:invalid"))
    elif mutation == "no-program-link":
        swap(graph, root, MUSIC.hasProgram, None)
    elif mutation == "preset-type":
        graph.remove((preset, RDF.type, SSTIM.Preset))
    elif mutation == "preset-label":
        swap(graph, preset, RDFS.label, None)
    elif mutation == "no-created":
        swap(graph, root, DCTERMS.created, None)
    elif mutation == "created-type":
        swap(graph, root, DCTERMS.created, Literal("not-a-date"))
    elif mutation == "created-value":
        swap(graph, root, DCTERMS.created,
             Literal("no-such-date", datatype=XSD.dateTime))
    elif mutation == "volume":
        swap(graph, root, SSTIM.masterVolume, Literal("nan"))
    elif mutation == "duration-value":
        swap(graph, root, SSTIM.durationSeconds, Literal("not-int"))
    elif mutation == "duration":
        swap(graph, root, SSTIM.durationSeconds,
             Literal(61, datatype=XSD.integer))
    elif mutation == "reproducibility":
        swap(graph, root, SSTIM.hasReproducibilityLevel,
             SSTIM_V.reproIdenticalRendering)
    elif mutation == "digest":
        swap(graph, root, SSTIM.configurationDigest,
             Literal("0" * 64, datatype=XSD.string))
    elif mutation == "digest-algorithm":
        swap(graph, root, SSTIM.digestAlgorithm, None)
    elif mutation == "extra-term":
        graph.add((root, MUSIC.unrecognizedControl, Literal(17)))
    else:
        for triple in list(graph):
            if str(triple[0]) == BASE + "program":
                graph.remove(triple)
    with pytest.raises(ValueError, match=match):
        from_sstim_graph(graph)


def test_a_session_specification_in_sstims_terms_alone_is_not_rendered():
    graph = Graph()
    for triple in planned():
        if not any(str(term).startswith(str(MUSIC)) for term in triple):
            graph.add(triple)
    with pytest.raises(ValueError, match="no terms for a session's phases"):
        from_sstim_graph(graph)


def test_the_configuration_digest_follows_the_preset_and_the_volume():
    def digest(graph):
        return str(graph.value(plan_root(graph), SSTIM.configurationDigest))
    original = digest(planned(master_volume=.2))
    assert digest(planned(master_volume=.3)) != original
    assert digest(planned(preset_iri=PRESET + "-different")) != original
    assert digest(planned(master_volume=.20)) == original
    later = planned(created_at=CREATED + timedelta(days=1))
    assert digest(later) == original


def test_a_plan_made_in_another_time_zone_reads_back():
    zone = timezone(timedelta(hours=-3))
    graph = planned(created_at=datetime(2026, 10, 11, 9, 30, tzinfo=zone))
    assert from_sstim_graph(graph).plan.created_at.utcoffset() == (
        timedelta(hours=-3))


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SSTIM_LIVE") != "1",
                    reason="opt-in live SSTIM validation on one CI runner")
@pytest.mark.parametrize("make", [
    lambda: to_sstim_graph(mixed(), base=BASE), planned])
def test_programs_and_plans_conform_to_sstims_full_profile(make):
    pytest.importorskip("sstim")
    report = validate_sstim(make())
    assert report.ok, str(report)


def test_a_phases_gain_is_read_from_the_program():
    graph = to_sstim_graph(mixed(), base=BASE)
    phase = URIRef(BASE + "phase-1")
    swap(graph, phase, MUSIC.gain, Literal(".9", datatype=XSD.decimal))
    assert from_sstim_graph(graph).phases[0].gain == .9
