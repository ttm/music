"""What the package can do with a description, said before it renders."""

from datetime import datetime, timezone

import pytest

pytest.importorskip("rdflib")
from rdflib import Graph, Literal, URIRef  # noqa: E402
from rdflib.namespace import RDF  # noqa: E402

import music  # noqa: E402
from music.stimulation.sstim import (  # noqa: E402
    BEAT_PROFILE, MUSIC, SSTIM, SSTIM_EX, SSTIM_V, STIMULI, SessionPlan,
    SstimCapability, inspect_sstim, render_sstim, to_sstim_graph,
)


def alone(graph):
    """`graph` without any of the package's terms."""
    out = Graph()
    for triple in graph:
        if not any(str(term).startswith(str(MUSIC)) for term in triple):
            out.add(triple)
    return out


@pytest.mark.parametrize("name", STIMULI)
def test_the_packages_own_description_needs_nothing(name):
    capability = inspect_sstim(to_sstim_graph(name))
    beat = name in ("binaural_beats", "monaural_beats")
    assert capability == SstimCapability(
        "stimulus", True, name, (), (BEAT_PROFILE,) if beat else (), "")
    assert capability.can_render


@pytest.mark.parametrize("name,needs", [
    ("binaural_beats", ("sample_rate",)),
    ("monaural_beats", ("sample_rate",)),
    ("isochronic_tones", ("sample_rate", "ramp_duration")),
    ("amplitude_modulation", ("sample_rate", "modulation_depth")),
    ("frequency_modulation", ("sample_rate", "frequency_deviation")),
    ("modulated_noise", ("sample_rate", "modulation_depth", "seed")),
    ("spatial_motion", ("sample_rate", "theta1", "theta2", "dist", "zeta",
                        "air_temp")),
])
def test_a_description_in_sstims_terms_alone_lists_what_it_needs(name,
                                                                 needs):
    capability = inspect_sstim(alone(to_sstim_graph(name)))
    assert capability.kind == "stimulus"
    assert not capability.written_by_package
    assert capability.generator == name
    assert capability.needs == needs
    assert not capability.can_render
    assert capability.profiles == ((BEAT_PROFILE,) if len(needs) == 1
                                   else ())


def test_what_inspection_lists_is_what_rendering_takes():
    graph = alone(to_sstim_graph("amplitude_modulation", duration=.01,
                                 modulation_depth=.3))
    needs = inspect_sstim(graph).needs
    given = {"sample_rate": 44100, "controls": {"modulation_depth": .3}}
    assert set(needs) == {"sample_rate"} | set(given["controls"])
    assert render_sstim(graph, **given).shape == (441,)


def test_sessions_and_plans_the_package_wrote_need_nothing():
    session = music.StimulationSession(sample_rate=8000)
    session.add(music.binaural_beats, duration=60., beat_freq=10)
    program = to_sstim_graph(session)
    assert inspect_sstim(program) == SstimCapability(
        "session", True, None, (), (), "")
    plan = SessionPlan("https://example.org/preset", "Preset",
                       datetime(2026, 10, 1, tzinfo=timezone.utc))
    planned = to_sstim_graph(session, plan=plan)
    assert inspect_sstim(planned) == SstimCapability(
        "planned-session", True, None, (), (), "")
    root = next(program.subjects(RDF.type, MUSIC.StimulationProgram))
    program.add((root, MUSIC.rampShape, Literal("unsupported")))
    assert inspect_sstim(program).kind == "unsupported"
    root = next(planned.subjects(RDF.type, SSTIM.SessionSpecification))
    planned.remove((root, SSTIM.durationSeconds, None))
    broken = inspect_sstim(planned)
    assert (broken.kind, broken.can_render) == ("unsupported", False)
    assert "durationSeconds" in broken.reason


def test_a_session_specification_in_sstims_terms_alone_is_unsupported():
    session = music.StimulationSession(sample_rate=8000)
    session.add(music.binaural_beats, duration=60.)
    plan = SessionPlan("https://example.org/preset", "Preset",
                       datetime(2026, 10, 1, tzinfo=timezone.utc))
    capability = inspect_sstim(alone(to_sstim_graph(session, plan=plan)))
    assert capability.kind == "unsupported"
    assert not capability.written_by_package
    assert "no terms for a session's phases" in capability.reason


def test_nothing_to_render_is_unsupported_and_says_why():
    capability = inspect_sstim(Graph())
    assert capability.kind == "unsupported"
    assert capability.reason == (
        "expected exactly one SSTIM StimulusSpecification")


def test_the_packages_terms_are_never_set_aside_for_sstims():
    graph = to_sstim_graph("binaural_beats", carrier_freq=240, beat_freq=8)
    root = next(graph.subjects(RDF.type, SSTIM.StimulusSpecification))
    graph.remove((root, MUSIC.parametersJson, None))
    graph.add((root, MUSIC.parametersJson,
               Literal('{"carrier_freq": 600, "beat_freq": 8}')))
    capability = inspect_sstim(graph)
    assert capability.kind == "unsupported"
    assert capability.written_by_package
    assert inspect_sstim(alone(graph)).can_render is False
    assert inspect_sstim(alone(graph)).kind == "stimulus"


def test_partial_package_terms_are_not_ignored():
    graph = to_sstim_graph("monaural_beats")
    graph.remove((None, MUSIC.generator, None))
    assert inspect_sstim(graph).kind == "unsupported"


def test_an_unknown_generator_is_never_run():
    graph = to_sstim_graph("frequency_modulation")
    root = next(graph.subjects(RDF.type, SSTIM.StimulusSpecification))
    graph.remove((root, MUSIC.generator, None))
    graph.add((root, MUSIC.generator, Literal("__import__('os')")))
    assert "not a stimulus" in inspect_sstim(graph).reason


def test_a_mechanism_the_package_does_not_render_is_unsupported():
    graph = alone(to_sstim_graph("isochronic_tones"))
    for rendering in list(graph.subjects(RDF.type, SSTIM.SignalRendering)):
        graph.remove((rendering, SSTIM.hasRenderingMechanism, None))
        graph.add((rendering, SSTIM.hasRenderingMechanism,
                   URIRef("https://example.org/mechanism")))
    capability = inspect_sstim(graph)
    assert capability.kind == "unsupported"
    assert "none of the package's stimuli" in capability.reason


def test_a_description_contradicting_itself_is_unsupported_at_once():
    graph = alone(to_sstim_graph("amplitude_modulation"))
    channel = next(graph.subjects(RDF.type, SSTIM_EX.StimulusChannel))
    graph.remove((channel, SSTIM_EX.deliveryMedium, None))
    graph.add((channel, SSTIM_EX.deliveryMedium, SSTIM_EX.mediumLight))
    capability = inspect_sstim(graph)
    assert capability.kind == "unsupported"
    assert "mediumLight" in capability.reason


def test_input_that_is_not_a_local_description_is_an_error():
    with pytest.raises(ValueError, match="URI RDF input denied"):
        inspect_sstim("https://example.org/description.ttl")
    with pytest.raises(TypeError, match="expected RDF Graph"):
        inspect_sstim(3)


def test_a_square_signal_with_a_sine_mechanism_is_isochronic():
    graph = alone(to_sstim_graph("isochronic_tones"))
    assert (None, SSTIM.hasSignalShape, SSTIM_V.shapeSquare) in graph
    assert inspect_sstim(graph).generator == "isochronic_tones"
