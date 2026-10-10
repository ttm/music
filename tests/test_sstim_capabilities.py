"""Capabilities are explicit, with unsupported RDF never executed."""

import pytest

pytest.importorskip("rdflib")
from rdflib import Graph, Literal  # noqa: E402

import music  # noqa: E402
from music.stimulation.sstim_io import (  # noqa: E402
    MUSIC, SSTIM, to_sstim_graph,
)
from music.stimulation.sstim_advanced import (  # noqa: E402
    to_sstim_advanced_graph,
)
from music.stimulation.sstim_program import (  # noqa: E402
    to_sstim_program_graph,
)
from music.stimulation.sstim_capabilities import (  # noqa: E402
    inspect_sstim_capabilities,
)


def drop_music(graph):
    for triple in list(graph):
        if str(triple[1]).startswith(str(MUSIC)):
            graph.remove(triple)
    return graph


@pytest.mark.parametrize("name,params", [
    ("binaural_beats", {"carrier_freq": 200, "beat_freq": 10}),
    ("monaural_beats", {"carrier_freq": 200, "beat_freq": 10}),
])
def test_standard_beats_are_renderable_only_with_explicit_assumptions(
        name, params):
    g = drop_music(to_sstim_graph(
        name, parameters=params, duration=.1, sample_rate=48000))
    report = inspect_sstim_capabilities(g)
    assert report.mode == "portable-beat-reference"
    assert report.can_render
    assert not report.exact_pcm_portable
    assert "phase" in report.missing_contract[0]


@pytest.mark.parametrize("name,params", [
    ("isochronic_tones", {"carrier_freq": 240, "pulse_rate": 11}),
    ("amplitude_modulation", {
        "carrier_freq": 240, "modulation_freq": 11}),
    ("frequency_modulation", {
        "carrier_freq": 240, "modulation_freq": 11}),
])
def test_rich_dsp_requires_music_engine_hints(name, params):
    g = to_sstim_graph(
        name, parameters=params, duration=.1, sample_rate=48000)
    present = inspect_sstim_capabilities(g)
    assert present.mode == "music-extension"
    assert present.can_render
    assert not present.exact_pcm_portable
    stripped = inspect_sstim_capabilities(drop_music(g))
    assert stripped.mode == "descriptive-only"
    assert not stripped.can_render


@pytest.mark.parametrize("name,params,expected", [
    ("modulated_noise", {"noise_type": "pink", "seed": 7}, "noise"),
    ("spatial_motion", {"carrier_freq": 240}, "spatial"),
])
def test_advanced_rdf_curation_vs_renderability(name, params, expected):
    g = to_sstim_advanced_graph(
        name, parameters=params, duration=.1, sample_rate=48000)
    with_hints = inspect_sstim_capabilities(g)
    assert with_hints.mode == "music-extension"
    assert with_hints.technique == name
    stripped = inspect_sstim_capabilities(drop_music(g))
    assert stripped.mode == "descriptive-only"
    assert stripped.technique == expected
    assert not stripped.can_render


def test_music_program_requires_valid_consistent_extension():
    plan = music.StimulationSession(sample_rate=48000)
    plan.add(music.binaural_beats, duration=.1, beat_freq=10)
    g = to_sstim_program_graph(plan)
    report = inspect_sstim_capabilities(g)
    assert report.mode == "music-program"
    assert not report.exact_pcm_portable
    root = next(g.subjects(None, MUSIC.StimulationProgram))
    g.add((root, MUSIC.rampShape, Literal("unsupported")))
    bad = inspect_sstim_capabilities(g)
    assert bad.mode == "unsupported"
    assert not bad.can_render


def test_unknown_and_invalid_music_hints_fail_closed():
    assert inspect_sstim_capabilities(Graph()).mode == "unsupported"
    g = to_sstim_graph(
        "frequency_modulation", parameters={"modulation_freq": 13},
        duration=.1)
    root = next(g.subjects(None, SSTIM.StimulusSpecification))
    g.add((root, MUSIC.generator, Literal("arbitrary-code")))
    assert inspect_sstim_capabilities(g).mode == "unsupported"
    g.remove((root, MUSIC.generator, None))
    g.add((root, MUSIC.generator, Literal("arbitrary-code")))
    bad = inspect_sstim_capabilities(g)
    assert bad.mode == "unsupported"


def test_unknown_standard_rendering_is_not_pretended_executable():
    g = to_sstim_graph("isochronic_tones",
                       parameters={"pulse_rate": 10}, duration=.1)
    drop_music(g)
    g.remove((None, SSTIM.hasRenderingMechanism, None))
    report = inspect_sstim_capabilities(g)
    assert report.mode == "unsupported"
    assert not report.can_render


def test_contradictory_music_beat_hints_cannot_be_downgraded():
    """Valid beat RDF with bad MUSIC params is not a portable reference."""
    graph = to_sstim_graph(
        "binaural_beats",
        parameters={"carrier_freq": 240, "beat_freq": 8},
        duration=.1)
    root = next(graph.subjects(None, SSTIM.StimulusSpecification))
    graph.remove((root, MUSIC.parametersJson, None))
    graph.add((root, MUSIC.parametersJson,
               Literal('{"carrier_freq": 600, "beat_freq": 8}')))
    assert inspect_sstim_capabilities(graph).mode == "unsupported"


def test_partial_music_annotations_cannot_be_ignored():
    graph = to_sstim_graph(
        "monaural_beats",
        parameters={"carrier_freq": 240, "beat_freq": 8},
        duration=.1)
    graph.remove((None, MUSIC.generator, None))
    assert inspect_sstim_capabilities(graph).mode == "unsupported"
