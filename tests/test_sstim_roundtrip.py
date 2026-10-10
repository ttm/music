"""SSTIM frozen-release interoperability: machine-readable round trips."""

import numpy as np
import pytest

import music

rdflib = pytest.importorskip("rdflib")
from rdflib import Literal, URIRef  # noqa: E402

from music.stimulation.sstim_io import (  # noqa: E402
    MUSIC, SSTIM, SSTIM_V, from_sstim_graph, render_sstim,
    to_sstim_graph, validate_sstim,
)

CASES = (
    ("binaural_beats", {"carrier_freq": 200, "beat_freq": 10}),
    ("monaural_beats", {"carrier_freq": 200, "beat_freq": 10}),
    ("isochronic_tones", {
        "carrier_freq": 200, "pulse_rate": 10, "duty_cycle": .3,
        "ramp_duration": .003}),
    ("amplitude_modulation", {
        "carrier_freq": 200, "modulation_freq": 10,
        "modulation_depth": .75}),
    ("frequency_modulation", {
        "carrier_freq": 200, "modulation_freq": 10,
        "frequency_deviation": 20}),
)


@pytest.mark.parametrize("generator,params", CASES)
def test_sstim_roundtrip_preserves_exact_output(generator, params):
    graph = to_sstim_graph(
        generator, parameters=params, duration=.02,
        sample_rate=48000, base="https://example.org/stimulus/test/")
    decoded = from_sstim_graph(graph)
    assert decoded.generator == generator
    assert decoded.sample_rate == 48000
    assert decoded.duration == .02
    expected = getattr(music, generator)(
        duration=.02, sample_rate=48000, **params)
    np.testing.assert_array_equal(render_sstim(graph), expected)
    turtle = graph.serialize(format="turtle")
    assert "https://w3id.org/sstim#" in turtle
    reread = from_sstim_graph(turtle)
    assert reread == decoded


def test_binaural_presence_is_perceptual_and_channels_are_distinct():
    graph = to_sstim_graph(
        "binaural_beats", parameters={
            "carrier_freq": 200, "beat_freq": 10}, duration=1)
    renderings = list(graph.subjects(
        SSTIM.hasRenderingPresence, SSTIM_V.presencePerceptual))
    assert len(renderings) == 2
    assert sorted(float(x) for x in graph.objects(
        None, SSTIM.renderingCarrierHz)) == [195, 205]


def test_tampered_signal_or_mechanism_must_fail_closed():
    graph = to_sstim_graph(
        "monaural_beats", parameters={
            "carrier_freq": 200, "beat_freq": 10}, duration=.01)
    signal = next(graph.subjects(SSTIM.hzMin, None))
    old = next(graph.objects(signal, SSTIM.hzMin))
    graph.remove((signal, SSTIM.hzMin, old))
    graph.add((signal, SSTIM.hzMin, Literal(33)))
    with pytest.raises(ValueError, match="signal assertions"):
        from_sstim_graph(graph)


def test_rdf_does_not_execute_arbitrary_generator():
    graph = to_sstim_graph(
        "binaural_beats", parameters={
            "carrier_freq": 200, "beat_freq": 10}, duration=.01)
    root = next(graph.subjects(MUSIC.generator, None))
    old = next(graph.objects(root, MUSIC.generator))
    graph.remove((root, MUSIC.generator, old))
    graph.add((root, MUSIC.generator, Literal("__import__('os').system")))
    with pytest.raises(ValueError, match="unsupported stimulus generator"):
        render_sstim(graph)


@pytest.mark.parametrize("generator,params", [
    ("not_a_generator", {}),
    ("binaural_beats", {"beat_freq": -1}),
    ("monaural_beats", {"carrier_freq": 5000, "beat_freq": 10000}),
    ("isochronic_tones", {"duty_cycle": 0}),
    ("amplitude_modulation", {"modulation_depth": 5}),
    ("binaural_beats", {"sample_rate": 24000}),
    ("binaural_beats", {"carrier_freq": float("nan")}),
])
def test_invalid_source_descriptions_fail(generator, params):
    with pytest.raises(ValueError):
        to_sstim_graph(generator, parameters=params, duration=.1)


def test_oversampling_rdf_render_matches_manual_call():
    pytest.importorskip("scipy")
    graph = to_sstim_graph(
        "isochronic_tones",
        parameters={"carrier_freq": 500, "pulse_rate": 20},
        duration=.02, sample_rate=48000)
    expected = music.render_oversampled(
        music.isochronic_tones, duration=.02, sample_rate=48000,
        carrier_freq=500, pulse_rate=20)
    np.testing.assert_array_equal(
        render_sstim(graph, oversampling_factor=4), expected)


def test_no_remote_graph_fetch_from_untrusted_iri():
    with pytest.raises(ValueError, match="local"):
        from_sstim_graph("https://example.org/not-a-file.ttl")


def test_official_019_full_profile_validation():
    pytest.importorskip("sstim")
    graph = to_sstim_graph(
        "binaural_beats",
        parameters={"carrier_freq": 200, "beat_freq": 10},
        duration=.1, base="https://example.org/music/official-test/")
    report = validate_sstim(graph, version="0.19.0")
    assert report.ok, str(report)
