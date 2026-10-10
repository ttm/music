"""Negative contracts: an RDF input must not execute contradictory stimuli."""

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rdflib")
from rdflib import Literal, URIRef  # noqa: E402
from rdflib.namespace import RDF, XSD  # noqa: E402

from music.stimulation.sstim_io import (
    MUSIC, SSTIM, SSTIM_V, SSTIM_EX, from_sstim_graph,
    to_sstim_graph, validate_sstim,
)


def basic():
    return to_sstim_graph(
        "binaural_beats", duration=.1,
        parameters={"carrier_freq": 200, "beat_freq": 10},
        base="https://example.org/music/test-case/")


def change(graph, node, predicate, replacement):
    current = next(graph.objects(node, predicate))
    graph.remove((node, predicate, current))
    if replacement is not None:
        graph.add((node, predicate, replacement))


def root_of(graph):
    return next(graph.subjects(RDF.type, SSTIM.StimulusSpecification))


def signal_of(graph):
    return next(graph.objects(root_of(graph), SSTIM.hasSignal))


def channel_of(graph):
    return next(graph.objects(root_of(graph), SSTIM.hasStimulusChannel))


def rendering_of(graph):
    return next(graph.objects(channel_of(graph), SSTIM.hasSignalRendering))


@pytest.mark.parametrize("kwargs,match", [
    ({"sample_rate": 1.5}, "sample_rate must be an integer"),
    ({"sample_rate": True}, "sample_rate must be an integer"),
    ({"sample_rate": 100}, "sample_rate must be between"),
    ({"sample_rate": 200000}, "sample_rate must be between"),
    ({"duration": "0.1"}, "duration must be numeric"),
    ({"duration": True}, "duration must be numeric"),
    ({"duration": 0}, "duration must be positive"),
    ({"duration": float("nan")}, "duration must be positive"),
    ({"duration": 1200}, "5M samples"),
    ({"parameters": {"beat_freq": "10"}}, "beat_freq must be numeric"),
    ({"parameters": {"beat_freq": True}}, "beat_freq must be numeric"),
    ({"parameters": {"beat_freq": np.nan}}, "beat_freq must be finite"),
    ({"parameters": {"unrecognized": 1}}, "unsupported or reserved"),
    ({"parameters": {"number_of_samples": 3}}, "unsupported or reserved"),
    ({"parameters": {"carrier_freq": 0}}, "carrier must be inside"),
    ({"parameters": {"carrier_freq": 24000}}, "carrier must be inside"),
    ({"parameters": {"beat_freq": 0}}, "rate positive"),
    ({"parameters": {"carrier_freq": 100, "beat_freq": 250}},
     "beat carriers"),
    ({"base": "https://w3id.org/sstim/stimulus/"},
     "your own http"),
    ({"base": "not-a-url"}, "your own http"),
])
def test_writer_rejects_invalid_source(kwargs, match):
    args = dict(generator="binaural_beats", duration=.1,
                parameters={"carrier_freq": 200, "beat_freq": 10})
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        to_sstim_graph(**args)


def test_read_from_path_and_path_string(tmp_path):
    graph = basic()
    target = tmp_path / "stimulus.ttl"
    target.write_text(graph.serialize(format="turtle"), encoding="utf-8")
    assert from_sstim_graph(target) == from_sstim_graph(str(target))


@pytest.mark.parametrize("kind,match", [
    ("no-root", "exactly one SSTIM"),
    ("extra-root", "exactly one SSTIM"),
    ("no-generator", "expected exactly one"),
    ("two-generators", "expected exactly one"),
    ("huge-parameters", "excessively long"),
    ("bad-json", "invalid parametersJson"),
    ("json-array", "must be an object"),
    ("wrong-rate-type", "sampleRateHz must be"),
    ("no-signal", "expected exactly one"),
    ("no-channel", "must have a channel"),
    ("wrong-channel-count", "wrong channel count"),
    ("bad-shape", "signal assertions"),
    ("non-determinate", "only determinate"),
    ("wrong-channel-duration", "channel duration mismatch"),
    ("wrong-rendering-signal", "signal rendering mismatch"),
    ("wrong-renderer-method", "rendering mechanism mismatch"),
    ("wrong-rendered-parameter", "rendering parameter mismatch"),
    ("wrong-presence", "rendering presence mismatch"),
    ("bad-placement", "invalid binaural channel placement"),
    ("duplicate-placement", "duplicate binaural channel placement"),
    ("wrong-carrier", "carrier frequency mismatch"),
])
def test_reader_rejects_inconsistent_graphs(kind, match):
    g = basic()
    root, signal, channel, rendering = (
        root_of(g), signal_of(g), channel_of(g), rendering_of(g))
    if kind == "no-root":
        g.remove((root, RDF.type, SSTIM.StimulusSpecification))
    elif kind == "extra-root":
        g.add((URIRef("https://example.org/extra"), RDF.type,
               SSTIM.StimulusSpecification))
    elif kind == "no-generator":
        change(g, root, MUSIC.generator, None)
    elif kind == "two-generators":
        g.add((root, MUSIC.generator, Literal("monaural_beats")))
    elif kind in ("huge-parameters", "bad-json", "json-array"):
        text = {"huge-parameters": "a" * 2049, "bad-json": "{",
                "json-array": "[]"}[kind]
        change(g, root, MUSIC.parametersJson, Literal(text))
    elif kind == "wrong-rate-type":
        change(g, root, MUSIC.sampleRateHz,
               Literal("44100.0", datatype=XSD.decimal))
    elif kind == "no-signal":
        change(g, root, SSTIM.hasSignal, None)
    elif kind == "no-channel":
        g.remove((root, SSTIM.hasStimulusChannel, None))
    elif kind == "wrong-channel-count":
        g.remove((root, SSTIM.hasStimulusChannel, channel))
    elif kind == "bad-shape":
        change(g, signal, SSTIM.hasSignalShape, SSTIM_V.shapeSquare)
    elif kind == "non-determinate":
        change(g, root, SSTIM.stimulusRegime, Literal("adaptive"))
    elif kind == "wrong-channel-duration":
        change(g, channel, SSTIM.channelDurationSeconds,
               Literal(".2", datatype=XSD.decimal))
    elif kind == "wrong-rendering-signal":
        change(g, rendering, SSTIM.rendersSignal, URIRef(
            "https://example.org/other-signal"))
    elif kind == "wrong-renderer-method":
        change(g, rendering, SSTIM.hasRenderingMechanism,
               SSTIM_V.mechanismFrequencyModulation)
    elif kind == "wrong-rendered-parameter":
        change(g, rendering, SSTIM.rendersOntoParameter,
               SSTIM_V.paramFrequency)
    elif kind == "wrong-presence":
        change(g, rendering, SSTIM.hasRenderingPresence,
               SSTIM_V.presencePhysical)
    elif kind == "bad-placement":
        change(g, channel, SSTIM_EX.hasBodyPlacement,
               SSTIM_EX.placementEyes)
    elif kind == "duplicate-placement":
        for ch in g.objects(root, SSTIM.hasStimulusChannel):
            change(g, ch, SSTIM_EX.hasBodyPlacement,
                   SSTIM_EX.placementEarLeft)
    elif kind == "wrong-carrier":
        change(g, rendering, SSTIM.renderingCarrierHz,
               Literal("999.0", datatype=XSD.decimal))
    with pytest.raises(ValueError, match=match):
        from_sstim_graph(g)


def test_validation_adapter_calls_official_client(monkeypatch):
    calls = []
    fake = SimpleNamespace(validate=lambda graph, **kw: (
        calls.append(kw), SimpleNamespace(ok=True))[1])
    monkeypatch.setitem(sys.modules, "sstim", fake)
    result = validate_sstim(basic())
    assert result.ok
    assert calls == [{"profile": "full", "version": "0.19.0"}]


def test_validation_adapter_requires_optional_package(monkeypatch):
    original = builtins.__import__

    def fail_import(name, *args, **kwargs):
        if name == "sstim":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_import)
    with pytest.raises(ImportError, match="music\\[sstim\\]"):
        validate_sstim(basic())
