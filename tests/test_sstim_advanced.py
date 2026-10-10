"""Test SSTIM noise bands and spatial trajectories without engine drift."""

import os

import numpy as np
import pytest

pytest.importorskip("rdflib")
from rdflib import Graph, Literal  # noqa: E402
from rdflib.namespace import RDF  # noqa: E402

import music  # noqa: E402
from music.stimulation.sstim_advanced import (  # noqa: E402
    from_sstim_advanced_graph, render_sstim_advanced,
    to_sstim_advanced_graph,
)
from music.stimulation.sstim_io import (  # noqa: E402
    MUSIC, SSTIM, SSTIM_V, validate_sstim,
)


def noise_graph(**kw):
    params = {"noise_type": "pink", "min_freq": 100,
              "max_freq": 10000, "modulation_freq": 12, "seed": 42}
    params.update(kw)
    return to_sstim_advanced_graph(
        "modulated_noise", parameters=params,
        duration=.1, sample_rate=48000,
        base="https://example.org/music/noise/")


def spatial_graph(**kw):
    params = {"carrier_freq": 440, "motion_rate": 0.6}
    params.update(kw)
    return to_sstim_advanced_graph(
        "spatial_motion", parameters=params,
        duration=.1, sample_rate=48000,
        base="https://example.org/music/spatial/")


def test_seeded_noise_roundtrip_is_sample_exact_and_global_state_untouched():
    g = noise_graph()
    name, params, duration, rate = from_sstim_advanced_graph(g)
    expected = getattr(music, name)(
        duration=duration, sample_rate=rate, **params)
    np.testing.assert_array_equal(render_sstim_advanced(g), expected)
    np.testing.assert_array_equal(render_sstim_advanced(g),
                                  render_sstim_advanced(g))
    assert str(next(g.objects(None, SSTIM.stimulusRegime))) == "stochastic"
    assert len(list(g.objects(None, SSTIM.hasSignal))) == 2
    assert (None, SSTIM.hasSignalShape, SSTIM_V.shapeNoise) in g


def test_unmodulated_noise_describes_one_spectral_signal():
    g = noise_graph(modulation_freq=0, seed=None)
    assert len(list(g.objects(None, SSTIM.hasSignal))) == 1
    assert render_sstim_advanced(g).shape == (4800,)


def test_spatial_motion_is_stereo_and_asserts_position_signal():
    g = spatial_graph()
    name, params, duration, rate = from_sstim_advanced_graph(g)
    signal = render_sstim_advanced(g)
    np.testing.assert_array_equal(signal, getattr(music, name)(
        duration=duration, sample_rate=rate, **params))
    assert signal.shape == (2, 4800)
    assert (None, SSTIM.rendersOntoParameter,
            SSTIM_V.paramSpatialPosition) in g
    assert len(list(g.objects(None, SSTIM.hasStimulusChannel))) == 2


def test_serialization_roundtrips():
    original = spatial_graph()
    text = original.serialize(format="turtle")
    assert from_sstim_advanced_graph(text) == (
        from_sstim_advanced_graph(original))


def test_semantic_tampering_rejected():
    g = noise_graph()
    root = next(g.subjects(RDF.type, SSTIM.StimulusSpecification))
    g.add((root, SSTIM.stimulusRegime, Literal("determinate")))
    with pytest.raises(ValueError, match="differs"):
        from_sstim_advanced_graph(g)


@pytest.mark.parametrize("generator,params,match", [
    ("fake", {}, "unsupported advanced"),
    ("modulated_noise", {"noise_type": "purple"}, "noise_type"),
    ("modulated_noise", {"noise_type": -3.}, "noise_type"),
    ("modulated_noise", {"seed": -1}, "seed"),
    ("modulated_noise", {"seed": True}, "seed"),
    ("modulated_noise", {"seed": 2**64}, "seed"),
    ("modulated_noise", {"min_freq": 20000}, "invalid noise"),
    ("modulated_noise", {"max_freq": 25000}, "invalid noise"),
    ("modulated_noise", {"modulation_depth": 2}, "invalid noise"),
    ("modulated_noise", {"modulation_freq": -1}, "invalid noise"),
    ("modulated_noise", {"sonic_vector": [1]}, "nonportable"),
    ("modulated_noise", {"modulation_waveform_table": [1]}, "nonportable"),
    ("spatial_motion", {"carrier_freq": 30000}, "invalid spatial"),
    ("spatial_motion", {"motion_rate": -1}, "invalid spatial"),
    ("spatial_motion", {"dist": -1}, "invalid spatial"),
    ("spatial_motion", {"zeta": -1}, "invalid spatial"),
    ("spatial_motion", {"air_temp": 200}, "invalid spatial"),
    ("spatial_motion", {"theta1": float("nan")}, "finite"),
    ("spatial_motion", {"theta1": "north"}, "finite"),
])
def test_reject_unsupported_advanced_inputs(generator, params, match):
    with pytest.raises(ValueError, match=match):
        to_sstim_advanced_graph(
            generator, parameters=params,
            duration=.1, sample_rate=48000)


@pytest.mark.parametrize("kwargs,match", [
    ({"duration": 0}, "duration"),
    ({"duration": float("nan")}, "duration"),
    ({"duration": "1"}, "duration"),
    ({"duration": 2000}, "duration"),
    ({"sample_rate": 1}, "sample_rate"),
    ({"sample_rate": True}, "sample_rate"),
    ({"base": "https://w3id.org/sstim/wrong/"}, "base"),
    ({"base": "not-a-url"}, "base"),
])
def test_reject_invalid_record_metadata(kwargs, match):
    args = dict(generator="spatial_motion",
                parameters={"carrier_freq": 200},
                duration=.1, sample_rate=48000)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        to_sstim_advanced_graph(**args)


def test_advanced_rejects_bad_metadata_and_graph_shape():
    g = noise_graph()
    root = next(g.subjects(RDF.type, SSTIM.StimulusSpecification))
    g.remove((root, MUSIC.parametersJson, None))
    with pytest.raises(ValueError, match="invalid engine parameters"):
        from_sstim_advanced_graph(g)
    with pytest.raises(ValueError, match="one advanced"):
        from_sstim_advanced_graph(Graph())


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SSTIM_LIVE") != "1",
                    reason="pinned ontology check runs on Python 3.12")
@pytest.mark.parametrize("make", [noise_graph, spatial_graph])
def test_official_full_profile_conformance(make):
    result = validate_sstim(make(), version="0.19.0")
    assert result.ok, str(result)
