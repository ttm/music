"""The seven stimuli, written as SSTIM descriptions and read back exactly."""

import builtins
import inspect
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rdflib")
from rdflib import Graph, Literal, URIRef  # noqa: E402
from rdflib.namespace import DCTERMS, RDF, RDFS, XSD  # noqa: E402

import music  # noqa: E402
from music.stimulation import stimuli  # noqa: E402
from music.stimulation.sstim import (  # noqa: E402
    MUSIC, SSTIM, SSTIM_EX, SSTIM_V, STIMULI, SstimStimulus,
    from_sstim_graph, render_sstim, to_sstim_graph, validate_sstim,
)
from music.stimulation.sstim._vocabulary import decimal  # noqa: E402

BASE = "https://example.org/music/test/"
CASES = (
    ("binaural_beats", {"carrier_freq": 200, "beat_freq": 10}),
    ("monaural_beats", {"carrier_freq": 200, "beat_freq": 10}),
    ("isochronic_tones", {"carrier_freq": 200, "pulse_rate": 10,
                          "duty_cycle": .3, "ramp_duration": .003}),
    ("amplitude_modulation", {"carrier_freq": 200, "modulation_freq": 10,
                              "modulation_depth": .75}),
    ("frequency_modulation", {"carrier_freq": 200, "modulation_freq": 10,
                              "frequency_deviation": 20}),
    ("modulated_noise", {"noise_type": "pink", "min_freq": 100,
                         "max_freq": 10000, "modulation_freq": 12,
                         "seed": 42}),
    ("spatial_motion", {"carrier_freq": 440, "motion_rate": .6}),
)


def described(name="binaural_beats", duration=.1, sample_rate=48000,
              **parameters):
    return to_sstim_graph(name, duration=duration, sample_rate=sample_rate,
                          base=BASE, **parameters)


def root_of(graph):
    return next(graph.subjects(RDF.type, SSTIM.StimulusSpecification))


def channels_of(graph):
    return sorted(graph.objects(root_of(graph), SSTIM.hasStimulusChannel))


def swap(graph, node, predicate, replacement):
    graph.remove((node, predicate, None))
    if replacement is not None:
        graph.add((node, predicate, replacement))


@pytest.mark.parametrize("name,parameters", CASES)
def test_a_description_renders_exactly_the_call_it_describes(name,
                                                             parameters):
    graph = described(name, **parameters)
    stimulus = from_sstim_graph(graph)
    assert stimulus.generator == name
    assert (stimulus.duration, stimulus.sample_rate) == (.1, 48000)
    expected = getattr(stimuli, name)(duration=.1, sample_rate=48000,
                                      **parameters)
    np.testing.assert_array_equal(render_sstim(graph), expected)
    np.testing.assert_array_equal(stimulus.render(), expected)
    assert from_sstim_graph(graph.serialize(format="turtle")) == stimulus


@pytest.mark.parametrize("name,parameters", CASES)
def test_every_argument_left_to_its_default_is_written_out(name, parameters):
    graph = described(name, **parameters)
    written = json.loads(str(graph.value(root_of(graph),
                                         MUSIC.parametersJson)))
    defaults = {k: v.default for k, v in inspect.signature(
        getattr(stimuli, name)).parameters.items()}
    assert set(written) == {k for k in defaults if k not in (
        "duration", "sample_rate", "number_of_samples", "waveform_table",
        "modulation_waveform_table", "sonic_vector")}
    for key, value in written.items():
        assert value == parameters.get(key, defaults[key])


def test_the_call_defaults_to_the_generators_length_and_rate():
    stimulus = from_sstim_graph(to_sstim_graph(stimuli.monaural_beats))
    assert (stimulus.duration, stimulus.sample_rate) == (2., 44100)
    assert stimulus.parameters == {"carrier_freq": 200., "beat_freq": 10.}


def test_a_stimulus_is_named_by_its_function_or_its_name():
    by_name = to_sstim_graph("isochronic_tones", base=BASE)
    by_function = to_sstim_graph(music.isochronic_tones, base=BASE)
    assert set(by_name) == set(by_function)
    read = from_sstim_graph(by_name)
    assert set(to_sstim_graph(read, base=BASE)) == set(by_name)


@pytest.mark.parametrize("value, lexical", [
    (15, "15.0"), (15., "15.0"), (.1, "0.1"),
])
def test_a_decimal_is_written_as_the_float_it_is_read_back_as(value,
                                                               lexical):
    """A reader rebuilds a description from floats and compares.

    An int was written as it came, "15", and rebuilt "15.0", so a
    description of whole seconds could not be read back.
    """
    assert str(decimal(value)) == lexical
    assert decimal(value) == decimal(float(lexical))


@pytest.mark.parametrize("name,parameters", CASES)
def test_a_whole_number_of_seconds_reads_back(name, parameters):
    graph = described(name, duration=1, **parameters)
    assert set(graph) == set(described(name, duration=1., **parameters))
    assert from_sstim_graph(graph).duration == 1.


def test_a_binaural_beat_is_two_perceptual_ears_with_their_tones():
    graph = described(carrier_freq=200, beat_freq=10)
    assert len(list(graph.subjects(
        SSTIM.hasRenderingPresence, SSTIM_V.presencePerceptual))) == 2
    assert sorted(float(x) for x in graph.objects(
        None, SSTIM.renderingCarrierHz)) == [195, 205]
    left, right = (next(graph.subjects(SSTIM_EX.hasBodyPlacement, p))
                   for p in (SSTIM_EX.placementEarLeft,
                             SSTIM_EX.placementEarRight))
    assert float(graph.value(left, SSTIM_EX.hasCarrierFrequencyHz)) == 195
    assert float(graph.value(right, SSTIM_EX.hasCarrierFrequencyHz)) == 205
    assert {float(x) for x in graph.objects(
        None, SSTIM_EX.hasBeatFrequencyHz)} == {10}


@pytest.mark.parametrize("name,parameters,term,value", [
    ("monaural_beats", {"beat_freq": 7}, SSTIM_EX.hasBeatFrequencyHz, 7.),
    ("isochronic_tones", {"duty_cycle": .25}, SSTIM_EX.hasDutyCycle, .25),
    ("isochronic_tones", {"pulse_rate": 6},
     SSTIM_EX.hasModulationFrequencyHz, 6.),
    ("amplitude_modulation", {"carrier_freq": 300},
     SSTIM_EX.hasCarrierFrequencyHz, 300.),
    ("frequency_modulation", {"modulation_freq": 4},
     SSTIM_EX.hasModulationFrequencyHz, 4.),
    ("modulated_noise", {"noise_type": "brown"},
     SSTIM_EX.hasStimulusPattern, SSTIM_EX.audioNoiseBrownRed),
    ("modulated_noise", {"modulation_freq": 3},
     SSTIM_EX.hasModulationFrequencyHz, 3.),
    ("spatial_motion", {"carrier_freq": 330},
     SSTIM_EX.hasToneFrequencyHz, 330.),
    ("spatial_motion", {"motion_rate": .5},
     SSTIM_EX.hasModulationFrequencyHz, .5),
])
def test_channels_state_their_frequencies_in_sstims_own_terms(
        name, parameters, term, value):
    graph = described(name, **parameters)
    stated = {x if isinstance(x, URIRef) else float(x)
              for x in graph.objects(None, term)}
    assert stated == {value}


@pytest.mark.parametrize("parameters", [
    {"noise_type": "black"}, {"noise_type": -4.5},
])
def test_a_noise_without_an_sstim_colour_states_none(parameters):
    graph = described("modulated_noise", **parameters)
    assert not list(graph.objects(None, SSTIM_EX.hasStimulusPattern))
    assert from_sstim_graph(graph).parameters["noise_type"] == (
        parameters["noise_type"])


def test_a_still_source_and_an_unmodulated_noise_state_no_modulation():
    for graph in (described("spatial_motion", motion_rate=0),
                  described("modulated_noise", modulation_freq=0,
                            seed=None)):
        assert not list(graph.objects(
            None, SSTIM_EX.hasModulationFrequencyHz))
    noise = described("modulated_noise", modulation_freq=0)
    assert len(list(noise.objects(None, SSTIM.hasSignal))) == 1
    assert render_sstim(noise).shape == (4800,)


def test_a_seeded_noise_renders_the_same_samples_every_time():
    graph = described("modulated_noise", seed=42)
    np.testing.assert_array_equal(render_sstim(graph), render_sstim(graph))
    assert str(next(graph.objects(None, SSTIM.stimulusRegime))) == (
        "stochastic")
    assert (None, SSTIM.hasSignalShape, SSTIM_V.shapeNoise) in graph


def test_spatial_motion_is_stereo_and_renders_a_position_signal():
    graph = described("spatial_motion", carrier_freq=440, motion_rate=.6)
    assert render_sstim(graph).shape == (2, 4800)
    assert (None, SSTIM.rendersOntoParameter,
            SSTIM_V.paramSpatialPosition) in graph
    assert len(channels_of(graph)) == 2


def test_read_from_a_path_a_path_string_and_turtle_text(tmp_path):
    graph = described()
    target = tmp_path / "stimulus.ttl"
    target.write_text(graph.serialize(format="turtle"), encoding="utf-8")
    assert (from_sstim_graph(target) == from_sstim_graph(str(target))
            == from_sstim_graph(target.read_text()))


def test_oversampled_render_matches_the_manual_call():
    pytest.importorskip("scipy")
    graph = described("isochronic_tones", duration=.02, carrier_freq=500,
                      pulse_rate=20)
    expected = music.render_oversampled(
        music.isochronic_tones, duration=.02, sample_rate=48000,
        carrier_freq=500, pulse_rate=20)
    np.testing.assert_array_equal(
        render_sstim(graph, oversampling_factor=4), expected)


def test_a_long_stimulus_is_described_and_read_but_rendered_only_on_request():
    graph = to_sstim_graph("binaural_beats", duration=3600, base=BASE)
    assert from_sstim_graph(graph).duration == 3600
    with pytest.raises(ValueError, match="158760000 samples a channel"):
        render_sstim(graph)
    short = described(duration=.01)
    with pytest.raises(ValueError, match="more than max_samples, 479"):
        render_sstim(short, max_samples=479)
    assert render_sstim(short, max_samples=480).shape == (2, 480)


@pytest.mark.parametrize("max_samples", [0, -1, 1.5, True, "10"])
def test_max_samples_is_a_positive_integer(max_samples):
    with pytest.raises(ValueError, match="max_samples must be"):
        render_sstim(described(), max_samples=max_samples)


@pytest.mark.parametrize("kwargs,match", [
    ({"sample_rate": 1.5}, "sample_rate must be an integer"),
    ({"sample_rate": True}, "sample_rate must be an integer"),
    ({"sample_rate": 100}, "sample_rate must be an integer"),
    ({"sample_rate": 200000}, "sample_rate must be an integer"),
    ({"duration": "0.1"}, "duration must be a positive"),
    ({"duration": True}, "duration must be a positive"),
    ({"duration": 0}, "duration must be a positive"),
    ({"duration": float("nan")}, "duration must be a positive"),
    ({"beat_freq": "10"}, "beat_freq must be a finite number"),
    ({"beat_freq": True}, "beat_freq must be a finite number"),
    ({"beat_freq": np.bool_(True)}, "beat_freq must be a finite number"),
    ({"beat_freq": np.nan}, "beat_freq must be a finite number"),
    ({"unrecognized": 1}, "takes no argument .* named unrecognized"),
    ({"number_of_samples": 3}, "named number_of_samples"),
    ({"waveform_table": [0, 1]}, "named waveform_table"),
    ({"carrier_freq": 0}, "carrier must lie"),
    ({"carrier_freq": 24000}, "carrier must lie"),
    ({"beat_freq": 0}, "rate must be positive"),
    ({"carrier_freq": 100, "beat_freq": 250}, "both tones of the beat"),
    ({"base": "https://w3id.org/sstim/stimulus/"}, "your own http"),
    ({"base": "not-a-url"}, "your own http"),
    ({"base": "https://example.org/no-separator"}, "your own http"),
])
def test_the_writer_refuses_what_it_cannot_describe(kwargs, match):
    arguments = dict(duration=.1, sample_rate=48000, base=BASE,
                     carrier_freq=200, beat_freq=10)
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        to_sstim_graph("binaural_beats", **arguments)


@pytest.mark.parametrize("name,parameters,match", [
    ("isochronic_tones", {"duty_cycle": 0}, "duty_cycle"),
    ("isochronic_tones", {"ramp_duration": -1}, "duty_cycle"),
    ("amplitude_modulation", {"modulation_depth": 5}, "modulation_depth"),
    ("modulated_noise", {"noise_type": "purple"}, "noise_type"),
    ("modulated_noise", {"noise_type": float("inf")}, "noise_type"),
    ("modulated_noise", {"seed": -1}, "seed must be"),
    ("modulated_noise", {"seed": True}, "seed must be"),
    ("modulated_noise", {"seed": 1.0}, "seed must be"),
    ("modulated_noise", {"seed": 2 ** 64}, "seed must be"),
    ("modulated_noise", {"min_freq": 20000}, "band rise"),
    ("modulated_noise", {"max_freq": 25000}, "band rise"),
    ("modulated_noise", {"modulation_depth": 2}, "band rise"),
    ("modulated_noise", {"modulation_freq": -1}, "band rise"),
    ("modulated_noise", {"sonic_vector": [1]}, "named sonic_vector"),
    ("modulated_noise", {"modulation_waveform_table": [1]},
     "named modulation_waveform_table"),
    ("spatial_motion", {"carrier_freq": 30000}, "Nyquist"),
    ("spatial_motion", {"motion_rate": -1}, "motion rate"),
    ("spatial_motion", {"dist": -1}, "dist and zeta"),
    ("spatial_motion", {"zeta": -1}, "dist and zeta"),
    ("spatial_motion", {"air_temp": 200}, "air_temp"),
    ("spatial_motion", {"theta1": float("nan")}, "theta1 must be"),
    ("spatial_motion", {"theta1": "north"}, "theta1 must be"),
])
def test_each_stimulus_refuses_arguments_it_cannot_render(name, parameters,
                                                          match):
    with pytest.raises(ValueError, match=match):
        described(name, **parameters)


def test_a_seed_may_be_a_numpy_integer():
    graph = described("modulated_noise", seed=np.uint64(2 ** 64 - 1))
    assert from_sstim_graph(graph).parameters["seed"] == 2 ** 64 - 1


@pytest.mark.parametrize("source", [
    "not_a_generator", music.noise, lambda **kw: None, 3,
])
def test_only_the_seven_stimuli_are_described(source):
    with pytest.raises(ValueError, match="one of the seven"):
        to_sstim_graph(source)


def test_a_function_that_only_shares_a_stimulus_name_is_refused():
    def binaural_beats(**kw):
        return stimuli.binaural_beats(**kw)

    with pytest.raises(ValueError, match="one of the seven"):
        to_sstim_graph(binaural_beats)


def test_a_stimulus_takes_no_plan_and_a_read_one_no_arguments():
    with pytest.raises(TypeError, match="plan is for a session"):
        to_sstim_graph("binaural_beats", plan=object())
    stimulus = from_sstim_graph(described())
    for arguments in ({"duration": 1}, {"sample_rate": 8000},
                      {"beat_freq": 4}):
        with pytest.raises(TypeError, match="its own arguments"):
            to_sstim_graph(stimulus, **arguments)


def test_a_hand_made_stimulus_is_checked_again_before_it_renders():
    with pytest.raises(ValueError, match="carrier must lie"):
        SstimStimulus("amplitude_modulation", {"carrier_freq": -3}, .04,
                      48000).render()
    with pytest.raises(ValueError, match="not a stimulus"):
        SstimStimulus("exec", {}, .04, 48000).render()
    with pytest.raises(TypeError, match="mapping"):
        SstimStimulus("binaural_beats", [], .04, 48000).render()


@pytest.mark.parametrize("kind,match", [
    ("no-root", "exactly one SSTIM StimulusSpecification"),
    ("extra-root", "exactly one SSTIM StimulusSpecification"),
    ("renamed-root", "not named ...specification"),
    ("no-generator", "expected exactly one .*generator"),
    ("two-generators", "expected exactly one .*generator"),
    ("code-generator", "not a stimulus a description can be written"),
    ("huge-parameters", "longer than any stimulus"),
    ("bad-json", "not JSON"),
    ("json-array", "must be a JSON object"),
    ("unknown-parameter", "named volume"),
    ("wrong-rate-type", "an xsd:integer"),
    ("no-channel", "has no channel"),
    ("bad-channel-duration", "duration is not a number"),
    ("one-channel-gone", "not those the package writes"),
    ("signal-rate", "not those the package writes"),
    ("signal-shape", "not those the package writes"),
    ("regime", "not those the package writes"),
    ("channel-duration", "not those the package writes"),
    ("rendering-signal", "not those the package writes"),
    ("mechanism", "not those the package writes"),
    ("parameter", "not those the package writes"),
    ("presence", "not those the package writes"),
    ("placement", "not those the package writes"),
    ("same-placement", "not those the package writes"),
    ("rendering-carrier", "not those the package writes"),
    ("channel-carrier", "not those the package writes"),
    ("modality", "not those the package writes"),
    ("medium", "not those the package writes"),
    ("signal-class", "not those the package writes"),
    ("extra-sstim-term", "unexpected .*hasGainLevel"),
    ("extra-package-term", "unexpected .*auditNote"),
])
def test_the_reader_refuses_a_description_it_did_not_write(kind, match):
    graph = described(carrier_freq=200, beat_freq=10)
    root = root_of(graph)
    signal = next(graph.objects(root, SSTIM.hasSignal))
    left, right = channels_of(graph)
    rendering = next(graph.objects(left, SSTIM.hasSignalRendering))
    if kind == "no-root":
        graph.remove((root, RDF.type, SSTIM.StimulusSpecification))
    elif kind == "extra-root":
        graph.add((URIRef(BASE + "other"), RDF.type,
                   SSTIM.StimulusSpecification))
    elif kind == "renamed-root":
        graph.remove((root, RDF.type, SSTIM.StimulusSpecification))
        graph.add((URIRef(BASE + "elsewhere"), RDF.type,
                   SSTIM.StimulusSpecification))
    elif kind == "no-generator":
        swap(graph, root, MUSIC.generator, None)
    elif kind == "two-generators":
        graph.add((root, MUSIC.generator, Literal("monaural_beats")))
    elif kind == "code-generator":
        swap(graph, root, MUSIC.generator,
             Literal("__import__('os').system"))
    elif kind in ("huge-parameters", "bad-json", "json-array",
                  "unknown-parameter"):
        swap(graph, root, MUSIC.parametersJson, Literal({
            "huge-parameters": "a" * 2049, "bad-json": "{",
            "json-array": "[]", "unknown-parameter": '{"volume": 1}'}[kind]))
    elif kind == "wrong-rate-type":
        swap(graph, root, MUSIC.sampleRateHz,
             Literal("48000.0", datatype=XSD.decimal))
    elif kind == "no-channel":
        graph.remove((root, SSTIM.hasStimulusChannel, None))
    elif kind == "bad-channel-duration":
        for channel in (left, right):
            swap(graph, channel, SSTIM.channelDurationSeconds,
                 Literal("long"))
    elif kind == "one-channel-gone":
        graph.remove((root, SSTIM.hasStimulusChannel, right))
    elif kind == "signal-rate":
        swap(graph, signal, SSTIM.hzMin, Literal(33))
    elif kind == "signal-shape":
        swap(graph, signal, SSTIM.hasSignalShape, SSTIM_V.shapeSquare)
    elif kind == "regime":
        swap(graph, root, SSTIM.stimulusRegime, Literal("adaptive"))
    elif kind == "channel-duration":
        swap(graph, right, SSTIM.channelDurationSeconds,
             Literal(".2", datatype=XSD.decimal))
    elif kind == "rendering-signal":
        swap(graph, rendering, SSTIM.rendersSignal, URIRef(BASE + "other"))
    elif kind == "mechanism":
        swap(graph, rendering, SSTIM.hasRenderingMechanism,
             SSTIM_V.mechanismFrequencyModulation)
    elif kind == "parameter":
        swap(graph, rendering, SSTIM.rendersOntoParameter,
             SSTIM_V.paramFrequency)
    elif kind == "presence":
        swap(graph, rendering, SSTIM.hasRenderingPresence,
             SSTIM_V.presencePhysical)
    elif kind == "placement":
        swap(graph, left, SSTIM_EX.hasBodyPlacement, SSTIM_EX.placementEyes)
    elif kind == "same-placement":
        swap(graph, right, SSTIM_EX.hasBodyPlacement,
             SSTIM_EX.placementEarLeft)
    elif kind == "rendering-carrier":
        swap(graph, rendering, SSTIM.renderingCarrierHz,
             Literal("999.0", datatype=XSD.decimal))
    elif kind == "channel-carrier":
        swap(graph, left, SSTIM_EX.hasCarrierFrequencyHz,
             Literal("999.0", datatype=XSD.decimal))
    elif kind == "modality":
        swap(graph, left, SSTIM_EX.perceivedModality,
             SSTIM_EX.modalityVisual)
    elif kind == "medium":
        swap(graph, left, SSTIM_EX.deliveryMedium, SSTIM_EX.mediumLight)
    elif kind == "signal-class":
        graph.remove((signal, RDF.type, SSTIM.StimulationSignal))
    elif kind == "extra-sstim-term":
        graph.add((left, SSTIM_EX.hasGainLevel, Literal(.5)))
    else:
        graph.add((root, MUSIC.auditNote, Literal("reviewed")))
    with pytest.raises(ValueError, match=match):
        from_sstim_graph(graph)


def test_annotations_in_other_vocabularies_are_left_alone():
    graph = described()
    root = root_of(graph)
    graph.add((root, RDFS.comment, Literal("reviewed")))
    graph.add((root, DCTERMS.creator, URIRef("https://example.org/me")))
    swap(graph, root, RDFS.label, Literal("Ein Schwebungston", lang="de"))
    graph.add((URIRef("https://example.org/elsewhere"), SSTIM.hzMin,
               Literal(3)))
    assert from_sstim_graph(graph) == from_sstim_graph(described())


def test_a_description_the_package_wrote_takes_no_controls():
    with pytest.raises(ValueError, match="takes no controls"):
        from_sstim_graph(described(), controls={"beat_freq": 10})
    assert from_sstim_graph(described(), controls={}).generator == (
        "binaural_beats")


def test_a_rate_or_length_given_for_one_must_be_its_own():
    graph = described(duration=.1, sample_rate=48000)
    assert from_sstim_graph(graph, sample_rate=48000, duration=.1)
    with pytest.raises(ValueError, match="fixes its sample rate at 48000"):
        from_sstim_graph(graph, sample_rate=44100)
    with pytest.raises(ValueError, match="fixes its sample rate"):
        from_sstim_graph(graph, sample_rate=True)
    with pytest.raises(ValueError, match="lasts 0.1 s"):
        from_sstim_graph(graph, duration=.2)
    with pytest.raises(ValueError, match="lasts 0.1 s"):
        from_sstim_graph(graph, duration="0.1")


def test_validation_calls_sstims_validator_on_the_full_profile(monkeypatch):
    calls = []
    fake = SimpleNamespace(validate=lambda graph, **kw: (
        calls.append((len(graph), kw)), SimpleNamespace(ok=True))[1])
    monkeypatch.setitem(sys.modules, "sstim", fake)
    assert validate_sstim(described()).ok
    assert validate_sstim(described(), version="0.18.0").ok
    assert [kw for _, kw in calls] == [
        {"profile": "full", "version": "0.19.0"},
        {"profile": "full", "version": "0.18.0"}]
    assert calls[0][0] == len(described())


def test_validation_says_how_to_install_the_validator(monkeypatch):
    original = builtins.__import__

    def fail_import(name, *args, **kwargs):
        if name == "sstim":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "sstim", raising=False)
    monkeypatch.setattr(builtins, "__import__", fail_import)
    with pytest.raises(ImportError, match="music\\[sstim\\]"):
        validate_sstim(described())


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SSTIM_LIVE") != "1",
                    reason="opt-in live SSTIM validation on one CI runner")
@pytest.mark.parametrize("name,parameters", CASES)
def test_every_description_conforms_to_sstims_full_profile(name,
                                                           parameters):
    pytest.importorskip("sstim")
    report = validate_sstim(described(name, **parameters))
    assert report.ok, str(report)


def test_the_seven_are_the_stimuli_module_s_seven():
    assert set(STIMULI) == {
        "binaural_beats", "monaural_beats", "isochronic_tones",
        "amplitude_modulation", "frequency_modulation", "modulated_noise",
        "spatial_motion"}
    assert all(callable(getattr(stimuli, name)) for name in STIMULI)
    assert isinstance(described(), Graph)
