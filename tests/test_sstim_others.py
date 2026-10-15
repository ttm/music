"""Descriptions another tool wrote, in SSTIM's terms alone.

The descriptions here use none of the package's terms, name their nodes
their own way, and write their numbers their own way. Whatever SSTIM
states -- carriers, rates, duty cycle, noise colour, length -- is read
from them; what it has no term for is given, and nothing is guessed.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdflib")
from rdflib import BNode, Graph, Literal, URIRef  # noqa: E402
from rdflib.namespace import RDF, RDFS, XSD  # noqa: E402

from music.stimulation import stimuli  # noqa: E402
from music.stimulation.sstim import (  # noqa: E402
    BEAT_PROFILE, MUSIC, SSTIM, SSTIM_EX, SSTIM_V, SstimBeat, SstimStimulus,
    from_sstim_graph, render_sstim, to_sstim_graph, validate_sstim,
)

FIXTURES = Path(__file__).parent / "fixtures"
CASES = [
    ("binaural_beats", {"carrier_freq": 220., "beat_freq": 9.}),
    ("monaural_beats", {"carrier_freq": 220., "beat_freq": 9.}),
    ("isochronic_tones", {"carrier_freq": 220., "pulse_rate": 9.,
                          "duty_cycle": .4, "ramp_duration": .003}),
    ("amplitude_modulation", {"carrier_freq": 220., "modulation_freq": 9.,
                              "modulation_depth": .7}),
    ("frequency_modulation", {"carrier_freq": 220., "modulation_freq": 9.,
                              "frequency_deviation": 14.}),
    ("modulated_noise", {"noise_type": "pink", "min_freq": 70.,
                         "max_freq": 9000., "modulation_freq": 9.,
                         "modulation_depth": .6, "seed": 27}),
    ("spatial_motion", {"carrier_freq": 220., "motion_rate": .5,
                        "theta1": 180., "theta2": 0., "dist": .1,
                        "zeta": .215, "air_temp": 20.}),
]
NEEDS = {
    "binaural_beats": ("sample_rate",),
    "monaural_beats": ("sample_rate",),
    "isochronic_tones": ("sample_rate", "ramp_duration"),
    "amplitude_modulation": ("sample_rate", "modulation_depth"),
    "frequency_modulation": ("sample_rate", "frequency_deviation"),
    "modulated_noise": ("sample_rate", "modulation_depth", "seed"),
    "spatial_motion": ("sample_rate", "theta1", "theta2", "dist", "zeta",
                       "air_temp"),
}
ORIGINAL = "https://external.example.net/instrument/alpha/"


def third_party(name, parameters, *, duration=.04, rate=48000):
    """The package's description, as another tool might have written it.

    Its own terms are dropped, its nodes renamed, its labels translated.
    """
    graph = to_sstim_graph(name, duration=duration, sample_rate=rate,
                           base=ORIGINAL, **parameters)
    out = Graph()

    def renamed(term):
        if isinstance(term, URIRef) and str(term).startswith(ORIGINAL):
            return URIRef(str(term).replace(ORIGINAL, "urn:independent:"))
        return term

    for subject, predicate, obj in graph:
        if str(predicate).startswith(str(MUSIC)):
            continue
        if predicate == RDFS.label:
            obj = Literal("Un'etichetta tradotta", lang="it")
        out.add((renamed(subject), predicate, renamed(obj)))
    return out


def unstated(name, parameters):
    """The arguments SSTIM has no term for."""
    return {key: value for key, value in parameters.items()
            if key in NEEDS[name]}


def handwritten_am():
    return Graph().parse(FIXTURES / "sstim_third_party_am.ttl")


AM = URIRef("urn:interop:third-party:am")
AM_SIGNAL = URIRef("urn:interop:third-party:am-signal")
AM_CHANNEL = URIRef("urn:interop:third-party:mono")
AM_RENDERING = URIRef("urn:interop:third-party:rendering")


def swap(graph, node, predicate, replacement):
    graph.remove((node, predicate, None))
    if replacement is not None:
        graph.add((node, predicate, replacement))


@pytest.mark.parametrize("name,parameters", CASES)
def test_each_stimulus_reads_from_sstims_terms_and_the_rest_given(
        name, parameters):
    graph = third_party(name, parameters)
    assert not [p for _, p, _ in graph if str(p).startswith(str(MUSIC))]
    stimulus = from_sstim_graph(graph, sample_rate=48000,
                                controls=unstated(name, parameters))
    assert stimulus.generator == name
    assert stimulus.duration == .04
    expected = getattr(stimuli, name)(duration=.04, sample_rate=48000,
                                      **parameters)
    np.testing.assert_array_equal(stimulus.render(), expected)
    np.testing.assert_array_equal(
        render_sstim(graph.serialize(format="turtle"), sample_rate=48000,
                     controls=unstated(name, parameters)), expected)


@pytest.mark.parametrize("name,parameters", CASES)
def test_what_is_missing_is_named(name, parameters):
    graph = third_party(name, parameters)
    with pytest.raises(ValueError) as caught:
        from_sstim_graph(graph)
    message = str(caught.value)
    assert f"is {name} in SSTIM's terms alone" in message
    assert "leave " + ", ".join(NEEDS[name]) + " unsaid" in message
    if len(NEEDS[name]) > 1:
        rest = ", ".join(NEEDS[name][1:])
        with pytest.raises(ValueError, match=f"leave {rest} unsaid: give "
                           f"controls for {rest}$"):
            from_sstim_graph(graph, sample_rate=48000)
    else:
        with pytest.raises(ValueError, match="unsaid: give sample_rate$"):
            from_sstim_graph(graph)


@pytest.mark.parametrize("name,parameters", CASES)
def test_controls_sstim_states_may_be_given_if_they_agree(name, parameters):
    graph = third_party(name, parameters)
    stimulus = from_sstim_graph(graph, sample_rate=48000,
                                controls=parameters)
    assert stimulus.parameters == from_sstim_graph(
        graph, sample_rate=48000,
        controls=unstated(name, parameters)).parameters


@pytest.mark.parametrize("name,parameters,control,value", [
    ("binaural_beats", {"carrier_freq": 220., "beat_freq": 9.},
     "carrier_freq", 221.),
    ("amplitude_modulation", {"carrier_freq": 220., "modulation_freq": 9.,
                              "modulation_depth": .7},
     "modulation_freq", 8),
    ("isochronic_tones", {"duty_cycle": .4, "ramp_duration": 0.},
     "duty_cycle", .5),
    ("modulated_noise", {"noise_type": "pink", "seed": 1,
                         "modulation_depth": 1.},
     "noise_type", "white"),
])
def test_a_given_control_that_contradicts_the_description_is_refused(
        name, parameters, control, value):
    graph = third_party(name, parameters)
    controls = {**unstated(name, parameters), control: value}
    with pytest.raises(ValueError, match=f"controls give {control} as "
                       f"{value!r}, but the description states"):
        from_sstim_graph(graph, sample_rate=48000, controls=controls)


def test_a_control_no_stimulus_takes_is_refused():
    graph = third_party(*CASES[0])
    with pytest.raises(ValueError, match="named volume"):
        from_sstim_graph(graph, sample_rate=48000, controls={"volume": 1})
    with pytest.raises(TypeError, match="mapping"):
        from_sstim_graph(graph, sample_rate=48000, controls=[("a", 1)])


def test_an_unmodulated_noise_needs_no_depth():
    graph = third_party("modulated_noise", {"modulation_freq": 0,
                                            "noise_type": "white"})
    stimulus = from_sstim_graph(graph, sample_rate=48000,
                                controls={"seed": 4})
    assert stimulus.parameters["modulation_freq"] == 0
    assert stimulus.parameters["noise_type"] == "white"
    np.testing.assert_array_equal(stimulus.render(), stimuli.modulated_noise(
        noise_type="white", modulation_freq=0, duration=.04,
        sample_rate=48000, seed=4))


@pytest.mark.parametrize("noise_type", ["black", -4.5])
def test_a_noise_colour_sstim_cannot_name_is_given(noise_type):
    graph = third_party("modulated_noise", {"noise_type": noise_type})
    with pytest.raises(ValueError, match="noise_type"):
        from_sstim_graph(graph, sample_rate=48000,
                         controls={"seed": 1, "modulation_depth": 1.})
    stimulus = from_sstim_graph(graph, sample_rate=48000, controls={
        "seed": 1, "modulation_depth": 1., "noise_type": noise_type})
    assert stimulus.parameters["noise_type"] == noise_type


def test_a_noise_pattern_the_package_does_not_render_is_refused():
    graph = third_party("modulated_noise", {"noise_type": "pink"})
    channel = next(graph.subjects(SSTIM_EX.hasStimulusPattern, None))
    swap(graph, channel, SSTIM_EX.hasStimulusPattern,
         SSTIM_EX.audioNoiseGrey)
    with pytest.raises(ValueError, match="pattern sstim-ex:audioNoiseGrey"):
        from_sstim_graph(graph, sample_rate=48000, controls={
            "seed": 1, "modulation_depth": 1., "noise_type": "pink"})


def test_a_length_the_channels_do_not_state_is_given():
    graph = handwritten_am()
    swap(graph, AM_CHANNEL, SSTIM.channelDurationSeconds, None)
    with pytest.raises(ValueError, match="leave duration unsaid"):
        from_sstim_graph(graph, sample_rate=48000,
                         controls={"modulation_depth": .7})
    stimulus = from_sstim_graph(graph, sample_rate=48000, duration=.5,
                                controls={"modulation_depth": .7})
    assert stimulus.duration == .5


def test_a_length_given_must_agree_with_the_channels():
    with pytest.raises(ValueError, match="duration is given as 0.5, but "
                       "the description states 0.04"):
        from_sstim_graph(handwritten_am(), sample_rate=48000, duration=.5,
                         controls={"modulation_depth": .7})
    assert from_sstim_graph(handwritten_am(), sample_rate=48000,
                            duration=.04, controls={"modulation_depth": .7})


def test_channels_lasting_different_times_are_refused():
    graph = third_party(*CASES[0])
    channel = next(graph.subjects(SSTIM_EX.hasBodyPlacement,
                                  SSTIM_EX.placementEarRight))
    swap(graph, channel, SSTIM.channelDurationSeconds, Literal(2.))
    with pytest.raises(ValueError, match="last different times"):
        from_sstim_graph(graph, sample_rate=48000)


@pytest.mark.parametrize("stem", ["am", "noise", "spatial"])
def test_the_hand_written_fixtures_render_with_their_arguments(stem):
    arguments = json.loads((FIXTURES / f"sstim_third_party_{stem}"
                            ".render.json").read_text())
    stimulus = from_sstim_graph(FIXTURES / f"sstim_third_party_{stem}.ttl",
                                **arguments)
    assert isinstance(stimulus, SstimStimulus)
    assert stimulus.sample_rate == 48000 and stimulus.duration == .04
    np.testing.assert_array_equal(
        stimulus.render(), getattr(stimuli, stimulus.generator)(
            duration=.04, sample_rate=48000, **stimulus.parameters))


def test_the_hand_written_am_states_its_carrier_and_rate():
    stimulus = from_sstim_graph(handwritten_am(), sample_rate=48000,
                                controls={"modulation_depth": .7})
    assert stimulus.parameters == {"carrier_freq": 220.,
                                   "modulation_freq": 9.,
                                   "modulation_depth": .7}


def test_a_channel_may_state_its_carrier_instead_of_its_rendering():
    graph = handwritten_am()
    swap(graph, AM_RENDERING, SSTIM.renderingCarrierHz, None)
    with pytest.raises(ValueError, match="leave carrier_freq"):
        from_sstim_graph(graph, sample_rate=48000,
                         controls={"modulation_depth": .7})
    graph.add((AM_CHANNEL, SSTIM_EX.hasCarrierFrequencyHz, Literal(230.)))
    assert from_sstim_graph(graph, sample_rate=48000, controls={
        "modulation_depth": .7}).parameters["carrier_freq"] == 230.


def test_one_binaural_carrier_and_the_beat_give_the_other():
    graph = third_party(*CASES[0])
    right = next(graph.subjects(SSTIM_EX.hasBodyPlacement,
                                SSTIM_EX.placementEarRight))
    left = next(graph.subjects(SSTIM_EX.hasBodyPlacement,
                               SSTIM_EX.placementEarLeft))
    for channel, keep in ((right, left), (left, right)):
        stripped = Graph() + graph
        rendering = next(stripped.objects(channel, SSTIM.hasSignalRendering))
        swap(stripped, rendering, SSTIM.renderingCarrierHz, None)
        swap(stripped, channel, SSTIM_EX.hasCarrierFrequencyHz, None)
        assert from_sstim_graph(stripped, sample_rate=48000).parameters == (
            {"carrier_freq": 220., "beat_freq": 9.})
    for channel in (left, right):
        rendering = next(graph.objects(channel, SSTIM.hasSignalRendering))
        swap(graph, rendering, SSTIM.renderingCarrierHz, None)
        swap(graph, channel, SSTIM_EX.hasCarrierFrequencyHz, None)
    with pytest.raises(ValueError, match="leave carrier_freq"):
        from_sstim_graph(graph, sample_rate=48000)


def test_carriers_that_contradict_the_beat_are_refused():
    graph = third_party(*CASES[0])
    right = next(graph.subjects(SSTIM_EX.hasBodyPlacement,
                                SSTIM_EX.placementEarRight))
    rendering = next(graph.objects(right, SSTIM.hasSignalRendering))
    swap(graph, rendering, SSTIM.renderingCarrierHz, Literal(226.))
    swap(graph, right, SSTIM_EX.hasCarrierFrequencyHz, Literal(226.))
    with pytest.raises(ValueError, match="carriers, 215.5 and 226 Hz, are "
                       "not its 9 Hz beat apart"):
        from_sstim_graph(graph, sample_rate=48000)


@pytest.mark.parametrize("term,value", [
    (SSTIM_EX.hasModulationFrequencyHz, 9.),
    (SSTIM_EX.hasFrequencyHz, 215.5),
    (SSTIM_EX.hasFrequencyHz, 9.),
])
def test_a_more_general_frequency_term_is_no_contradiction(term, value):
    graph = third_party(*CASES[0])
    left = next(graph.subjects(SSTIM_EX.hasBodyPlacement,
                               SSTIM_EX.placementEarLeft))
    graph.add((left, term, Literal(value)))
    assert from_sstim_graph(graph, sample_rate=48000).generator == (
        "binaural_beats")


@pytest.mark.parametrize("term,value", [
    (SSTIM_EX.hasModulationFrequencyHz, 10.),
    (SSTIM_EX.hasFrequencyHz, 500.),
    (SSTIM_EX.hasToneFrequencyHz, 215.5),
    (SSTIM_EX.hasDutyCycle, .5),
])
def test_a_frequency_the_package_does_not_render_is_refused(term, value):
    graph = third_party(*CASES[0])
    left = next(graph.subjects(SSTIM_EX.hasBodyPlacement,
                               SSTIM_EX.placementEarLeft))
    graph.add((left, term, Literal(value)))
    with pytest.raises(ValueError, match="which the package does not "
                       "render"):
        from_sstim_graph(graph, sample_rate=48000)


@pytest.mark.parametrize("node,term,value", [
    (AM, SSTIM.describesStimulation, URIRef("urn:process")),
    (AM, SSTIM.hasStimulationTarget, URIRef("urn:listener")),
    (AM_SIGNAL, SSTIM.signalWithinBand, SSTIM_V.alpha10),
    (AM_CHANNEL, SSTIM_EX.channelRole, URIRef("urn:role")),
    (AM_CHANNEL, SSTIM_EX.hasComfortBoundary, URIRef("urn:boundary")),
    (AM_CHANNEL, SSTIM_EX.requiresDeviceCapability, URIRef("urn:phones")),
    (AM, URIRef("https://other.example/note"), Literal("a note")),
])
def test_what_says_nothing_about_the_sound_is_left_alone(node, term, value):
    graph = handwritten_am()
    graph.add((node, term, value))
    assert from_sstim_graph(graph, sample_rate=48000, controls={
        "modulation_depth": .7}).generator == "amplitude_modulation"


@pytest.mark.parametrize("node,term,value", [
    (AM_CHANNEL, SSTIM_EX.hasGainLevel, Literal(.5)),
    (AM_CHANNEL, SSTIM_EX.hasPhaseOffset, Literal(.25)),
    (AM_CHANNEL, SSTIM.soundPressureLevelDb, Literal(65.)),
    (AM_CHANNEL, SSTIM_EX.pitchShiftCents, Literal(-31.77)),
    (AM_SIGNAL, SSTIM.hasAdaptationPolicy, URIRef("urn:policy")),
    (AM_SIGNAL, SSTIM.signalSourceAsset, URIRef("urn:recording")),
    (AM_RENDERING, URIRef("https://w3id.org/sstim/ecosystem#x"),
     Literal(1)),
])
def test_what_would_change_the_sound_unread_is_refused(node, term, value):
    graph = handwritten_am()
    graph.add((node, term, value))
    with pytest.raises(ValueError, match="cannot render as stated"):
        from_sstim_graph(graph, sample_rate=48000,
                         controls={"modulation_depth": .7})


@pytest.mark.parametrize("tamper,match", [
    ("no-root", "exactly one SSTIM StimulusSpecification"),
    ("second-root", "exactly one SSTIM StimulusSpecification"),
    ("no-regime", "states no sstim:stimulusRegime"),
    ("second-regime", "more than one sstim:stimulusRegime"),
    ("regime", "stimulusRegime is 'adaptive', not 'determinate'"),
    ("no-signals", "states no sstim:hasSignal"),
    ("no-channels", "states no sstim:hasStimulusChannel"),
    ("untyped-signal", "the sstim-v:shapeSine signal is not typed "
     "sstim:StimulationSignal"),
    ("literal-signal", "a signal states no sstim:hasSignalShape"),
    ("extra-class", "also typed sstim:SignalRendering"),
    ("untyped-channel", "the sstim-ex:placementEars channel is not typed "
     "sstim-ex:StimulusChannel"),
    ("extra-channel", "a channel states no sstim-ex:hasBodyPlacement"),
    ("untyped-rendering", "is not typed sstim:SignalRendering"),
    ("no-shape", "states no sstim:hasSignalShape"),
    ("literal-shape", "none of the package's stimuli"),
    ("no-rate", "states no sstim:hzMin"),
    ("text-rate", "not a finite number"),
    ("nan-rate", "not a finite number"),
    ("iri-rate", "not a finite number"),
    ("boolean-rate", "not a finite number"),
    ("variable-rate", "spans 9 to 10 Hz, not 9 to 9"),
    ("no-placement", "states no sstim-ex:hasBodyPlacement"),
    ("no-renderings", "states no sstim:hasSignalRendering"),
    ("foreign-signal", "renders a signal the stimulus does not have"),
    ("second-carrier", "more than one sstim:renderingCarrierHz"),
    ("second-modality", "more than one sstim-ex:perceivedModality"),
    ("no-mechanism", "states no sstim:hasRenderingMechanism"),
    ("unrenderable-pair", "none of the package's stimuli: it renders "
     "sstim-v:mechanismMonauralBeat with signals of shape "
     "sstim-v:shapeSquare"),
    ("unknown-mechanism", "none of the package's stimuli"),
    ("parameter", "with parameter sstim-v:paramFrequency"),
    ("presence", "with presence sstim-v:presencePerceptual"),
    ("placement", "channels are at sstim-ex:placementEarLeft, not "
     "sstim-ex:placementEars"),
    ("modality", "modality is sstim-ex:modalityVisual"),
    ("medium", "medium is sstim-ex:mediumLight"),
    ("duration", "duration is given as 0.04, but the description states 1"),
    ("two-durations", "more than one sstim:channelDurationSeconds"),
    ("second-signal", "signals are of shape"),
    ("same-shape-twice", "two signals of shape sstim-v:shapeSine"),
    ("signal-twice", "renders its sstim-v:shapeSine signal twice"),
])
def test_a_description_the_package_would_not_render_is_refused(tamper,
                                                               match):
    graph = handwritten_am()
    other = URIRef("urn:interop:third-party:other")
    if tamper == "no-root":
        graph.remove((AM, RDF.type, SSTIM.StimulusSpecification))
    elif tamper == "second-root":
        graph.add((other, RDF.type, SSTIM.StimulusSpecification))
    elif tamper == "no-regime":
        swap(graph, AM, SSTIM.stimulusRegime, None)
    elif tamper == "second-regime":
        graph.add((AM, SSTIM.stimulusRegime, Literal("stochastic")))
    elif tamper == "regime":
        swap(graph, AM, SSTIM.stimulusRegime, Literal("adaptive"))
    elif tamper == "no-signals":
        swap(graph, AM, SSTIM.hasSignal, None)
    elif tamper == "no-channels":
        swap(graph, AM, SSTIM.hasStimulusChannel, None)
    elif tamper == "untyped-signal":
        graph.remove((AM_SIGNAL, RDF.type, SSTIM.StimulationSignal))
    elif tamper == "literal-signal":
        swap(graph, AM, SSTIM.hasSignal, Literal("signal"))
    elif tamper == "extra-class":
        graph.add((AM_SIGNAL, RDF.type, SSTIM.SignalRendering))
    elif tamper == "untyped-channel":
        graph.remove((AM_CHANNEL, RDF.type, SSTIM_EX.StimulusChannel))
    elif tamper == "extra-channel":
        graph.add((AM, SSTIM.hasStimulusChannel, other))
    elif tamper == "untyped-rendering":
        graph.remove((AM_RENDERING, RDF.type, SSTIM.SignalRendering))
    elif tamper == "no-shape":
        swap(graph, AM_SIGNAL, SSTIM.hasSignalShape, None)
    elif tamper == "literal-shape":
        swap(graph, AM_SIGNAL, SSTIM.hasSignalShape, Literal("shapeSine"))
    elif tamper == "no-rate":
        swap(graph, AM_SIGNAL, SSTIM.hzMin, None)
    elif tamper == "text-rate":
        swap(graph, AM_SIGNAL, SSTIM.hzMin, Literal("invalid"))
    elif tamper == "nan-rate":
        swap(graph, AM_SIGNAL, SSTIM.hzMin, Literal(float("nan")))
    elif tamper == "iri-rate":
        swap(graph, AM_SIGNAL, SSTIM.hzMin, URIRef("urn:not-a-number"))
    elif tamper == "boolean-rate":
        swap(graph, AM_SIGNAL, SSTIM.hzMin, Literal(True))
    elif tamper == "variable-rate":
        swap(graph, AM_SIGNAL, SSTIM.hzMax, Literal(10.))
    elif tamper == "no-placement":
        swap(graph, AM_CHANNEL, SSTIM_EX.hasBodyPlacement, None)
    elif tamper == "no-renderings":
        swap(graph, AM_CHANNEL, SSTIM.hasSignalRendering, None)
    elif tamper == "foreign-signal":
        swap(graph, AM_RENDERING, SSTIM.rendersSignal, other)
    elif tamper == "second-carrier":
        graph.add((AM_RENDERING, SSTIM.renderingCarrierHz, Literal(221.)))
    elif tamper == "second-modality":
        graph.add((AM_CHANNEL, SSTIM_EX.perceivedModality,
                   SSTIM_EX.modalityTactile))
    elif tamper == "no-mechanism":
        swap(graph, AM_RENDERING, SSTIM.hasRenderingMechanism, None)
    elif tamper == "unrenderable-pair":
        swap(graph, AM_RENDERING, SSTIM.hasRenderingMechanism,
             SSTIM_V.mechanismMonauralBeat)
        swap(graph, AM_SIGNAL, SSTIM.hasSignalShape, SSTIM_V.shapeSquare)
    elif tamper == "unknown-mechanism":
        swap(graph, AM_RENDERING, SSTIM.hasRenderingMechanism,
             URIRef("urn:mechanism"))
    elif tamper == "parameter":
        swap(graph, AM_RENDERING, SSTIM.rendersOntoParameter,
             SSTIM_V.paramFrequency)
    elif tamper == "presence":
        swap(graph, AM_RENDERING, SSTIM.hasRenderingPresence,
             SSTIM_V.presencePerceptual)
    elif tamper == "placement":
        swap(graph, AM_CHANNEL, SSTIM_EX.hasBodyPlacement,
             SSTIM_EX.placementEarLeft)
    elif tamper == "modality":
        swap(graph, AM_CHANNEL, SSTIM_EX.perceivedModality,
             SSTIM_EX.modalityVisual)
    elif tamper == "medium":
        swap(graph, AM_CHANNEL, SSTIM_EX.deliveryMedium,
             SSTIM_EX.mediumLight)
    elif tamper == "duration":
        swap(graph, AM_CHANNEL, SSTIM.channelDurationSeconds, Literal(1.))
    elif tamper == "two-durations":
        graph.add((AM_CHANNEL, SSTIM.channelDurationSeconds, Literal(1.)))
    elif tamper in ("second-signal", "same-shape-twice"):
        graph.add((AM, SSTIM.hasSignal, other))
        for p, o in ((RDF.type, SSTIM.StimulationSignal),
                     (SSTIM.hasSignalShape, SSTIM_V.shapeTriangle
                      if tamper == "second-signal" else SSTIM_V.shapeSine),
                     (SSTIM.hzMin, Literal(1.)), (SSTIM.hzMax, Literal(1.))):
            graph.add((other, p, o))
    elif tamper == "signal-twice":
        graph.add((AM_CHANNEL, SSTIM.hasSignalRendering, other))
        for p, o in ((RDF.type, SSTIM.SignalRendering),
                     (SSTIM.rendersSignal, AM_SIGNAL)):
            graph.add((other, p, o))
    with pytest.raises(ValueError, match=match):
        from_sstim_graph(graph, sample_rate=48000, duration=.04,
                         controls={"modulation_depth": .7})


def test_nodes_may_be_blank():
    graph = Graph()
    names = {}
    for triple in handwritten_am():
        graph.add(tuple(names.setdefault(term, BNode())
                        if isinstance(term, URIRef)
                        and str(term).startswith("urn:") else term
                        for term in triple))
    assert from_sstim_graph(graph, sample_rate=48000, controls={
        "modulation_depth": .7}).generator == "amplitude_modulation"


def authored_beat(stereo=True):
    """A beat in SSTIM's terms alone, written without the package."""
    def channel(name, placement, hz, mechanism, presence):
        return f"""
<urn:portable:{name}> a sstim-ex:StimulusChannel ;
    rdfs:label "{name} audio channel" ;
    sstim-ex:perceivedModality sstim-ex:modalityAuditory ;
    sstim-ex:deliveryMedium sstim-ex:mediumAirConductedSound ;
    sstim-ex:hasBodyPlacement sstim-ex:{placement} ;
    sstim:channelDurationSeconds 1.0 ;
    sstim:hasSignalRendering <urn:portable:r-{name}> .
<urn:portable:r-{name}> a sstim:SignalRendering ;
    rdfs:label "{name} beat rendering" ;
    sstim:rendersSignal <urn:portable:signal> ;
    sstim:rendersOntoParameter sstim-v:paramAmplitude ;
    sstim:hasRenderingMechanism sstim-v:{mechanism} ;
    sstim:hasRenderingPresence sstim-v:{presence} ;
    sstim:renderingCarrierHz {hz} .
"""
    if stereo:
        body = (channel("left", "placementEarLeft", "195.0",
                        "mechanismBinauralBeat", "presencePerceptual")
                + channel("right", "placementEarRight", "205.0",
                          "mechanismBinauralBeat", "presencePerceptual"))
        channels = "<urn:portable:left>, <urn:portable:right>"
    else:
        body = channel("mono", "placementEars", "200.0",
                       "mechanismMonauralBeat", "presencePhysical")
        channels = "<urn:portable:mono>"
    return Graph().parse(data=f"""
@prefix sstim: <https://w3id.org/sstim#> .
@prefix sstim-v: <https://w3id.org/sstim/vocab#> .
@prefix sstim-ex: <https://w3id.org/sstim/exposure#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<urn:portable:spec> a sstim:StimulusSpecification ;
    rdfs:label "independent beat" ;
    sstim:stimulusRegime "determinate" ;
    sstim:hasSignal <urn:portable:signal> ;
    sstim:hasStimulusChannel {channels} .
<urn:portable:signal> a sstim:StimulationSignal ;
    rdfs:label "10 Hz sine beat" ;
    sstim:hasSignalShape sstim-v:shapeSine ;
    sstim:hzMin 10.0 ; sstim:hzMax 10.0 .
{body}
""", format="turtle")


@pytest.mark.parametrize("stereo", [True, False])
def test_a_beat_renders_by_the_named_convention(stereo):
    beat = from_sstim_graph(authored_beat(stereo), profile=BEAT_PROFILE,
                            sample_rate=48000)
    assert beat == SstimBeat("binaural_beats" if stereo else
                             "monaural_beats", 200., 10., 1., 48000)
    t = np.arange(48000) / 48000
    low, high = np.sin(2 * np.pi * 195 * t), np.sin(2 * np.pi * 205 * t)
    out = render_sstim(authored_beat(stereo), profile=BEAT_PROFILE,
                       sample_rate=48000)
    np.testing.assert_allclose(
        out, np.vstack((low, high)) if stereo else (low + high) / 2,
        atol=1e-11)


@pytest.mark.parametrize("stereo", [True, False])
def test_a_beat_also_renders_by_the_packages_own_generator(stereo):
    stimulus = from_sstim_graph(authored_beat(stereo), sample_rate=48000)
    assert stimulus.generator == ("binaural_beats" if stereo
                                  else "monaural_beats")
    assert stimulus.parameters == {"carrier_freq": 200., "beat_freq": 10.}


def test_the_packages_own_beat_renders_by_the_convention_too():
    graph = to_sstim_graph("binaural_beats", duration=.1, sample_rate=48000)
    beat = from_sstim_graph(graph, profile=BEAT_PROFILE)
    assert beat == SstimBeat("binaural_beats", 200., 10., .1, 48000)
    assert render_sstim(graph, profile=BEAT_PROFILE).shape == (2, 4800)
    with pytest.raises(ValueError, match="fixes its sample rate"):
        from_sstim_graph(graph, profile=BEAT_PROFILE, sample_rate=44100)


def test_the_convention_is_named_and_only_for_beats():
    with pytest.raises(ValueError, match="the only profile is"):
        from_sstim_graph(authored_beat(), profile="guess",
                         sample_rate=48000)
    with pytest.raises(ValueError, match="this description is "
                       "amplitude_modulation"):
        from_sstim_graph(handwritten_am(), profile=BEAT_PROFILE,
                         sample_rate=48000)
    with pytest.raises(ValueError, match="with no controls"):
        from_sstim_graph(authored_beat(), profile=BEAT_PROFILE,
                         sample_rate=48000, controls={"beat_freq": 10})
    with pytest.raises(ValueError, match="leave sample_rate unsaid"):
        from_sstim_graph(authored_beat(), profile=BEAT_PROFILE)
    with pytest.raises(ValueError, match="oversampling_factor is for"):
        render_sstim(authored_beat(), profile=BEAT_PROFILE,
                     sample_rate=48000, oversampling_factor=4)


def test_a_hand_made_beat_is_checked_before_it_renders():
    with pytest.raises(ValueError, match="binaural_beats or monaural"):
        SstimBeat("other", 200., 10., .1, 48000).render()
    with pytest.raises(ValueError, match="duration must be a positive"):
        SstimBeat("binaural_beats", 200., 10., 0., 48000).render()
    with pytest.raises(ValueError, match="both tones of the beat"):
        SstimBeat("monaural_beats", 23990., 30., .1, 48000).render()


@pytest.mark.parametrize("stereo,change,match", [
    (True, "regime", "'stochastic', not 'determinate'"),
    (True, "shape", "none of the package's stimuli"),
    (True, "beat-zero", "195 and 205 Hz, are not its 0 Hz beat apart"),
    (False, "beat-zero", "rate must be positive"),
    (True, "beat-variable", "spans 10 to 11 Hz"),
    (True, "one-channel", "channels are at sstim-ex:placementEarLeft, not"),
    (True, "parameter", "parameter sstim-v:paramFrequency"),
    (True, "duplicate-placements", "two channels at"),
    (True, "different-duration", "different times"),
    (True, "bad-duration", "duration must be a positive"),
    (True, "presence", "presence sstim-v:presencePhysical"),
    (True, "carrier-reversed", "220 and 205 Hz, are not its 10 Hz beat"),
    (True, "carrier-mismatch", "195 and 204 Hz, are not its 10 Hz beat"),
    (True, "carrier-nyquist", "both tones of the beat"),
    (False, "presence", "presence sstim-v:presencePerceptual"),
    (False, "placement", "channels are at sstim-ex:placementEarLeft"),
    (False, "carrier-nyquist", "both tones of the beat"),
])
def test_the_convention_refuses_a_beat_that_contradicts_itself(
        stereo, change, match):
    graph = authored_beat(stereo)
    root = URIRef("urn:portable:spec")
    signal = URIRef("urn:portable:signal")
    first = URIRef("urn:portable:left" if stereo else "urn:portable:mono")
    first_r = URIRef("urn:portable:r-left" if stereo
                     else "urn:portable:r-mono")
    second = URIRef("urn:portable:right")
    second_r = URIRef("urn:portable:r-right")
    if change == "regime":
        swap(graph, root, SSTIM.stimulusRegime, Literal("stochastic"))
    elif change == "shape":
        swap(graph, signal, SSTIM.hasSignalShape, SSTIM_V.shapeSquare)
    elif change == "beat-zero":
        swap(graph, signal, SSTIM.hzMin, Literal(0))
    elif change == "beat-variable":
        swap(graph, signal, SSTIM.hzMax, Literal(11))
    elif change == "one-channel":
        graph.remove((root, SSTIM.hasStimulusChannel, second))
    elif change == "parameter":
        swap(graph, first_r, SSTIM.rendersOntoParameter,
             SSTIM_V.paramFrequency)
    elif change == "duplicate-placements":
        swap(graph, second, SSTIM_EX.hasBodyPlacement,
             SSTIM_EX.placementEarLeft)
    elif change == "different-duration":
        swap(graph, second, SSTIM.channelDurationSeconds, Literal(2))
    elif change == "bad-duration":
        for channel in (first, second) if stereo else (first,):
            swap(graph, channel, SSTIM.channelDurationSeconds, Literal(-1))
    elif change == "presence":
        swap(graph, first_r, SSTIM.hasRenderingPresence,
             SSTIM_V.presencePhysical if stereo
             else SSTIM_V.presencePerceptual)
    elif change == "placement":
        swap(graph, first, SSTIM_EX.hasBodyPlacement,
             SSTIM_EX.placementEarLeft)
    elif change == "carrier-reversed":
        swap(graph, first_r, SSTIM.renderingCarrierHz, Literal(220))
    elif change == "carrier-mismatch":
        swap(graph, second_r, SSTIM.renderingCarrierHz, Literal(204))
    elif change == "carrier-nyquist":
        swap(graph, second_r if stereo else first_r,
             SSTIM.renderingCarrierHz, Literal(24000 if stereo else 23999))
        if stereo:
            swap(graph, first_r, SSTIM.renderingCarrierHz, Literal(23990))
    with pytest.raises(ValueError, match=match):
        from_sstim_graph(graph, profile=BEAT_PROFILE, sample_rate=48000)


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SSTIM_LIVE") != "1",
                    reason="opt-in live SSTIM validation on one CI runner")
@pytest.mark.parametrize("make", [
    lambda: authored_beat(True), lambda: authored_beat(False),
    handwritten_am,
    lambda: Graph().parse(FIXTURES / "sstim_third_party_noise.ttl"),
    lambda: Graph().parse(FIXTURES / "sstim_third_party_spatial.ttl"),
] + [lambda case=case: third_party(*case) for case in CASES])
def test_descriptions_written_elsewhere_conform_to_sstim(make):
    pytest.importorskip("sstim")
    report = validate_sstim(make())
    assert report.ok, str(report)


def test_the_decimals_another_tool_writes_are_read_as_numbers():
    graph = handwritten_am()
    swap(graph, AM_SIGNAL, SSTIM.hzMin, Literal("9", datatype=XSD.integer))
    swap(graph, AM_SIGNAL, SSTIM.hzMax, Literal("9.000e0",
                                                datatype=XSD.double))
    assert from_sstim_graph(graph, sample_rate=48000, controls={
        "modulation_depth": .7}).parameters["modulation_freq"] == 9.


def test_a_rendering_by_another_mechanism_is_refused():
    graph = Graph().parse(FIXTURES / "sstim_third_party_noise.ttl")
    swap(graph, URIRef("urn:interop:noise:envelope-rendering"),
         SSTIM.hasRenderingMechanism, SSTIM_V.mechanismFrequencyModulation)
    with pytest.raises(ValueError, match="renders its sstim-v:shapeSine "
                       "signal with mechanism "
                       "sstim-v:mechanismFrequencyModulation, not "
                       "sstim-v:mechanismAmplitudeModulation"):
        from_sstim_graph(graph, sample_rate=48000)


def test_a_carrier_on_a_directly_presented_signal_is_refused():
    graph = Graph().parse(FIXTURES / "sstim_third_party_spatial.ttl")
    graph.add((URIRef("urn:interop:spatial:l-audio"),
               SSTIM.renderingCarrierHz, Literal(220.)))
    with pytest.raises(ValueError, match="on a 220 Hz carrier, but the "
                       "package renders it on none"):
        from_sstim_graph(graph, sample_rate=48000)


def test_a_description_is_refused_for_what_it_says_before_what_it_lacks():
    graph = handwritten_am()
    swap(graph, AM_CHANNEL, SSTIM_EX.deliveryMedium, SSTIM_EX.mediumLight)
    with pytest.raises(ValueError, match="medium is sstim-ex:mediumLight"):
        from_sstim_graph(graph)


def test_when_a_stand_in_does_not_fit_what_is_missing_is_asked_for():
    graph = authored_beat()
    signal = URIRef("urn:portable:signal")
    for side in ("left", "right"):
        swap(graph, URIRef(f"urn:portable:r-{side}"),
             SSTIM.renderingCarrierHz, None)
    swap(graph, signal, SSTIM.hzMin, Literal(450.))
    swap(graph, signal, SSTIM.hzMax, Literal(450.))
    with pytest.raises(ValueError, match="leave sample_rate, carrier_freq "
                       "unsaid: give sample_rate, and controls for "
                       "carrier_freq$"):
        from_sstim_graph(graph)
    with pytest.raises(ValueError, match="carriers the description states; "
                       "this one states none"):
        from_sstim_graph(graph, profile=BEAT_PROFILE, sample_rate=48000)
    beat = from_sstim_graph(graph, sample_rate=48000,
                            controls={"carrier_freq": 1000.})
    assert beat.parameters == {"carrier_freq": 1000., "beat_freq": 450.}


def test_a_functional_term_stated_twice_is_refused():
    graph = handwritten_am()
    graph.add((AM_CHANNEL, SSTIM_EX.hasCarrierFrequencyHz, Literal(220.)))
    graph.add((AM_CHANNEL, SSTIM_EX.hasCarrierFrequencyHz, Literal(221.)))
    with pytest.raises(ValueError, match="more than one "
                       "sstim-ex:hasCarrierFrequencyHz"):
        from_sstim_graph(graph, sample_rate=48000,
                         controls={"modulation_depth": .7})


def test_an_isochronic_tone_without_its_duty_cycle_asks_for_it():
    graph = third_party("isochronic_tones", {"duty_cycle": .4})
    graph.remove((None, SSTIM_EX.hasDutyCycle, None))
    with pytest.raises(ValueError, match="leave duty_cycle, ramp_duration"):
        from_sstim_graph(graph, sample_rate=48000)
    stimulus = from_sstim_graph(graph, sample_rate=48000, controls={
        "duty_cycle": .4, "ramp_duration": 0.})
    assert stimulus.parameters["duty_cycle"] == .4


def spatial_without(shape):
    """The spatial fixture, less its signal of `shape` and its renderings."""
    graph = Graph().parse(FIXTURES / "sstim_third_party_spatial.ttl")
    gone = {"tone": "audio", "position": "position"}[shape]
    signal = URIRef(f"urn:interop:spatial:{shape}")
    for triple in list(graph):
        if signal in triple or any(
                str(term).endswith(f"-{gone}") for term in triple):
            graph.remove(triple)
    if shape == "position":
        for side in ("l", "r"):
            swap(graph, URIRef(f"urn:interop:spatial:{side}-audio"),
                 SSTIM.rendersOntoParameter, SSTIM_V.paramSpatialPosition)
    return graph


@pytest.mark.parametrize("shape,match", [
    ("tone", "signals are of shape sstim-v:shapeTriangle, not "
     "sstim-v:shapeSine, sstim-v:shapeTriangle"),
    ("position", "signals are of shape sstim-v:shapeSine, not "
     "sstim-v:shapeSine, sstim-v:shapeTriangle"),
])
def test_a_moving_source_without_its_tone_or_its_motion_is_refused(shape,
                                                                   match):
    with pytest.raises(ValueError, match=match):
        from_sstim_graph(spatial_without(shape), sample_rate=48000)


def test_a_channel_that_leaves_a_signal_unrendered_is_refused():
    graph = Graph().parse(FIXTURES / "sstim_third_party_noise.ttl")
    graph.remove((URIRef("urn:interop:noise:ears"),
                  SSTIM.hasSignalRendering,
                  URIRef("urn:interop:noise:envelope-rendering")))
    with pytest.raises(ValueError, match="renders signals of shape "
                       "sstim-v:shapeNoise, not sstim-v:shapeNoise, "
                       "sstim-v:shapeSine"):
        from_sstim_graph(graph, sample_rate=48000)
