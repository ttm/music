"""What the mutation audit of `music.stimulation.sstim` found unasserted.

The SSTIM tests read descriptions back and refused tampered ones, but
nothing fixed the form a description is written in -- the names of its
nodes, its labels, its parameters' JSON, its prefixes, the plan's digest
-- so a change to any of them would have left descriptions written
before it unreadable, unnoticed. The bounds each stimulus is checked
against were tested on one side only; a reader that took a statement
about another resource for one about its own passed; and the refusals
were checked for what they refused, not for where they said the problem
was. These tests assert each.
"""
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdflib")
from rdflib import Graph, Literal, URIRef  # noqa: E402
from rdflib.namespace import DCTERMS, PROV, RDF, RDFS, XSD  # noqa: E402

import music  # noqa: E402
from music.stimulation import stimuli  # noqa: E402
from music.stimulation.sstim import (  # noqa: E402
    BEAT_PROFILE, MUSIC, SSTIM, SSTIM_EX, SSTIM_V, SessionPlan,
    SstimCapability, SstimStimulus, from_sstim_graph, inspect_sstim,
    render_sstim, to_sstim_graph,
)
from music.stimulation.sstim._vocabulary import (  # noqa: E402
    MAX_TRIPLES, MAX_TURTLE_BYTES, difference, read_graph,
)

BASE = "https://example.org/format/"
FIXTURES = Path(__file__).parent / "fixtures"
AM = URIRef("urn:interop:third-party:am")
AM_SIGNAL = URIRef("urn:interop:third-party:am-signal")
AM_CHANNEL = URIRef("urn:interop:third-party:mono")
AM_RENDERING = URIRef("urn:interop:third-party:rendering")
CREATED = datetime(2026, 10, 11, tzinfo=timezone.utc)


def relative(term):
    return str(term)[len(BASE):] if str(term).startswith(BASE) else term


def swap(graph, node, predicate, replacement):
    graph.remove((node, predicate, None))
    if replacement is not None:
        graph.add((node, predicate, replacement))


def handwritten_am():
    return Graph().parse(FIXTURES / "sstim_third_party_am.ttl")


def read_am(graph, **kwargs):
    kwargs.setdefault("sample_rate", 48000)
    kwargs.setdefault("controls", {"modulation_depth": .7})
    return from_sstim_graph(graph, **kwargs)


# --------------------------------------------------------------------------
# the form a description is written in
# --------------------------------------------------------------------------

#: Each stimulus's nodes, the signal each rendering renders, each
#: channel's placement and every label, as the package writes them. A
#: description written today has to read the same way tomorrow, so these
#: are a format, not an implementation detail.
FORMAT = {
    "binaural_beats": (
        {}, {"rendering-1": "signal", "rendering-2": "signal"},
        {"channel-1": "placementEarLeft", "channel-2": "placementEarRight"},
        {"specification": "binaural_beats: 10 Hz",
         "signal": "10 Hz sine signal",
         "channel-1": "placementEarLeft", "channel-2": "placementEarRight",
         "rendering-1": "binaural_beats: placementEarLeft",
         "rendering-2": "binaural_beats: placementEarRight"}),
    "isochronic_tones": (
        {}, {"rendering-1": "signal"}, {"channel-1": "placementEars"},
        {"specification": "isochronic_tones: 10 Hz",
         "signal": "10 Hz square signal", "channel-1": "placementEars",
         "rendering-1": "isochronic_tones: placementEars"}),
    "modulated_noise": (
        {}, {"rendering-1-1": "signal-noise",
             "rendering-1-2": "signal-modulation"},
        {"channel-1": "placementEars"},
        {"specification": "modulated_noise: pink",
         "signal-noise": "pink noise band",
         "signal-modulation": "10 Hz noise amplitude modulation",
         "channel-1": "placementEars",
         "rendering-1-1": "placementEars: mechanismDirectPresentation",
         "rendering-1-2": "placementEars: mechanismAmplitudeModulation"}),
    "spatial_motion": (
        {}, {"rendering-1-1": "signal-tone", "rendering-1-2": "signal-motion",
             "rendering-2-1": "signal-tone", "rendering-2-2": "signal-motion"},
        {"channel-1": "placementEarLeft", "channel-2": "placementEarRight"},
        {"specification": "spatial_motion: 200 Hz, 0.2 Hz",
         "signal-tone": "200 Hz acoustic source tone",
         "signal-motion": "0.2 Hz triangular azimuth motion",
         "channel-1": "placementEarLeft", "channel-2": "placementEarRight",
         "rendering-1-1": "placementEarLeft: paramAmplitude",
         "rendering-1-2": "placementEarLeft: paramSpatialPosition",
         "rendering-2-1": "placementEarRight: paramAmplitude",
         "rendering-2-2": "placementEarRight: paramSpatialPosition"}),
}


@pytest.mark.parametrize("name", sorted(FORMAT))
def test_a_description_names_its_nodes_and_labels_them_as_it_always_has(
        name):
    arguments, links, placements, labels = FORMAT[name]
    graph = to_sstim_graph(name, base=BASE, **arguments)
    subjects = {relative(s) for s in graph.subjects()
                if str(s).startswith(BASE)}
    assert subjects == set(labels)
    assert {relative(s): relative(o) for s, o in
            graph.subject_objects(SSTIM.rendersSignal)} == links
    assert {relative(s): str(o).split("#")[1] for s, o in
            graph.subject_objects(SSTIM_EX.hasBodyPlacement)} == placements
    assert {relative(s): str(o) for s, o in
            graph.subject_objects(RDFS.label)} == labels


def test_an_unmodulated_noise_has_one_signal_and_one_rendering():
    graph = to_sstim_graph("modulated_noise", modulation_freq=0, base=BASE)
    assert {relative(s) for s in graph.subjects()
            if str(s).startswith(BASE)} == {
        "specification", "signal-noise", "channel-1", "rendering-1-1"}


@pytest.mark.parametrize("name,arguments,text", [
    ("binaural_beats", {}, '{"beat_freq":10.0,"carrier_freq":200.0}'),
    ("modulated_noise", {"seed": 7},
     '{"max_freq":15000.0,"min_freq":15.0,"modulation_depth":1.0,'
     '"modulation_freq":10.0,"noise_type":"pink","seed":7}'),
    ("modulated_noise", {"noise_type": -4.5},
     '{"max_freq":15000.0,"min_freq":15.0,"modulation_depth":1.0,'
     '"modulation_freq":10.0,"noise_type":-4.5,"seed":null}'),
])
def test_the_arguments_are_written_as_compact_sorted_json(name, arguments,
                                                          text):
    graph = to_sstim_graph(name, base=BASE, **arguments)
    root = URIRef(BASE + "specification")
    assert str(graph.value(root, MUSIC.parametersJson)) == text
    assert graph.value(root, MUSIC.sampleRateHz) == Literal(
        44100, datatype=XSD.integer)


def test_the_turtle_uses_the_vocabularies_own_prefixes():
    text = to_sstim_graph("binaural_beats").serialize(format="turtle")
    for prefix, space in (("sstim", SSTIM), ("sstim-v", SSTIM_V),
                          ("sstim-ex", SSTIM_EX), ("music-impl", MUSIC)):
        assert f"@prefix {prefix}: <{space}> ." in text


def test_a_modulation_slower_than_one_hertz_is_still_a_signal():
    graph = to_sstim_graph("modulated_noise", modulation_freq=.5, base=BASE)
    assert URIRef(BASE + "signal-modulation") in set(graph.subjects())
    assert {float(x) for x in graph.objects(
        None, SSTIM_EX.hasModulationFrequencyHz)} == {.5}


# --------------------------------------------------------------------------
# the plan's digest, computed here from its definition
# --------------------------------------------------------------------------

def documented_digest(session, plan, base):
    """sha256-music-plan-json-v1, from what the documentation says.

    A JSON object of the program's N-Triples, sorted, with the preset's
    IRI, the master volume as a plain decimal and the whole seconds; its
    keys sorted, no spaces, not escaped to ASCII; hashed as UTF-8.
    """
    program = to_sstim_graph(session, base=base)
    lines = sorted(line.strip() for line in
                   program.serialize(format="nt").splitlines()
                   if line.strip())
    volume = format(Decimal(str(float(plan.master_volume))).normalize(), "f")
    text = json.dumps({"program_ntriples": lines,
                       "preset_iri": plan.preset_iri,
                       "master_volume": volume,
                       "duration_seconds": int(round(session.duration))},
                      sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return sha256(text.encode("utf-8")).hexdigest()


def minute(sample_rate=8000, seconds=60.):
    session = music.StimulationSession(sample_rate=sample_rate)
    session.add(music.binaural_beats, duration=seconds, label="Ruhe – Atem")
    return session


@pytest.mark.parametrize("plan", [
    SessionPlan("https://example.org/presets/ruhe-ä", "Ruhe", CREATED, .25),
    SessionPlan("http://example.org/presets/plain", "Plain", CREATED, 0),
    SessionPlan("https://example.org/presets/loud", "Loud", CREATED, 1),
    SessionPlan("https://example.org/presets/hush", "Hush", CREATED, 1e-7),
])
def test_the_digest_is_the_one_its_name_defines(plan):
    graph = to_sstim_graph(minute(), plan=plan, base=BASE)
    root = URIRef(BASE + "session-specification")
    assert graph.value(root, SSTIM.configurationDigest) == Literal(
        documented_digest(minute(), plan, BASE), datatype=XSD.string)
    assert from_sstim_graph(graph).plan == plan


def test_a_plan_a_sample_longer_than_whole_seconds_is_refused():
    """7,199 s and a sample: inside the range, so only the whole seconds
    can refuse it, as they must however long the plan."""
    session = minute(sample_rate=192000, seconds=7199 + 1 / 192000)
    with pytest.raises(ValueError, match="whole number of seconds"):
        to_sstim_graph(session, plan=SessionPlan(
            "https://example.org/preset", "P", CREATED), base=BASE)


@pytest.mark.parametrize("volume", [1.5, -.1])
def test_a_master_volume_outside_zero_to_one_is_refused(volume):
    with pytest.raises(ValueError, match="master volume"):
        to_sstim_graph(minute(), plan=SessionPlan(
            "https://example.org/preset", "P", CREATED, volume), base=BASE)


# --------------------------------------------------------------------------
# the bounds each stimulus is checked against, from both sides
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,arguments", [
    ("amplitude_modulation", {"carrier_freq": .5, "modulation_freq": .25}),
    ("amplitude_modulation", {"modulation_freq": .5}),
    ("amplitude_modulation", {"modulation_depth": 0}),
    ("monaural_beats", {"carrier_freq": 1, "beat_freq": 1}),
    ("binaural_beats", {"carrier_freq": 200, "beat_freq": 150}),
    ("binaural_beats", {"carrier_freq": 23000, "beat_freq": 1000}),
    ("isochronic_tones", {"duty_cycle": 1}),
    ("modulated_noise", {"modulation_depth": 0}),
    ("modulated_noise", {"min_freq": 0}),
    ("modulated_noise", {"seed": 0}),
    ("spatial_motion", {"carrier_freq": .5}),
    ("spatial_motion", {"air_temp": -99.5}),
    ("spatial_motion", {"air_temp": 99.5}),
])
def test_a_value_at_the_edge_of_what_is_allowed_is_described(name,
                                                              arguments):
    graph = to_sstim_graph(name, duration=.01, sample_rate=48000,
                           **arguments)
    assert from_sstim_graph(graph).parameters == {
        **from_sstim_graph(graph).parameters, **arguments}


@pytest.mark.parametrize("name,arguments,match", [
    ("monaural_beats", {"carrier_freq": 5, "beat_freq": 10},
     "both tones of the beat"),
    ("isochronic_tones", {"duty_cycle": 1.5}, "duty_cycle"),
    ("amplitude_modulation", {"modulation_depth": 1.5}, "modulation_depth"),
    ("modulated_noise", {"min_freq": 300, "max_freq": 300}, "band rise"),
    ("spatial_motion", {"carrier_freq": 0}, "Nyquist"),
    ("spatial_motion", {"carrier_freq": 24000}, "Nyquist"),
    ("spatial_motion", {"dist": 0}, "dist and zeta"),
    ("spatial_motion", {"zeta": 0}, "dist and zeta"),
    ("spatial_motion", {"air_temp": -100}, "air_temp"),
    ("spatial_motion", {"air_temp": -100.5}, "air_temp"),
    ("spatial_motion", {"air_temp": 100}, "air_temp"),
])
def test_a_value_just_past_the_edge_is_refused(name, arguments, match):
    with pytest.raises(ValueError, match=match):
        to_sstim_graph(name, duration=.01, sample_rate=48000, **arguments)


@pytest.mark.parametrize("sample_rate", [192001, 48000.0, 7999])
def test_a_rate_outside_the_whole_numbers_allowed_is_refused(sample_rate):
    with pytest.raises(ValueError, match="sample_rate must be an integer"):
        to_sstim_graph("binaural_beats", sample_rate=sample_rate)


def test_the_highest_and_lowest_rates_are_described():
    for rate in (8000, 192000):
        graph = to_sstim_graph("binaural_beats", duration=.01,
                               sample_rate=rate)
        assert from_sstim_graph(graph).sample_rate == rate


def test_what_a_refusal_lists_is_listed_in_full():
    with pytest.raises(ValueError, match="they are binaural_beats, "
                       "monaural_beats, isochronic_tones, "
                       "amplitude_modulation, frequency_modulation, "
                       "modulated_noise, spatial_motion$"):
        to_sstim_graph("chirp")
    with pytest.raises(ValueError, match="named alpha, beta$"):
        to_sstim_graph("binaural_beats", beta=1, alpha=2)
    with pytest.raises(ValueError, match="one of black, blue, brown, pink, "
                       "violet, white or a slope"):
        to_sstim_graph("modulated_noise", noise_type="grey")


@pytest.mark.parametrize("base", [
    "https://example.org/a#", "http://example.org/a/"])
def test_a_base_of_ones_own_ending_in_a_hash_or_on_http_is_taken(base):
    graph = to_sstim_graph("binaural_beats", base=base)
    assert URIRef(base + "specification") in set(graph.subjects())


@pytest.mark.parametrize("base", [3, None.__class__, b"https://x/"])
def test_a_base_that_is_not_text_is_refused_as_such(base):
    with pytest.raises(ValueError, match="base must be your own"):
        to_sstim_graph("binaural_beats", base=base)


# --------------------------------------------------------------------------
# sessions at their bounds
# --------------------------------------------------------------------------

def fifty(extra=0):
    session = music.StimulationSession(sample_rate=8000)
    for _ in range(50 + extra):
        session.add(music.monaural_beats, duration=.01)
    return session


def test_fifty_phases_are_written_and_read():
    graph = to_sstim_graph(fifty(), base=BASE)
    assert len(from_sstim_graph(graph).phases) == 50


def test_a_fifty_first_phase_is_refused_both_ways():
    with pytest.raises(ValueError, match="from 1 to 50 phases"):
        to_sstim_graph(fifty(1), base=BASE)
    graph = to_sstim_graph(fifty(), base=BASE)
    graph.add((URIRef(BASE + "program"), MUSIC.hasPhase,
               URIRef(BASE + "phase-51")))
    with pytest.raises(ValueError, match="from 1 to 50 phases"):
        from_sstim_graph(graph)


def test_a_silent_phase_is_written_and_a_phase_of_no_length_is_not():
    session = music.StimulationSession(sample_rate=8000)
    session.add(music.binaural_beats, duration=.1, gain=0)
    assert from_sstim_graph(to_sstim_graph(session)).phases[0].gain == 0
    session.phases[0].duration = 0
    with pytest.raises(ValueError, match="a phase's duration must be "
                       "positive"):
        to_sstim_graph(session)


# --------------------------------------------------------------------------
# statements about other resources are not the description's
# --------------------------------------------------------------------------

ELSEWHERE = URIRef("https://example.org/elsewhere/thing")


def test_the_package_s_own_terms_about_other_resources_are_left_alone():
    graph = to_sstim_graph("binaural_beats", base=BASE)
    for predicate, value in (
            (MUSIC.generator, Literal("monaural_beats")),
            (MUSIC.parametersJson, Literal("{}")),
            (MUSIC.sampleRateHz, Literal(8000)),
            (SSTIM.hasStimulusChannel, URIRef(BASE + "channel-1")),
            (RDFS.seeAlso, SSTIM.StimulusSpecification)):
        graph.add((ELSEWHERE, predicate, value))
    assert from_sstim_graph(graph).generator == "binaural_beats"


def test_a_program_and_a_plan_read_only_their_own_statements():
    program = to_sstim_graph(minute(), base=BASE)
    for predicate, value in (
            (MUSIC.rampShape, Literal("linear")),
            (MUSIC.endRampSeconds, Literal(1.)),
            (MUSIC.hasPhase, URIRef(BASE + "phase-1")),
            (RDFS.seeAlso, MUSIC.StimulationProgram)):
        program.add((ELSEWHERE, predicate, value))
    assert from_sstim_graph(program).ramp_shape == "equal_power"
    plan = SessionPlan("https://example.org/preset", "P", CREATED, .5)
    planned = to_sstim_graph(minute(), plan=plan, base=BASE)
    for predicate, value in (
            (DCTERMS.created, Literal("2020-01-01T00:00:00+00:00",
                                      datatype=XSD.dateTime)),
            (SSTIM.masterVolume, Literal(.9)),
            (SSTIM.durationSeconds, Literal(61)),
            (SSTIM.referencesPreset, URIRef("https://example.org/other")),
            (MUSIC.hasProgram, URIRef(BASE + "program")),
            (RDFS.seeAlso, SSTIM.SessionSpecification)):
        planned.add((ELSEWHERE, predicate, value))
    planned.add((URIRef(plan.preset_iri), DCTERMS.created,
                 Literal("2020-01-01", datatype=XSD.date)))
    assert from_sstim_graph(planned).plan == plan


def test_sstims_terms_about_other_resources_are_not_the_stimulus_s():
    graph = handwritten_am()
    for predicate, value in (
            (SSTIM.stimulusRegime, Literal("stochastic")),
            (SSTIM.hasSignal, AM_SIGNAL),
            (SSTIM.hasStimulusChannel, AM_CHANNEL),
            (SSTIM_EX.hasStimulusPattern, SSTIM_EX.audioNoiseWhite),
            (RDFS.seeAlso, SSTIM.StimulusSpecification)):
        graph.add((ELSEWHERE, predicate, value))
    assert read_am(graph).parameters["modulation_freq"] == 9.


def test_a_class_or_a_link_in_another_vocabulary_is_an_annotation():
    graph = to_sstim_graph("binaural_beats", base=BASE)
    root = URIRef(BASE + "specification")
    graph.add((root, RDF.type, PROV.Entity))
    graph.add((root, RDFS.seeAlso, SSTIM_V.techBinauralBeats))
    assert from_sstim_graph(graph).generator == "binaural_beats"


def test_a_class_of_the_package_s_alone_makes_a_description_its_own():
    graph = handwritten_am()
    graph.add((ELSEWHERE, RDF.type, MUSIC.StimulationPhase))
    capability = inspect_sstim(graph)
    assert capability.written_by_package
    assert capability.kind == "unsupported"


# --------------------------------------------------------------------------
# where a refusal says the problem is
# --------------------------------------------------------------------------

CHANNEL = "the sstim-ex:placementEars channel"
RENDERING = "a rendering of the sstim-ex:placementEars channel"


@pytest.mark.parametrize("change,message", [
    (lambda g: swap(g, AM, SSTIM.stimulusRegime, None),
     "the stimulus states no sstim:stimulusRegime"),
    (lambda g: g.add((AM, RDF.type, SSTIM.Preset)),
     "the stimulus is also typed sstim:Preset, which the package does not "
     "render"),
    (lambda g: swap(g, AM, SSTIM.hasSignal, None),
     "the stimulus states no sstim:hasSignal"),
    (lambda g: swap(g, AM, SSTIM.hasStimulusChannel, None),
     "the stimulus states no sstim:hasStimulusChannel"),
    (lambda g: swap(g, AM_SIGNAL, SSTIM.hasSignalShape, None),
     "a signal states no sstim:hasSignalShape"),
    (lambda g: g.remove((AM_SIGNAL, RDF.type, SSTIM.StimulationSignal)),
     "the sstim-v:shapeSine signal is not typed sstim:StimulationSignal"),
    (lambda g: swap(g, AM_SIGNAL, SSTIM.hzMin, None),
     "the sstim-v:shapeSine signal states no sstim:hzMin"),
    (lambda g: swap(g, AM_SIGNAL, SSTIM.hzMax, Literal("fast")),
     "the sstim-v:shapeSine signal's sstim:hzMax is not a finite number: "
     '"fast"'),
    (lambda g: swap(g, AM_CHANNEL, SSTIM_EX.hasBodyPlacement, None),
     "a channel states no sstim-ex:hasBodyPlacement"),
    (lambda g: (g.add((AM_CHANNEL, SSTIM_EX.hasPhaseOffset, Literal(.2))),
                g.add((AM_CHANNEL, SSTIM_EX.hasGainLevel, Literal(.5)))),
     f"{CHANNEL} states sstim-ex:hasGainLevel, sstim-ex:hasPhaseOffset, "
     f"which the package cannot render as stated"),
    (lambda g: g.add((AM_CHANNEL, SSTIM_EX.perceivedModality,
                      SSTIM_EX.modalityTactile)),
     f"{CHANNEL} states more than one sstim-ex:perceivedModality"),
    (lambda g: g.add((AM_CHANNEL, SSTIM_EX.deliveryMedium,
                      SSTIM_EX.mediumLight)),
     f"{CHANNEL} states more than one sstim-ex:deliveryMedium"),
    (lambda g: swap(g, AM_CHANNEL, SSTIM.channelDurationSeconds,
                    Literal("long")),
     f"{CHANNEL}'s sstim:channelDurationSeconds is not a finite number: "
     '"long"'),
    (lambda g: g.add((AM_CHANNEL, SSTIM_EX.hasCarrierFrequencyHz,
                      URIRef("urn:high"))),
     f"{CHANNEL}'s sstim-ex:hasCarrierFrequencyHz is not a finite number: "
     "<urn:high>"),
    (lambda g: g.remove((AM_RENDERING, RDF.type, SSTIM.SignalRendering)),
     f"{RENDERING} is not typed sstim:SignalRendering"),
    (lambda g: swap(g, AM_RENDERING, SSTIM.rendersSignal, None),
     f"{RENDERING} states no sstim:rendersSignal"),
    (lambda g: swap(g, AM_RENDERING, SSTIM.rendersOntoParameter, None),
     f"{RENDERING} states no sstim:rendersOntoParameter"),
    (lambda g: swap(g, AM_RENDERING, SSTIM.hasRenderingMechanism, None),
     f"{RENDERING} states no sstim:hasRenderingMechanism"),
    (lambda g: swap(g, AM_RENDERING, SSTIM.hasRenderingPresence, None),
     f"{RENDERING} states no sstim:hasRenderingPresence"),
    (lambda g: swap(g, AM_RENDERING, SSTIM.renderingCarrierHz,
                    Literal(True)),
     f"{RENDERING}'s sstim:renderingCarrierHz is not a finite number: "
     '"true"^^<http://www.w3.org/2001/XMLSchema#boolean>'),
    (lambda g: (swap(g, AM_RENDERING, SSTIM.hasRenderingMechanism,
                     URIRef("urn:b")),
                g.add((AM_RENDERING, SSTIM.hasRenderingMechanism,
                       URIRef("urn:a")))),
     f"{RENDERING} states more than one sstim:hasRenderingMechanism"),
    (lambda g: swap(g, AM_RENDERING, SSTIM.hasRenderingMechanism,
                    URIRef("urn:b")),
     "the description's signals and mechanisms are none of the package's "
     "stimuli: it renders <urn:b> with signals of shape sstim-v:shapeSine"),
    (lambda g: swap(g, AM_CHANNEL, SSTIM_EX.hasBodyPlacement,
                    SSTIM_EX.placementEarLeft),
     "the description says what the package would not render for it as "
     "amplitude_modulation: its channels are at sstim-ex:placementEarLeft, "
     "not sstim-ex:placementEars"),
])
def test_a_refusal_says_where_the_problem_is(change, message):
    graph = handwritten_am()
    change(graph)
    with pytest.raises(ValueError) as caught:
        read_am(graph)
    assert str(caught.value) == message


def test_a_refusal_lists_the_mechanisms_and_shapes_it_found():
    graph = Graph().parse(FIXTURES / "sstim_third_party_noise.ttl")
    for rendering in ("source-rendering", "envelope-rendering"):
        swap(graph, URIRef(f"urn:interop:noise:{rendering}"),
             SSTIM.hasRenderingMechanism, URIRef(f"urn:{rendering}"))
    swap(graph, URIRef("urn:interop:noise:band"), SSTIM.hasSignalShape,
         SSTIM_V.shapeSquare)
    with pytest.raises(ValueError) as caught:
        from_sstim_graph(graph, sample_rate=48000)
    assert str(caught.value).endswith(
        "it renders <urn:envelope-rendering>, <urn:source-rendering> with "
        "signals of shape sstim-v:shapeSine, sstim-v:shapeSquare")


@pytest.mark.parametrize("stem,node,span,message", [
    ("spatial", "position", (.5, .7),
     "its sstim-v:shapeTriangle signal spans 0.5 to 0.7 Hz, not 0.5 to "
     "0.5"),
    ("spatial", "tone", (220., 230.),
     "its sstim-v:shapeSine signal spans 220 to 230 Hz, not 220 to 220"),
    ("noise", "mod", (9., 10.),
     "its sstim-v:shapeSine signal spans 9 to 10 Hz, not 9 to 9"),
])
def test_a_rate_is_the_lower_bound_of_a_point_signal(stem, node, span,
                                                     message):
    graph = Graph().parse(FIXTURES / f"sstim_third_party_{stem}.ttl")
    signal = URIRef(f"urn:interop:{stem}:{node}")
    swap(graph, signal, SSTIM.hzMin, Literal(span[0]))
    swap(graph, signal, SSTIM.hzMax, Literal(span[1]))
    with pytest.raises(ValueError) as caught:
        from_sstim_graph(graph, sample_rate=48000)
    assert message in str(caught.value)


def test_every_channel_s_problems_are_reported_in_order():
    graph = to_sstim_graph("binaural_beats", duration=.04, sample_rate=48000)
    for triple in list(graph):
        if str(triple[1]).startswith(str(MUSIC)):
            graph.remove(triple)
    left = URIRef("https://example.org/music/stimulus/channel-1")
    right = URIRef("https://example.org/music/stimulus/channel-2")
    graph.add((left, SSTIM_EX.hasToneFrequencyHz, Literal(195.)))
    graph.add((right, SSTIM_EX.hasToneFrequencyHz, Literal(205.)))
    with pytest.raises(ValueError) as caught:
        from_sstim_graph(graph, sample_rate=48000)
    assert str(caught.value).endswith(
        "the sstim-ex:placementEarLeft channel states "
        "sstim-ex:hasToneFrequencyHz 195, which the package does not render; "
        "the sstim-ex:placementEarRight channel states "
        "sstim-ex:hasToneFrequencyHz 205, which the package does not render")


def test_what_is_missing_is_asked_for_in_one_sentence():
    graph = handwritten_am()
    swap(graph, AM_CHANNEL, SSTIM.channelDurationSeconds, None)
    with pytest.raises(ValueError) as caught:
        from_sstim_graph(graph)
    assert str(caught.value) == (
        "the description is amplitude_modulation in SSTIM's terms alone, "
        "which leave sample_rate, duration, modulation_depth unsaid: give "
        "sample_rate and duration, and controls for modulation_depth")
    with pytest.raises(ValueError) as caught:
        from_sstim_graph(graph, controls={"modulation_depth": .7})
    assert str(caught.value).endswith(
        "leave sample_rate, duration unsaid: give sample_rate and duration")


def test_a_description_that_differs_names_what_is_missing_and_added():
    """Three of each, of the four that differ either way."""
    found = {(URIRef("urn:s"), URIRef("urn:p"), Literal(n))
             for n in range(6)}
    expected = {(URIRef("urn:s"), URIRef("urn:p"), Literal(n))
                for n in range(4, 10)}
    assert difference(found, expected) == (
        'missing <urn:s> <urn:p> "6"^^<http://www.w3.org/2001/XMLSchema#'
        'integer>; <urn:s> <urn:p> "7"^^<http://www.w3.org/2001/XMLSchema#'
        'integer>; <urn:s> <urn:p> "8"^^<http://www.w3.org/2001/XMLSchema#'
        'integer>, unexpected <urn:s> <urn:p> "0"^^<http://www.w3.org/2001/'
        'XMLSchema#integer>; <urn:s> <urn:p> "1"^^<http://www.w3.org/2001/'
        'XMLSchema#integer>; <urn:s> <urn:p> "2"^^<http://www.w3.org/2001/'
        'XMLSchema#integer>')


# --------------------------------------------------------------------------
# numbers that agree, and ones another tool states more generally
# --------------------------------------------------------------------------

def test_a_given_control_agrees_to_a_part_in_a_billion():
    stimulus = read_am(handwritten_am(), controls={
        "modulation_depth": .7, "modulation_freq": 9.000000001})
    assert stimulus.parameters["modulation_freq"] == 9.000000001
    with pytest.raises(ValueError, match="controls give modulation_freq"):
        read_am(handwritten_am(), controls={
            "modulation_depth": .7, "modulation_freq": 9.0001})


def test_near_zero_numbers_agree_to_a_billionth():
    graph = Graph().parse(FIXTURES / "sstim_third_party_noise.ttl")
    swap(graph, URIRef("urn:interop:noise:band"), SSTIM.hzMin, Literal(0.))
    stimulus = from_sstim_graph(graph, sample_rate=48000, controls={
        "noise_type": "pink", "modulation_depth": .6, "seed": 27,
        "min_freq": 1e-10})
    assert stimulus.parameters["min_freq"] == 1e-10


def test_a_moving_source_may_state_its_tone_as_a_plain_frequency():
    graph = Graph().parse(FIXTURES / "sstim_third_party_spatial.ttl")
    graph.add((URIRef("urn:interop:spatial:left"), SSTIM_EX.hasFrequencyHz,
               Literal(220.)))
    arguments = json.loads((FIXTURES / "sstim_third_party_spatial"
                            ".render.json").read_text())
    assert from_sstim_graph(graph, **arguments).generator == "spatial_motion"


# --------------------------------------------------------------------------
# what is given reaches the render
# --------------------------------------------------------------------------

def beat_without_length():
    graph = to_sstim_graph("binaural_beats", duration=.02, sample_rate=48000)
    for triple in list(graph):
        if (str(triple[1]).startswith(str(MUSIC))
                or triple[1] == SSTIM.channelDurationSeconds):
            graph.remove(triple)
    return graph


def test_a_length_given_reaches_a_beat_by_its_convention():
    graph = beat_without_length()
    beat = from_sstim_graph(graph, profile=BEAT_PROFILE, sample_rate=48000,
                            duration=.01)
    assert beat.duration == .01
    assert render_sstim(graph, profile=BEAT_PROFILE, sample_rate=48000,
                        duration=.01).shape == (2, 480)
    assert render_sstim(graph, sample_rate=48000,
                        duration=.01).shape == (2, 480)


def test_a_render_of_one_sample_is_allowed_one_sample():
    graph = to_sstim_graph("monaural_beats", duration=1 / 48000,
                           sample_rate=48000)
    assert render_sstim(graph, max_samples=1).shape == (1,)


def test_the_oversampling_factor_asked_for_is_the_one_used():
    pytest.importorskip("scipy")
    stimulus = SstimStimulus("isochronic_tones", {"carrier_freq": 500.,
                                                  "pulse_rate": 20.},
                             .02, 48000)
    np.testing.assert_array_equal(
        stimulus.render(oversampling_factor=2),
        music.render_oversampled(stimuli.isochronic_tones, factor=2,
                                 duration=.02, sample_rate=48000,
                                 carrier_freq=500., pulse_rate=20.))


def test_an_unsupported_description_needs_nothing_and_has_no_profile():
    assert inspect_sstim(Graph()) == SstimCapability(
        "unsupported", False, None, (), (),
        "expected exactly one SSTIM StimulusSpecification")


# --------------------------------------------------------------------------
# what is read as Turtle, and how much
# --------------------------------------------------------------------------

TRIPLE = "<urn:a> <urn:p> <urn:b> ."


@pytest.mark.parametrize("text", [
    "@prefix e: <urn:e:> . e:a e:p e:b .",
    "@base <urn:x> . " + TRIPLE,
    "PREFIX e: <urn:e:> e:a e:p e:b .",
    "BASE <urn:x> " + TRIPLE,
    TRIPLE,
    "# written elsewhere\n" + TRIPLE,
])
def test_turtle_on_one_line_or_after_a_comment_is_read_as_turtle(text):
    assert len(read_graph(text)) == 1


def test_turtle_of_exactly_the_byte_budget_is_read(tmp_path):
    text = TRIPLE + "\n#" + "x" * (MAX_TURTLE_BYTES - len(TRIPLE) - 2)
    assert len(text.encode("utf-8")) == MAX_TURTLE_BYTES
    assert len(read_graph(text)) == 1
    path = tmp_path / "budget.ttl"
    path.write_text(text, encoding="utf-8")
    assert len(read_graph(path)) == 1


def test_a_graph_of_exactly_the_triple_budget_is_read():
    graph = Graph()
    for i in range(MAX_TRIPLES):
        graph.add((URIRef("urn:many"), URIRef("urn:count"), Literal(i)))
    assert read_graph(graph) is graph


def test_parameters_of_any_length_up_to_the_limit_are_compared_not_refused():
    graph = to_sstim_graph("binaural_beats", base=BASE)
    root = URIRef(BASE + "specification")
    text = str(graph.value(root, MUSIC.parametersJson))
    swap(graph, root, MUSIC.parametersJson,
         Literal(text[:-1] + " " * (2048 - len(text)) + "}"))
    assert len(str(graph.value(root, MUSIC.parametersJson))) == 2048
    with pytest.raises(ValueError, match="not those the package writes"):
        from_sstim_graph(graph)


# --------------------------------------------------------------------------
# what the second run found unasserted
# --------------------------------------------------------------------------

def test_a_program_and_a_plan_label_themselves_as_they_always_have():
    plan = SessionPlan("https://example.org/preset", "Preset", CREATED)
    graph = to_sstim_graph(minute(), plan=plan, base=BASE)
    labels = {relative(s): str(o) for s, o in
              graph.subject_objects(RDFS.label)}
    assert labels["program"] == "MUSIC stimulus program"
    assert labels["session-specification"] == "MUSIC planned SSTIM session"
    assert labels[URIRef(plan.preset_iri)] == "Preset"
    assert graph.value(URIRef(BASE + "phase-1"), MUSIC.order) == Literal(
        1, datatype=XSD.integer)


def test_a_plan_whose_program_is_longer_is_refused_for_its_duration():
    plan = SessionPlan("https://example.org/preset", "P", CREATED)
    graph = to_sstim_graph(minute(sample_rate=192000, seconds=7199.),
                           plan=plan, base=BASE)
    longer = to_sstim_graph(
        minute(sample_rate=192000, seconds=7199 + 1 / 192000), base=BASE)
    for triple in list(graph):
        if str(triple[0]) == BASE + "program" or str(
                triple[0]).startswith(BASE + "phase-"):
            graph.remove(triple)
    graph += longer
    with pytest.raises(ValueError) as caught:
        from_sstim_graph(graph)
    assert str(caught.value) == "the plan's duration is not its program's"


@pytest.mark.parametrize("change,message", [
    (lambda g: g.add((AM_RENDERING, SSTIM.renderingCarrierHz,
                      Literal(221.))),
     f"{RENDERING} states more than one sstim:renderingCarrierHz"),
    (lambda g: g.add((AM_CHANNEL, SSTIM.channelDurationSeconds,
                      Literal(1.))),
     f"{CHANNEL} states more than one sstim:channelDurationSeconds"),
    (lambda g: swap(g, AM_CHANNEL, SSTIM.hasSignalRendering, None),
     f"{CHANNEL} states no sstim:hasSignalRendering"),
])
def test_a_refusal_of_an_optional_or_a_missing_term_says_where(change,
                                                               message):
    graph = handwritten_am()
    change(graph)
    with pytest.raises(ValueError) as caught:
        read_am(graph)
    assert str(caught.value) == message


def test_every_problem_is_reported_after_a_channel_that_renders_less():
    graph = Graph().parse(FIXTURES / "sstim_third_party_spatial.ttl")
    graph.remove((URIRef("urn:interop:spatial:left"),
                  SSTIM.hasSignalRendering,
                  URIRef("urn:interop:spatial:l-position")))
    graph.add((URIRef("urn:interop:spatial:right"),
               SSTIM_EX.hasToneFrequencyHz, Literal(999.)))
    with pytest.raises(ValueError) as caught:
        from_sstim_graph(graph, sample_rate=48000)
    assert str(caught.value).endswith(
        "the sstim-ex:placementEarLeft channel renders signals of shape "
        "sstim-v:shapeSine, not sstim-v:shapeSine, sstim-v:shapeTriangle; "
        "the sstim-ex:placementEarRight channel states "
        "sstim-ex:hasToneFrequencyHz 999, which the package does not render")


def test_signals_and_channels_that_differ_are_listed_in_full():
    graph = Graph().parse(FIXTURES / "sstim_third_party_spatial.ttl")
    swap(graph, URIRef("urn:interop:spatial:right"),
         SSTIM_EX.hasBodyPlacement, SSTIM_EX.placementEars)
    with pytest.raises(ValueError) as caught:
        from_sstim_graph(graph, sample_rate=48000)
    assert str(caught.value).endswith(
        "its channels are at sstim-ex:placementEarLeft, "
        "sstim-ex:placementEars, not sstim-ex:placementEarLeft, "
        "sstim-ex:placementEarRight")
    noise = Graph().parse(FIXTURES / "sstim_third_party_noise.ttl")
    swap(noise, URIRef("urn:interop:noise:mod"), SSTIM.hasSignalShape,
         SSTIM_V.shapeTriangle)
    swap(noise, URIRef("urn:interop:noise:envelope-rendering"),
         SSTIM.hasRenderingMechanism, SSTIM_V.mechanismAmplitudeModulation)
    with pytest.raises(ValueError) as caught:
        from_sstim_graph(noise, sample_rate=48000, controls={
            "noise_type": "pink", "modulation_depth": .6, "seed": 27})
    assert ("its signals are of shape sstim-v:shapeNoise, "
            "sstim-v:shapeTriangle, not sstim-v:shapeNoise" in
            str(caught.value))


def test_what_a_description_says_is_checked_before_its_length_is_asked():
    graph = handwritten_am()
    swap(graph, AM_CHANNEL, SSTIM.channelDurationSeconds, None)
    swap(graph, AM_CHANNEL, SSTIM_EX.deliveryMedium, SSTIM_EX.mediumLight)
    with pytest.raises(ValueError, match="medium is sstim-ex:mediumLight"):
        from_sstim_graph(graph, sample_rate=48000)


def test_a_refusal_names_three_problems_however_many_there_are():
    graph = handwritten_am()
    for term, value in ((SSTIM_EX.hasToneFrequencyHz, 1.),
                        (SSTIM_EX.hasFrequencyHz, 2.),
                        (SSTIM_EX.hasDutyCycle, .3),
                        (SSTIM_EX.hasBeatFrequencyHz, 4.)):
        graph.add((AM_CHANNEL, term, Literal(value)))
    with pytest.raises(ValueError) as caught:
        read_am(graph)
    assert str(caught.value).count("which the package does not render") == 3


def test_a_stimulus_that_is_none_of_the_seven_lists_them():
    with pytest.raises(ValueError) as caught:
        SstimStimulus("chirp", {}, .01, 48000).render()
    assert str(caught.value) == (
        "not a stimulus a description can be written for: 'chirp'; they "
        "are binaural_beats, monaural_beats, isochronic_tones, "
        "amplitude_modulation, frequency_modulation, modulated_noise, "
        "spatial_motion")


def test_a_link_to_a_term_of_the_package_s_is_an_annotation():
    graph = handwritten_am()
    graph.add((AM, RDFS.seeAlso, MUSIC.generator))
    assert inspect_sstim(graph).written_by_package is False
    assert read_am(graph).generator == "amplitude_modulation"


def test_a_short_length_given_agrees_to_a_billionth_of_a_second():
    graph = to_sstim_graph("binaural_beats", duration=.001,
                           sample_rate=48000)
    assert from_sstim_graph(graph, duration=.001 + 5e-10).duration == .001
    with pytest.raises(ValueError, match="lasts 0.001 s"):
        from_sstim_graph(graph, duration=.001 + 5e-9)


def test_a_rate_or_length_refused_says_whose_it_is():
    graph = to_sstim_graph("binaural_beats", duration=.1, sample_rate=48000)
    with pytest.raises(ValueError) as caught:
        from_sstim_graph(graph, sample_rate=44100)
    assert str(caught.value) == (
        "the description fixes its sample rate at 48000 Hz; sample_rate is "
        "given as 44100")
    session = music.StimulationSession(sample_rate=8000)
    session.add(music.binaural_beats, duration=.3)
    with pytest.raises(ValueError) as caught:
        from_sstim_graph(to_sstim_graph(session), duration=1)
    assert str(caught.value) == (
        "the session lasts 0.3 s; duration is given as 1")
