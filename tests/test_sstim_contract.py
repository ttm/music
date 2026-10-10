"""Third-party SSTIM + explicit sidecar contract interoperability tests.

The independent fixtures have no MUSIC hints. Their RDF IRI choices,
statement ordering and numeric lexical forms differ from MUSIC output.
Rendering still requires an explicit complete engine-dependent sidecar.
"""
import os
import numpy as np
import pytest

pytest.importorskip("rdflib")
from rdflib import Graph, Literal, URIRef  # noqa: E402
from rdflib.namespace import RDF, RDFS  # noqa: E402

from music.stimulation import stimuli  # noqa: E402
from music.stimulation.sstim_advanced import (  # noqa: E402
    to_sstim_advanced_graph,
)
from music.stimulation.sstim_io import (  # noqa: E402
    MUSIC, SSTIM, SSTIM_EX, SSTIM_V, to_sstim_graph, validate_sstim,
)
from music.stimulation.sstim_contract import (  # noqa: E402
    PortableResolution, render_sstim_contract, resolve_sstim_contract,
)


CASES = [
    ("binaural_beats", {"carrier_freq": 220., "beat_freq": 9.}),
    ("monaural_beats", {"carrier_freq": 220., "beat_freq": 9.}),
    ("isochronic_tones", {
        "carrier_freq": 220., "pulse_rate": 9., "duty_cycle": .4,
        "ramp_duration": .003}),
    ("amplitude_modulation", {
        "carrier_freq": 220., "modulation_freq": 9.,
        "modulation_depth": .7}),
    ("frequency_modulation", {
        "carrier_freq": 220., "modulation_freq": 9.,
        "frequency_deviation": 14.}),
    ("modulated_noise", {
        "noise_type": "pink", "min_freq": 70., "max_freq": 9000.,
        "modulation_freq": 9., "modulation_depth": .6, "seed": 27}),
    ("spatial_motion", {
        "carrier_freq": 220., "motion_rate": .5, "theta1": 180.,
        "theta2": 0., "dist": .1, "zeta": .215, "air_temp": 20.}),
]


def third_party(name, params, *, duration=.04, rate=48000):
    """Use real SSTIM structure but foreign resource identifiers and labels."""
    kwargs = dict(parameters=params, duration=duration,
                  sample_rate=rate,
                  base="https://external.example.net/instrument/alpha/")
    if name in ("modulated_noise", "spatial_motion"):
        graph = to_sstim_advanced_graph(name, **kwargs)
    else:
        graph = to_sstim_graph(name, **kwargs)
    out = Graph()
    original = kwargs["base"]
    for subject, predicate, obj in graph:
        if str(predicate).startswith(str(MUSIC)):
            continue
        if predicate == RDFS.label:
            # Descriptive labels are non-normative and can be translated.
            obj = Literal("A translated independent label", lang="it")
        if isinstance(subject, URIRef) and str(subject).startswith(original):
            subject = URIRef(str(subject).replace(
                original, "urn:independent:"))
        if isinstance(obj, URIRef) and str(obj).startswith(original):
            obj = URIRef(str(obj).replace(
                original, "urn:independent:"))
        out.add((subject, predicate, obj))
    # A third party's description is not required to use MUSIC's namespace.
    assert not [p for _, p, _ in out if str(p).startswith(str(MUSIC))]
    return out


@pytest.mark.parametrize("name,params", CASES)
def test_all_seven_independent_iri_profiles_render_with_full_sidecar(
        name, params):
    graph = third_party(name, params)
    decoded = resolve_sstim_contract(
        graph, generator=name, parameters=params,
        duration=.04, sample_rate=48000)
    assert decoded.generator == name
    assert not decoded.exact_pcm_portable
    received = render_sstim_contract(decoded)
    direct = getattr(stimuli, name)(
        duration=.04, sample_rate=48000, **decoded.parameters)
    np.testing.assert_array_equal(received, direct)


@pytest.mark.parametrize("name,params", CASES)
def test_no_implicit_engine_defaults_on_independent_graph(name, params):
    graph = third_party(name, params)
    incomplete = dict(params)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="explicitly pin"):
        resolve_sstim_contract(
            graph, generator=name, parameters=incomplete,
            duration=.04, sample_rate=48000)


def test_non_music_annotated_turtle_is_accepted():
    graph = third_party(*CASES[0])
    root = next(graph.subjects(RDF.type, SSTIM.StimulusSpecification))
    graph.add((root, URIRef("https://other.example/note"),
               Literal("external note")))
    text = graph.serialize(format="turtle")
    answer = resolve_sstim_contract(
        text, generator=CASES[0][0],
        parameters=CASES[0][1], duration=.04, sample_rate=48000)
    assert answer.generator == "binaural_beats"


INDEPENDENT_AM_TURTLE = """
@prefix s: <https://w3id.org/sstim#> .
@prefix v: <https://w3id.org/sstim/vocab#> .
@prefix e: <https://w3id.org/sstim/exposure#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<urn:outside:spec> a s:StimulusSpecification ;
  rdfs:label "external AM" ;
  s:stimulusRegime "determinate" ;
  s:hasSignal <urn:outside:signal> ;
  s:hasStimulusChannel <urn:outside:channel> .
<urn:outside:signal> a s:StimulationSignal ;
  rdfs:label "9 Hz modulator" ;
  s:hasSignalShape v:shapeSine ;
  s:hzMin 9 ; s:hzMax 9.0 .
<urn:outside:channel> a e:StimulusChannel ;
  rdfs:label "air-conducted mono" ;
  e:perceivedModality e:modalityAuditory ;
  e:deliveryMedium e:mediumAirConductedSound ;
  e:hasBodyPlacement e:placementEars ;
  s:channelDurationSeconds 0.04 ;
  s:hasSignalRendering <urn:outside:r> .
<urn:outside:r> a s:SignalRendering ;
  rdfs:label "physical amplitude modulation" ;
  s:rendersSignal <urn:outside:signal> ;
  s:rendersOntoParameter v:paramAmplitude ;
  s:hasRenderingMechanism v:mechanismAmplitudeModulation ;
  s:hasRenderingPresence v:presencePhysical ;
  s:renderingCarrierHz 220.0 .
"""


def handwritten_am():
    return Graph().parse(data=INDEPENDENT_AM_TURTLE, format="turtle")


def test_hand_authored_am_fixture_does_not_depend_on_exporter():
    decoded = resolve_sstim_contract(
        handwritten_am(), generator="amplitude_modulation",
        parameters=CASES[3][1], duration=.04, sample_rate=48000)
    assert decoded.parameters["modulation_depth"] == .7
    assert render_sstim_contract(decoded).shape == (1920,)


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SSTIM_LIVE") != "1",
                    reason="frozen validator on Python 3.12 only")
def test_hand_authored_am_conforms_to_official_full_profile():
    validation = validate_sstim(handwritten_am(), version="0.19.0")
    assert validation.ok, str(validation)


@pytest.mark.parametrize("tamper,match", [
    ("wrong-carrier", "contradict"),
    ("wrong-modulation", "contradict"),
    ("wrong-mechanism", "contradict"),
    ("wrong-placement", "contradict"),
    ("extra-channel", "contradict"),
    ("extra-sstim-assertion", "unsupported SSTIM"),
    ("bad-signal-numeric", "invalid numeric"),
    ("nonfinite-numeric", "nonfinite"),
    ("nonliteral-numeric", "numeric"),
    ("different-duration", "contradict"),
    ("unrelated-signal", "foreign signal"),
])
def test_fail_closed_on_contradictory_important_sstim_assertions(
        tamper, match):
    g = handwritten_am()
    root = URIRef("urn:outside:spec")
    chan = URIRef("urn:outside:channel")
    render = URIRef("urn:outside:r")
    signal = URIRef("urn:outside:signal")

    def change(subject, predicate, obj):
        g.remove((subject, predicate, None))
        g.add((subject, predicate, obj))

    if tamper == "wrong-carrier":
        change(render, SSTIM.renderingCarrierHz, Literal(211))
    elif tamper == "wrong-modulation":
        change(signal, SSTIM.hzMin, Literal(8))
        change(signal, SSTIM.hzMax, Literal(8))
    elif tamper == "wrong-mechanism":
        change(render, SSTIM.hasRenderingMechanism,
               SSTIM_V.mechanismFrequencyModulation)
    elif tamper == "wrong-placement":
        change(chan, SSTIM_EX.hasBodyPlacement,
               SSTIM_EX.placementEarLeft)
    elif tamper == "extra-channel":
        g.add((root, SSTIM.hasStimulusChannel, URIRef("urn:outside:extra")))
    elif tamper == "extra-sstim-assertion":
        g.add((signal, SSTIM.hasAdaptationPolicy,
               URIRef("urn:outside:policy")))
    elif tamper == "bad-signal-numeric":
        change(signal, SSTIM.hzMin, Literal("invalid"))
    elif tamper == "nonfinite-numeric":
        change(signal, SSTIM.hzMin, Literal("NaN"))
    elif tamper == "nonliteral-numeric":
        change(signal, SSTIM.hzMin, URIRef("urn:outside:not-number"))
    elif tamper == "different-duration":
        change(chan, SSTIM.channelDurationSeconds, Literal(1))
    else:
        change(render, SSTIM.rendersSignal,
               URIRef("urn:outside:other"))
    with pytest.raises(ValueError, match=match):
        resolve_sstim_contract(
            g, generator="amplitude_modulation",
            parameters=CASES[3][1],
            duration=.04, sample_rate=48000)


def test_music_hints_must_be_verified_by_their_own_importer():
    g = to_sstim_graph(
        "monaural_beats",
        parameters=CASES[1][1], duration=.04, sample_rate=48000)
    with pytest.raises(ValueError, match="MUSIC hints"):
        resolve_sstim_contract(
            g, generator=CASES[1][0],
            parameters=CASES[1][1], duration=.04, sample_rate=48000)


def test_reject_unknown_sidecars_and_manual_nonallowlisted_resolution():
    g = handwritten_am()
    with pytest.raises(ValueError, match="unsupported sidecar"):
        resolve_sstim_contract(
            g, generator="exec", parameters={}, duration=.04)
    with pytest.raises(ValueError, match="invalid portable"):
        render_sstim_contract(
            PortableResolution("exec", {}, .04, 48000))
    with pytest.raises(ValueError, match="invalid portable"):
        render_sstim_contract("not a resolution")


def test_manual_resolution_revalidates_parameters_and_bounds():
    resolution = PortableResolution(
        "amplitude_modulation", {
            "carrier_freq": -3, "modulation_freq": 10,
            "modulation_depth": 1}, .04, 48000)
    with pytest.raises(ValueError, match="carrier"):
        render_sstim_contract(resolution)
    noise = PortableResolution(
        "modulated_noise", {"noise_type": "no-such-color"}, .04, 48000)
    with pytest.raises(ValueError, match="noise_type"):
        render_sstim_contract(noise)
