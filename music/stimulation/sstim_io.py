"""RDF interoperability between supported MUSIC generators and SSTIM 0.19.

Export uses the *published* SSTIM StimulusSpecification / StimulationSignal /
StimulusChannel / SignalRendering vocabulary. Reproducible engine parameters
that SSTIM does not specify (duty cycle, modulation depth, etc.) are stored
in this library's own namespace, never minted as SSTIM terms.

Only an explicit allowlist of numeric generators is executable. Reading
untrusted RDF never imports code named by the graph, evaluates expressions
or reads network resources.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import inspect
import json
from pathlib import Path
import math
from typing import Any, Mapping

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD

SSTIM = Namespace("https://w3id.org/sstim#")
SSTIM_V = Namespace("https://w3id.org/sstim/vocab#")
SSTIM_EX = Namespace("https://w3id.org/sstim/exposure#")
MUSIC = Namespace("https://github.com/ttm/music/terms#")

# MUSIC is not SSTIM: these names are executable-library configuration,
# explicitly not additions to the portable stimulus ontology.
_GENERATORS = {
    "binaural_beats": ("binaural-beat", "sine", "amplitude", "beat_freq"),
    "monaural_beats": ("monaural-beat", "sine", "amplitude", "beat_freq"),
    "isochronic_tones": ("amplitude-modulation", "square", "amplitude",
                        "pulse_rate"),
    "amplitude_modulation": ("amplitude-modulation", "sine", "amplitude",
                             "modulation_freq"),
    "frequency_modulation": ("frequency-modulation", "sine", "frequency",
                             "modulation_freq"),
}
_MECHANISMS = {
    "binaural-beat": "mechanismBinauralBeat",
    "monaural-beat": "mechanismMonauralBeat",
    "amplitude-modulation": "mechanismAmplitudeModulation",
    "frequency-modulation": "mechanismFrequencyModulation",
}
_PARAMETERS = {"amplitude": "paramAmplitude", "frequency": "paramFrequency"}
_SHAPES = {"sine": "shapeSine", "square": "shapeSquare"}
_MAX_SAMPLES = 5_000_000


@dataclass(frozen=True)
class RenderableStimulus:
    """A MUSIC-renderable subset of one SSTIM StimulusSpecification."""

    generator: str
    parameters: dict[str, float]
    duration: float
    sample_rate: int


def _decimal(value: float) -> Literal:
    return Literal(str(Decimal(str(value))), datatype=XSD.decimal)


def _input(generator: str, params: Mapping[str, Any], duration: float,
           sample_rate: int) -> RenderableStimulus:
    """Validate library-specific parameters without evaluating anything."""
    from . import stimuli

    if generator not in _GENERATORS:
        raise ValueError(f"unsupported stimulus generator: {generator!r}")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise ValueError("sample_rate must be an integer")
    if not 8000 <= sample_rate <= 192000:
        raise ValueError("sample_rate must be between 8000 and 192000 Hz")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        raise ValueError("duration must be numeric")
    duration = float(duration)
    if (not math.isfinite(duration) or duration <= 0
            or int(duration * sample_rate) > _MAX_SAMPLES):
        raise ValueError("duration must be positive with at most 5M samples")
    signature = inspect.signature(getattr(stimuli, generator))
    fields = signature.parameters
    prohibited = {"sample_rate", "number_of_samples", "duration",
                  "waveform_table", "modulation_waveform_table"}
    if any(k not in fields or k in prohibited for k in params):
        raise ValueError("unsupported or reserved generator parameter")
    numbers = {}
    for k, v in params.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{k} must be numeric")
        if not math.isfinite(float(v)):
            raise ValueError(f"{k} must be finite")
        numbers[k] = float(v)

    # Include ALL scalar defaults, so an exported specification does not
    # silently inherit different defaults after a future package upgrade.
    for k, arg in fields.items():
        if k in prohibited or k in numbers:
            continue
        if isinstance(arg.default, (int, float)):
            numbers[k] = float(arg.default)
    carrier = numbers["carrier_freq"]
    modulation = numbers[_GENERATORS[generator][3]]
    if not 0 < carrier < sample_rate / 2 or modulation <= 0:
        raise ValueError("carrier must be inside Nyquist and rate positive")
    if generator in ("binaural_beats", "monaural_beats"):
        if carrier - modulation / 2 <= 0 or carrier + modulation / 2 >= sample_rate / 2:
            raise ValueError("beat carriers must be between 0 and Nyquist")
    if generator == "isochronic_tones":
        if not 0 < numbers["duty_cycle"] <= 1 or numbers["ramp_duration"] < 0:
            raise ValueError("invalid isochronic duty_cycle or ramp_duration")
    if generator == "amplitude_modulation":
        if not 0 <= numbers["modulation_depth"] <= 1:
            raise ValueError("modulation_depth must be between 0 and 1")
    return RenderableStimulus(generator, numbers, duration, sample_rate)


def to_sstim_graph(
        generator: str, *, parameters: Mapping[str, Any],
        duration: float, sample_rate: int = 44100,
        base: str = "https://example.org/music/stimulus/") -> Graph:
    """Describe an allowed auditory generator with portable SSTIM triples.

    Supply your own globally unique base IRI for publication; it must end
    in / or # and may not be in the SSTIM namespace. Exact MUSIC rendering
    parameters are stored in a distinct MUSIC namespace, since SSTIM's
    engine-independent descriptions do not specify every implementation
    detail. Validate with `sstim.validate(graph, profile="full")`.

    Raises
    ------
    ValueError
        If the generator, numerical parameters or namespace are invalid.
    """
    if (not isinstance(base, str) or not base.startswith(("http://", "https://"))
            or not base.endswith(("/", "#")) or
            base.startswith("https://w3id.org/sstim")):
        raise ValueError("base must be your own http(s) namespace ending / or #")
    spec = _input(generator, parameters, duration, sample_rate)
    method, shape, parameter, rate_key = _GENERATORS[generator]
    mod_rate = spec.parameters[rate_key]
    carrier = spec.parameters["carrier_freq"]
    bin_pair = generator == "binaural_beats"
    presence = ("presencePerceptual" if bin_pair else "presencePhysical")
    channels = (2 if bin_pair else 1)

    g = Graph()
    for prefix, ns in (("sstim", SSTIM), ("sstim-v", SSTIM_V),
                       ("sstim-ex", SSTIM_EX), ("music-impl", MUSIC)):
        g.bind(prefix, ns)
    root = URIRef(base + "specification")
    signal = URIRef(base + "signal")
    g.add((root, RDF.type, SSTIM.StimulusSpecification))
    g.add((root, RDFS.label, Literal(f"{generator}: {mod_rate:g} Hz")))
    g.add((root, SSTIM.stimulusRegime, Literal("determinate")))
    g.add((root, SSTIM.hasSignal, signal))
    g.add((signal, RDF.type, SSTIM.StimulationSignal))
    g.add((signal, RDFS.label, Literal(f"{mod_rate:g} Hz {shape} signal")))
    g.add((signal, SSTIM.hzMin, _decimal(mod_rate)))
    g.add((signal, SSTIM.hzMax, _decimal(mod_rate)))
    g.add((signal, SSTIM.hasSignalShape, SSTIM_V[_SHAPES[shape]]))
    g.add((root, MUSIC.generator, Literal(spec.generator)))
    g.add((root, MUSIC.parametersJson, Literal(
        json.dumps(spec.parameters, sort_keys=True, separators=(",", ":")))))
    g.add((root, MUSIC.sampleRateHz, Literal(sample_rate, datatype=XSD.integer)))
    for index in range(channels):
        chan = URIRef(base + f"channel-{index+1}")
        rendering = URIRef(base + f"rendering-{index+1}")
        g.add((root, SSTIM.hasStimulusChannel, chan))
        g.add((chan, RDF.type, SSTIM_EX.StimulusChannel))
        placement = ("placementEarLeft" if index == 0 else "placementEarRight"
                     ) if bin_pair else "placementEars"
        g.add((chan, RDFS.label, Literal(placement)))
        g.add((chan, SSTIM.channelDurationSeconds, _decimal(spec.duration)))
        g.add((chan, SSTIM_EX.perceivedModality, SSTIM_EX.modalityAuditory))
        g.add((chan, SSTIM_EX.deliveryMedium, SSTIM_EX.mediumAirConductedSound))
        g.add((chan, SSTIM_EX.hasBodyPlacement, SSTIM_EX[placement]))
        g.add((chan, SSTIM.hasSignalRendering, rendering))
        g.add((rendering, RDF.type, SSTIM.SignalRendering))
        g.add((rendering, RDFS.label, Literal(f"{generator}: {placement}")))
        g.add((rendering, SSTIM.rendersSignal, signal))
        g.add((rendering, SSTIM.rendersOntoParameter,
               SSTIM_V[_PARAMETERS[parameter]]))
        g.add((rendering, SSTIM.hasRenderingMechanism,
               SSTIM_V[_MECHANISMS[method]]))
        g.add((rendering, SSTIM.hasRenderingPresence, SSTIM_V[presence]))
        frequency = (carrier + (-0.5 if index == 0 else 0.5) * mod_rate
                     if bin_pair else carrier)
        g.add((rendering, SSTIM.renderingCarrierHz, _decimal(frequency)))
    return g


def _graph(value: Graph | str | Path) -> Graph:
    if isinstance(value, Graph):
        return value
    g = Graph()
    if isinstance(value, Path):
        g.parse(value, format="turtle")
    elif isinstance(value, str) and (
            "\n" in value or value.lstrip().startswith("@prefix")):
        g.parse(data=value, format="turtle")
    elif isinstance(value, str) and value.startswith(("http://", "https://")):
        raise ValueError("provide local Turtle data or a local file, not a URL")
    else:
        g.parse(str(value), format="turtle")
    return g


def from_sstim_graph(value: Graph | str | Path) -> RenderableStimulus:
    """Decode only the explicitly supported MUSIC subset of an SSTIM graph.

    Refuse ambiguous graphs or a mismatch between engine hints and the
    portable SSTIM assertions. Unknown predicates are left untouched;
    unknown executable generator names are never dispatched.
    """
    g = _graph(value)
    roots = list(g.subjects(RDF.type, SSTIM.StimulusSpecification))
    if len(roots) != 1:
        raise ValueError("expected exactly one SSTIM StimulusSpecification")
    root = roots[0]

    def one(node, predicate):
        values = list(g.objects(node, predicate))
        if len(values) != 1:
            raise ValueError(f"expected exactly one {predicate}")
        return values[0]

    generator = str(one(root, MUSIC.generator))
    params_text = str(one(root, MUSIC.parametersJson))
    if len(params_text) > 2048:
        raise ValueError("excessively long engine parameters")
    try:
        params = json.loads(params_text)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid parametersJson") from exc
    if not isinstance(params, dict):
        raise ValueError("parametersJson must be an object")
    rate = one(root, MUSIC.sampleRateHz)
    if rate.datatype != XSD.integer:
        raise ValueError("sampleRateHz must be xsd:integer")
    signal = one(root, SSTIM.hasSignal)
    mod_rate = float(one(signal, SSTIM.hzMin))
    if float(one(signal, SSTIM.hzMax)) != mod_rate:
        raise ValueError("variable-rate signals are not supported")
    channels = list(g.objects(root, SSTIM.hasStimulusChannel))
    expected_count = 2 if generator == "binaural_beats" else 1
    if len(channels) != expected_count:
        raise ValueError("wrong channel count for the generator")
    duration = float(one(channels[0], SSTIM.channelDurationSeconds))
    spec = _input(generator, params, duration, int(rate))
    method, shape, param, rate_key = _GENERATORS[generator]
    if mod_rate != spec.parameters[rate_key] or (
            one(signal, SSTIM.hasSignalShape) != SSTIM_V[_SHAPES[shape]]):
        raise ValueError("signal assertions contradict engine parameters")
    if str(one(root, SSTIM.stimulusRegime)) != "determinate":
        raise ValueError("only determinate stimuli are executable")
    seen_placements = set()
    for index, channel in enumerate(channels):
        if float(one(channel, SSTIM.channelDurationSeconds)) != duration:
            raise ValueError("channel duration mismatch")
        rendering = one(channel, SSTIM.hasSignalRendering)
        if one(rendering, SSTIM.rendersSignal) != signal:
            raise ValueError("signal rendering mismatch")
        if one(rendering, SSTIM.hasRenderingMechanism) != (
                SSTIM_V[_MECHANISMS[method]]):
            raise ValueError("rendering mechanism mismatch")
        if one(rendering, SSTIM.rendersOntoParameter) != (
                SSTIM_V[_PARAMETERS[param]]):
            raise ValueError("rendering parameter mismatch")
        presence = ("presencePerceptual" if generator == "binaural_beats"
                    else "presencePhysical")
        if one(rendering, SSTIM.hasRenderingPresence) != SSTIM_V[presence]:
            raise ValueError("rendering presence mismatch")
        carrier = spec.parameters["carrier_freq"]
        if generator == "binaural_beats":
            # Ordered by placement, not RDF triple iteration order.
            placement = one(channel, SSTIM_EX.hasBodyPlacement)
            if placement not in (SSTIM_EX.placementEarLeft,
                                 SSTIM_EX.placementEarRight):
                raise ValueError("invalid binaural channel placement")
            if placement in seen_placements:
                raise ValueError("duplicate binaural channel placement")
            seen_placements.add(placement)
            index = 0 if placement == SSTIM_EX.placementEarLeft else 1
            expected = carrier + (-0.5 if index == 0 else 0.5) * mod_rate
        else:
            expected = carrier
        if float(one(rendering, SSTIM.renderingCarrierHz)) != expected:
            raise ValueError("carrier frequency mismatch")
    return spec


def render_sstim(value: Graph | str | Path, *,
                 oversampling_factor: int | None = None):
    """Render validated MUSIC hints in a portable SSTIM stimulus graph.

    Default is the original generator. Set `oversampling_factor=4` to
    low-pass before decimating, for example for hard-gated isochronic tones.
    SSTIM itself specifies the stimulus, not the engine's numerical fidelity.
    """
    from . import stimuli
    spec = from_sstim_graph(value)
    generator = getattr(stimuli, spec.generator)  # checked allowlist
    if oversampling_factor is None:
        return generator(duration=spec.duration,
                         sample_rate=spec.sample_rate, **spec.parameters)
    from ..core.synths.oversampling import render_oversampled
    return render_oversampled(
        generator, duration=spec.duration, sample_rate=spec.sample_rate,
        factor=oversampling_factor, **spec.parameters)


def validate_sstim(value: Graph | str | Path, *,
                   version: str = "0.19.0"):
    """Verify actual SSTIM Full-profile conformance using its validator.

    Downloads a versioned manifest/closure the first time, checksum checks
    the modules, and caches them. No network required after caching. This
    validation is stricter than the local engine-specific round-trip checks.
    """
    try:
        import sstim
    except ImportError as exc:
        raise ImportError(
            "install the optional SSTIM integration with "
            "pip install 'music[sstim]'") from exc
    return sstim.validate(_graph(value), profile="full", version=version)
