"""Noise and spatial-motion SSTIM 0.19.0 stimulus descriptions.

Unlike the five tone-only renderings, a noise source is stochastic and has a
*band*, not a tonal carrier. Spatial motion is a time-varying position signal
coupled to a tonal source. SSTIM signals preserve those distinctions; numeric
engine controls live in the MUSIC extension namespace. These records describe
a stimulus, never a treatment effect or a delivered SessionInstance.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from . import stimuli
from .sstim_io import (
    MUSIC, SSTIM, SSTIM_EX, SSTIM_V, _decimal, _graph,
)

_ADVANCED = {"modulated_noise", "spatial_motion"}
_COLORS = {"white", "pink", "brown", "blue", "violet", "black"}
_MAX_SAMPLES = 5_000_000


def _parameters(name: str, supplied: Mapping[str, Any], duration: float,
                sample_rate: int) -> dict[str, Any]:
    """Validate and explicitly pin a restricted, executable parameter set."""
    if name not in _ADVANCED:
        raise ValueError("unsupported advanced stimulus")
    if (not isinstance(sample_rate, int) or isinstance(sample_rate, bool)
            or not 8000 <= sample_rate <= 192000):
        raise ValueError("sample_rate must be in [8000, 192000]")
    if (not isinstance(duration, (int, float)) or isinstance(duration, bool)
            or not math.isfinite(duration) or duration <= 0
            or int(duration * sample_rate) > _MAX_SAMPLES):
        raise ValueError("duration must be positive and bounded")
    prohibited = {"duration", "number_of_samples", "sample_rate",
                  "waveform_table", "modulation_waveform_table",
                  "sonic_vector"}
    defaults = inspect.signature(getattr(stimuli, name)).parameters
    if any(k not in defaults or k in prohibited for k in supplied):
        raise ValueError("unknown or nonportable generator parameter")
    params = {}
    for k, arg in defaults.items():
        if k in prohibited:
            continue
        value = supplied.get(k, arg.default)
        if k == "noise_type":
            if not isinstance(value, str) or value not in _COLORS:
                raise ValueError("noise_type must be a named noise color")
        elif k == "seed":
            if value is not None and (
                    not isinstance(value, int) or isinstance(value, bool)
                    or not 0 <= value < 2**64):
                raise ValueError("seed must be a uint64 integer or None")
        elif (isinstance(value, bool) or
              not isinstance(value, (int, float)) or
              not math.isfinite(value)):
            raise ValueError(f"{k} must be a finite number")
        else:
            value = float(value)
        params[k] = value
    if name == "modulated_noise":
        if (not 0 <= params["modulation_depth"] <= 1
                or params["modulation_freq"] < 0
                or not 0 <= params["min_freq"] < params["max_freq"]
                or params["max_freq"] > sample_rate / 2):
            raise ValueError("invalid noise depth, rate, or frequency band")
    elif (not 0 < params["carrier_freq"] < sample_rate / 2
          or params["motion_rate"] < 0 or params["dist"] <= 0
          or params["zeta"] <= 0 or not -100 < params["air_temp"] < 100):
        raise ValueError("invalid spatial carrier, motion, or geometry")
    return params


def to_sstim_advanced_graph(
        generator: str, *, parameters: Mapping[str, Any],
        duration: float, sample_rate: int = 44100,
        base: str = "https://example.org/music/advanced/") -> Graph:
    """Export noise or spatial motion with correct signal/regime distinctions.

    Only default sine carriers and sine noise modulators are accepted.
    Source arrays and custom wavetables are not portable through this
    adapter. A noise seed reproduces one stochastic realization in MUSIC.

    Raises
    ------
    ValueError
        For invalid generators, parameters, or record namespaces.
    """
    # Use the same SSTIM namespace discipline as the basic adapter.
    if (not isinstance(base, str) or
            not base.startswith(("https://", "http://")) or
            not base.endswith(("/", "#")) or
            base.startswith("https://w3id.org/sstim")):
        raise ValueError("base must be a non-SSTIM http(s) IRI ending / or #")
    params = _parameters(generator, parameters, duration, sample_rate)
    graph = Graph()
    for prefix, ns in (("sstim", SSTIM), ("sstim-v", SSTIM_V),
                       ("sstim-ex", SSTIM_EX), ("music-impl", MUSIC)):
        graph.bind(prefix, ns)
    root = URIRef(base + "specification")
    graph.add((root, RDF.type, SSTIM.StimulusSpecification))
    graph.add((root, RDFS.label, Literal(f"MUSIC {generator} stimulus")))
    regime = "stochastic" if generator == "modulated_noise" else "determinate"
    graph.add((root, SSTIM.stimulusRegime, Literal(regime)))
    graph.add((root, MUSIC.generator, Literal(generator)))
    graph.add((root, MUSIC.parametersJson, Literal(
        json.dumps(params, sort_keys=True, separators=(",", ":")))))
    graph.add((root, MUSIC.sampleRateHz,
               Literal(sample_rate, datatype=XSD.integer)))

    if generator == "modulated_noise":
        source = URIRef(base + "signal-noise")
        graph.add((root, SSTIM.hasSignal, source))
        graph.add((source, RDF.type, SSTIM.StimulationSignal))
        graph.add((source, RDFS.label,
                   Literal(params["noise_type"] + " noise band")))
        graph.add((source, SSTIM.hasSignalShape, SSTIM_V.shapeNoise))
        graph.add((source, SSTIM.hzMin, _decimal(params["min_freq"])))
        graph.add((source, SSTIM.hzMax, _decimal(params["max_freq"])))
        signals = [(source, "paramAmplitude",
                    "mechanismDirectPresentation")]
        if params["modulation_freq"] > 0:
            mod = URIRef(base + "signal-modulation")
            graph.add((root, SSTIM.hasSignal, mod))
            graph.add((mod, RDF.type, SSTIM.StimulationSignal))
            graph.add((mod, RDFS.label, Literal("noise amplitude modulation")))
            graph.add((mod, SSTIM.hasSignalShape, SSTIM_V.shapeSine))
            graph.add((mod, SSTIM.hzMin,
                       _decimal(params["modulation_freq"])))
            graph.add((mod, SSTIM.hzMax,
                       _decimal(params["modulation_freq"])))
            signals.append((mod, "paramAmplitude",
                            "mechanismAmplitudeModulation"))
        placements: tuple[str, ...] = ("placementEars",)
    else:
        tone = URIRef(base + "signal-tone")
        motion = URIRef(base + "signal-motion")
        for signal, label, shape, hz in (
                (tone, "acoustic source tone", "shapeSine",
                 params["carrier_freq"]),
                (motion, "triangular azimuth motion", "shapeTriangle",
                 params["motion_rate"])):
            graph.add((root, SSTIM.hasSignal, signal))
            graph.add((signal, RDF.type, SSTIM.StimulationSignal))
            graph.add((signal, RDFS.label, Literal(label)))
            graph.add((signal, SSTIM.hasSignalShape, SSTIM_V[shape]))
            graph.add((signal, SSTIM.hzMin, _decimal(hz)))
            graph.add((signal, SSTIM.hzMax, _decimal(hz)))
        signals = [(tone, "paramAmplitude", "mechanismDirectPresentation"),
                   (motion, "paramSpatialPosition",
                    "mechanismDirectPresentation")]
        placements = ("placementEarLeft", "placementEarRight")

    for n, placement in enumerate(placements, 1):
        channel = URIRef(base + f"channel-{n}")
        graph.add((root, SSTIM.hasStimulusChannel, channel))
        graph.add((channel, RDF.type, SSTIM_EX.StimulusChannel))
        graph.add((channel, RDFS.label, Literal(placement)))
        graph.add((channel, SSTIM.channelDurationSeconds, _decimal(duration)))
        graph.add((channel, SSTIM_EX.perceivedModality,
                   SSTIM_EX.modalityAuditory))
        graph.add((channel, SSTIM_EX.deliveryMedium,
                   SSTIM_EX.mediumAirConductedSound))
        graph.add((channel, SSTIM_EX.hasBodyPlacement,
                   SSTIM_EX[placement]))
        for index, (signal, parameter, mechanism) in enumerate(signals, 1):
            rendering = URIRef(base + f"rendering-{n}-{index}")
            graph.add((channel, SSTIM.hasSignalRendering, rendering))
            graph.add((rendering, RDF.type, SSTIM.SignalRendering))
            graph.add((rendering, RDFS.label,
                       Literal(f"{placement}: {parameter}")))
            graph.add((rendering, SSTIM.rendersSignal, signal))
            graph.add((rendering, SSTIM.rendersOntoParameter,
                       SSTIM_V[parameter]))
            graph.add((rendering, SSTIM.hasRenderingMechanism,
                       SSTIM_V[mechanism]))
            graph.add((rendering, SSTIM.hasRenderingPresence,
                       SSTIM_V.presencePhysical))
    return graph


def from_sstim_advanced_graph(value: Graph | str | Path):
    """Read exactly a MUSIC-exported noise or spatial stimulus subgraph.

    The comparison is deliberately strict: changed semantic triples
    invalidate the executable hints. To ingest an independent engine's
    SSTIM descriptions, a separate mapping review is required.
    """
    graph = _graph(value)
    roots = list(graph.subjects(RDF.type, SSTIM.StimulusSpecification))
    if len(roots) != 1:
        raise ValueError("expected one advanced StimulusSpecification")
    root = roots[0]
    base = str(root)
    if not base.endswith("specification"):
        raise ValueError("unexpected stimulus IRI")
    base = base[:-len("specification")]

    def one(predicate):
        items = list(graph.objects(root, predicate))
        if len(items) != 1:
            raise ValueError("missing or ambiguous engine hint")
        return items[0]

    name = str(one(MUSIC.generator))
    try:
        params = json.loads(str(one(MUSIC.parametersJson)))
        sample_rate = int(one(MUSIC.sampleRateHz))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid engine parameters") from exc
    channels = list(graph.objects(root, SSTIM.hasStimulusChannel))
    if not channels:
        raise ValueError("missing stimulus channel")
    duration = float(str(next(graph.objects(
        channels[0], SSTIM.channelDurationSeconds))))
    expected = to_sstim_advanced_graph(
        name, parameters=params, duration=duration,
        sample_rate=sample_rate, base=base)
    if set(graph) != set(expected):
        raise ValueError("SSTIM signal/rendering graph differs from engine")
    return name, params, duration, sample_rate


def render_sstim_advanced(value: Graph | str | Path) -> np.ndarray:
    """Render a verified advanced stimulus; deterministic only if seeded.

    Returns
    -------
    ndarray
        Mono noise or two-channel spatial sound.
    """
    name, params, duration, sample_rate = from_sstim_advanced_graph(value)
    return getattr(stimuli, name)(
        duration=duration, sample_rate=sample_rate, **params)
