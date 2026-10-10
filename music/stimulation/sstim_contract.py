"""Validate executable sidecar controls against *standard-only* SSTIM RDF.

A third-party engine can author SSTIM 0.19.0 signal/channel/rendering
descriptions without adopting MUSIC's private RDF vocabulary. SSTIM does
not specify every synthesis detail (seed, AM depth, fades, spatial
geometry). To render one such description with MUSIC, a caller must
provide a *separate, explicit*, allowlisted engine contract.

The verifier compares the complete used SSTIM signal/channel/rendering
projection, independent of RDF node identities, statement ordering,
literal numeric lexical forms and descriptive labels. It does not
claim that MUSIC's implementation is the only valid SSTIM renderer.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF

from .sstim_advanced import _ADVANCED, to_sstim_advanced_graph
from .sstim_io import (
    MUSIC, SSTIM, SSTIM_EX, SSTIM_V, _GENERATORS, _graph, to_sstim_graph,
)

_ALLOWED = frozenset(_GENERATORS) | _ADVANCED
_NAMESPACE_PREFIXES = (str(SSTIM), str(SSTIM_EX), str(SSTIM_V))
_ROOT_FIELDS = frozenset((SSTIM.stimulusRegime, SSTIM.hasSignal,
                          SSTIM.hasStimulusChannel))
_SIGNAL_FIELDS = frozenset((SSTIM.hasSignalShape, SSTIM.hzMin, SSTIM.hzMax))
_CHANNEL_FIELDS = frozenset((
    SSTIM.channelDurationSeconds, SSTIM.hasSignalRendering,
    SSTIM_EX.perceivedModality, SSTIM_EX.deliveryMedium,
    SSTIM_EX.hasBodyPlacement))
_RENDER_FIELDS = frozenset((
    SSTIM.rendersSignal, SSTIM.rendersOntoParameter,
    SSTIM.hasRenderingMechanism, SSTIM.hasRenderingPresence,
    SSTIM.renderingCarrierHz))


@dataclass(frozen=True)
class PortableResolution:
    """MUSIC's explicit local rendering choice for standard SSTIM RDF."""

    generator: str
    parameters: dict[str, Any]
    duration: float
    sample_rate: int
    # Always false: the external contract is not an SSTIM execution norm.
    exact_pcm_portable: bool = False


def _one(graph: Graph, subject, predicate):
    values = list(graph.objects(subject, predicate))
    if len(values) != 1:
        raise ValueError(f"expected exactly one {predicate}")
    return values[0]


def _number(graph: Graph, subject, predicate):
    node = _one(graph, subject, predicate)
    if not isinstance(node, Literal) or isinstance(node.value, bool):
        raise ValueError("numeric SSTIM value must be a literal")
    try:
        number = Decimal(str(node))
    except InvalidOperation as exc:
        raise ValueError("invalid numeric SSTIM literal") from exc
    if not number.is_finite():
        raise ValueError("nonfinite numeric SSTIM literal")
    return number


def _iri(graph: Graph, subject, predicate):
    value = _one(graph, subject, predicate)
    if not isinstance(value, URIRef):
        raise ValueError("SSTIM relationship must be an IRI")
    return value


def _fields(graph: Graph, node, allowed):
    """Refuse extra *SSTIM* predicates not described by this executable
    subset: unknown fields may change the stimulus in a third-party graph.
    Other namespaces are descriptive and ignored, as are labels.
    """
    for predicate in graph.predicates(node, None):
        if (any(str(predicate).startswith(p) for p in _NAMESPACE_PREFIXES)
                and predicate not in allowed):
            raise ValueError(f"unsupported SSTIM assertion: {predicate}")


def _typed(graph: Graph, node, expected):
    classes = set(graph.objects(node, RDF.type))
    if expected not in classes:
        raise ValueError(f"missing expected RDF type {expected}")
    for cls in classes:
        if (cls != expected and isinstance(cls, URIRef)
                and any(str(cls).startswith(p) for p in _NAMESPACE_PREFIXES)):
            raise ValueError("contradictory SSTIM RDF type")


def _signal(graph: Graph, node):
    _typed(graph, node, SSTIM.StimulationSignal)
    _fields(graph, node, _SIGNAL_FIELDS)
    return (_iri(graph, node, SSTIM.hasSignalShape),
            _number(graph, node, SSTIM.hzMin),
            _number(graph, node, SSTIM.hzMax))


def _render(graph: Graph, node, owned_signals):
    _typed(graph, node, SSTIM.SignalRendering)
    _fields(graph, node, _RENDER_FIELDS)
    source = _iri(graph, node, SSTIM.rendersSignal)
    if source not in owned_signals:
        raise ValueError("rendering references a foreign signal")
    carrier = list(graph.objects(node, SSTIM.renderingCarrierHz))
    if len(carrier) > 1:
        raise ValueError("ambiguous rendering carrier")
    return (
        _signal(graph, source),
        _iri(graph, node, SSTIM.rendersOntoParameter),
        _iri(graph, node, SSTIM.hasRenderingMechanism),
        _iri(graph, node, SSTIM.hasRenderingPresence),
        (_number(graph, node, SSTIM.renderingCarrierHz)
         if carrier else None),
    )


def _channel(graph: Graph, node, signals):
    _typed(graph, node, SSTIM_EX.StimulusChannel)
    _fields(graph, node, _CHANNEL_FIELDS)
    renderings = list(graph.objects(node, SSTIM.hasSignalRendering))
    if not renderings or len(renderings) > 8:
        raise ValueError("invalid rendering count")
    return (
        _number(graph, node, SSTIM.channelDurationSeconds),
        _iri(graph, node, SSTIM_EX.perceivedModality),
        _iri(graph, node, SSTIM_EX.deliveryMedium),
        _iri(graph, node, SSTIM_EX.hasBodyPlacement),
        frozenset(_render(graph, r, signals) for r in renderings),
        len(renderings),
    )


def _projection(graph: Graph):
    roots = list(graph.subjects(RDF.type, SSTIM.StimulusSpecification))
    if len(roots) != 1:
        raise ValueError("expected one SSTIM StimulusSpecification")
    root = roots[0]
    _typed(graph, root, SSTIM.StimulusSpecification)
    _fields(graph, root, _ROOT_FIELDS)
    signals = list(graph.objects(root, SSTIM.hasSignal))
    channels = list(graph.objects(root, SSTIM.hasStimulusChannel))
    if not (1 <= len(signals) <= 8 and 1 <= len(channels) <= 8):
        raise ValueError("invalid stimulus signal/channel cardinality")
    if any(not isinstance(x, URIRef) for x in signals + channels):
        raise ValueError("SSTIM nodes must have IRIs")
    return (
        str(_one(graph, root, SSTIM.stimulusRegime)),
        frozenset(_signal(graph, s) for s in signals), len(signals),
        frozenset(_channel(graph, c, set(signals)) for c in channels),
        len(channels),
    )


def resolve_sstim_contract(
        value: Graph | str | Path, *, generator: str,
        parameters: Mapping[str, Any], duration: float,
        sample_rate: int = 44100) -> PortableResolution:
    """Resolve a third-party SSTIM stimulus using explicit external controls.

    All seven built-in auditory generators are supported *only* with
    caller-supplied MUSIC controls. Numeric controls not represented by
    SSTIM must never be silently guessed from labels or engine defaults
    on an independent graph.

    MUSIC-owned RDF predicate hints are refused here; use the separate
    strict MUSIC round-trip importer to validate those. Thus stale or
    contradictory MUSIC metadata cannot be hidden behind a valid
    standard-only description.

    Does not infer a delivered SessionInstance or portable PCM identity.
    """
    graph = _graph(value)
    if any(str(pred).startswith(str(MUSIC)) for _, pred, _ in graph):
        raise ValueError("MUSIC hints require the strict MUSIC importer")
    if generator not in _ALLOWED:
        raise ValueError("unsupported sidecar generator")
    if generator in _ADVANCED:
        expected = to_sstim_advanced_graph(
            generator, parameters=parameters, duration=duration,
            sample_rate=sample_rate)
    else:
        expected = to_sstim_graph(
            generator, parameters=parameters, duration=duration,
            sample_rate=sample_rate)
    if _projection(graph) != _projection(expected):
        raise ValueError("SSTIM assertions contradict explicit sidecar")
    # Use the existing strict validator to pin scalar defaults and
    # reject incompatible units/values. No executable RDF is evaluated.
    if generator in _ADVANCED:
        from .sstim_advanced import from_sstim_advanced_graph
        name, params, seconds, rate = from_sstim_advanced_graph(expected)
    else:
        from .sstim_io import from_sstim_graph
        checked = from_sstim_graph(expected)
        name, params, seconds, rate = (
            checked.generator, checked.parameters, checked.duration,
            checked.sample_rate)
    return PortableResolution(name, params, seconds, rate)


def render_sstim_contract(resolution: PortableResolution):
    """Render a previously checked sidecar contract with allowlisted MUSIC."""
    from . import stimuli
    if (not isinstance(resolution, PortableResolution)
            or resolution.generator not in _ALLOWED):
        raise ValueError("invalid portable rendering resolution")
    # The ordinary resolution returned above was strictly checked.
    # Re-validate untrusted manually built dataclass instances too.
    if resolution.generator in _ADVANCED:
        from .sstim_advanced import _parameters
        params = _parameters(
            resolution.generator, resolution.parameters,
            resolution.duration, resolution.sample_rate)
    else:
        from .sstim_io import _input
        spec = _input(
            resolution.generator, resolution.parameters,
            resolution.duration, resolution.sample_rate)
        params = spec.parameters
    return getattr(stimuli, resolution.generator)(
        duration=resolution.duration,
        sample_rate=resolution.sample_rate, **params)
