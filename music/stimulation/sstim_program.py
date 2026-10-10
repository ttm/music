"""Ordered, executable MUSIC programs containing SSTIM stimulus graphs.

SSTIM describes a stimulus; an ordered MUSIC program states how this
library combines several of those stimuli with overlapping fades. This
module does NOT mint sstim:SessionSpecification or sstim:SessionInstance:
neither an execution clock nor delivered-exposure observations exist here.
It uses MUSIC-owned terms for ordering, ramps and rendering choices while
reusing valid SSTIM specifications for each constituent stimulus.
"""

from __future__ import annotations

import math
from pathlib import Path

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from . import stimuli
from .session import StimulationSession
from .sstim_advanced import (
    _ADVANCED, from_sstim_advanced_graph, to_sstim_advanced_graph,
)
from .sstim_io import (
    MUSIC, _GENERATORS, _decimal, _graph,
    from_sstim_graph, to_sstim_graph,
)

_ALLOWED = set(_GENERATORS) | _ADVANCED


def _one(graph: Graph, subject, predicate: URIRef):
    values = list(graph.objects(subject, predicate))
    if len(values) != 1:
        raise ValueError(f"expected one {predicate}")
    return values[0]


def to_sstim_program_graph(
        session: StimulationSession, *,
        base: str = "https://example.org/music/program/") -> Graph:
    """Export a MUSIC session as ordered SSTIM stimulus descriptions.

    Only supported generator callables (not pre-rendered arrays, external
    closures or custom waveform tables) can be encoded. Engine-owned terms
    preserve gain, phase order, ramp shape and fade times exactly.

    Raises
    ------
    ValueError
        On unsupported phases, non-finite controls, or invalid lengths.
    """
    if (not isinstance(base, str)
            or not base.startswith(("http://", "https://"))
            or not base.endswith(("/", "#"))
            or base.startswith("https://w3id.org/sstim")):
        raise ValueError("base must be an http(s) non-SSTIM namespace")
    if session.ramp_shape not in ("equal_power", "linear"):
        raise ValueError("unsupported ramp shape")
    if (not math.isfinite(session.end_ramp)
            or session.end_ramp < 0):
        raise ValueError("end_ramp must be finite and nonnegative")
    if not session.phases or len(session.phases) > 50:
        raise ValueError("program must have 1 to 50 phases")
    if any(not math.isfinite(p.duration) or p.duration < 0
           for p in session.phases):
        raise ValueError("invalid duration, ramp, or gain")
    if session.duration * session.sample_rate > 5_000_000:
        raise ValueError("program exceeds 5 million output samples")
    g = Graph()
    g.bind("music-impl", MUSIC)
    root = URIRef(base + "program")
    g.add((root, RDF.type, MUSIC.StimulationProgram))
    g.add((root, RDFS.label, Literal("MUSIC stimulus program")))
    g.add((root, MUSIC.sampleRateHz,
           Literal(session.sample_rate, datatype=XSD.integer)))
    g.add((root, MUSIC.rampShape, Literal(session.ramp_shape)))
    g.add((root, MUSIC.endRampSeconds, _decimal(session.end_ramp)))

    for index, phase in enumerate(session.phases, 1):
        name = getattr(phase.stimulus, "__name__", None)
        if name not in _ALLOWED or (
                phase.stimulus is not getattr(stimuli, name)):
            raise ValueError("unsupported or nonportable stimulus phase")
        if (not all(math.isfinite(x) for x in
                    (phase.duration, phase.ramp, phase.gain))
                or phase.duration <= 0 or phase.ramp < 0
                or phase.gain < 0):
            raise ValueError("invalid duration, ramp, or gain")
        stem = base + f"phase-{index}/"
        if name in _ADVANCED:
            graph = to_sstim_advanced_graph(
                name, parameters=phase.parameters,
                duration=phase.duration,
                sample_rate=session.sample_rate, base=stem)
        else:
            graph = to_sstim_graph(
                name, parameters=phase.parameters,
                duration=phase.duration,
                sample_rate=session.sample_rate, base=stem)
        g += graph
        part = URIRef(base + f"phase-{index}")
        g.add((root, MUSIC.hasPhase, part))
        g.add((part, RDF.type, MUSIC.StimulationPhase))
        g.add((part, MUSIC.order,
               Literal(index, datatype=XSD.integer)))
        g.add((part, MUSIC.phaseStimulus, URIRef(stem + "specification")))
        g.add((part, MUSIC.phaseDurationSeconds, _decimal(phase.duration)))
        g.add((part, MUSIC.rampSeconds, _decimal(phase.ramp)))
        g.add((part, MUSIC.gain, _decimal(phase.gain)))
        g.add((part, MUSIC.phaseLabel, Literal(phase.label)))
    return g


def from_sstim_program_graph(
        value: Graph | str | Path) -> StimulationSession:
    """Read a checked program without importing code from RDF strings.

    All generator names are resolved through the internal allowlist.
    The fully reconstructed graph must match, protecting against altered
    orders, signals and channels. Extra triples are intentionally refused.

    Raises
    ------
    ValueError
        For malformed or contradictory session definitions.
    """
    g = _graph(value)
    roots = list(g.subjects(RDF.type, MUSIC.StimulationProgram))
    if len(roots) != 1:
        raise ValueError("expected one MUSIC StimulationProgram")
    root = roots[0]
    base = str(root)
    if not base.endswith("program"):
        raise ValueError("unexpected program IRI")
    base = base[:-len("program")]
    try:
        rate = int(_one(g, root, MUSIC.sampleRateHz))
        shape = str(_one(g, root, MUSIC.rampShape))
        end_ramp = float(_one(g, root, MUSIC.endRampSeconds))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid program metadata") from exc
    session = StimulationSession(
        sample_rate=rate, ramp_shape=shape, end_ramp=end_ramp)
    phases = list(g.objects(root, MUSIC.hasPhase))
    if not phases or len(phases) > 50:
        raise ValueError("program must have 1 to 50 phases")
    ordered = []
    for phase in phases:
        order = int(_one(g, phase, MUSIC.order))
        ordered.append((order, phase))
    ordered.sort()
    if [n for n, _ in ordered] != list(range(1, len(phases) + 1)):
        raise ValueError("phase order must be unique and contiguous")

    for number, part in ordered:
        stimulus = _one(g, part, MUSIC.phaseStimulus)
        stem = base + f"phase-{number}/"
        if stimulus != URIRef(stem + "specification"):
            raise ValueError("phase stimulus IRI mismatch")
        scoped = Graph()
        for s, p, o in g:
            if isinstance(s, URIRef) and str(s).startswith(stem):
                scoped.add((s, p, o))
        if not scoped:
            raise ValueError("missing stimulus subgraph")
        name_values = list(scoped.objects(stimulus, MUSIC.generator))
        if len(name_values) != 1:
            raise ValueError("missing executable generator")
        name = str(name_values[0])
        if name in _ADVANCED:
            decoded = from_sstim_advanced_graph(scoped)
        elif name in _GENERATORS:
            d = from_sstim_graph(scoped)
            decoded = (d.generator, d.parameters, d.duration, d.sample_rate)
        else:
            raise ValueError("unknown generator")
        gen, params, duration, sample_rate = decoded
        if sample_rate != rate or float(
                _one(g, part, MUSIC.phaseDurationSeconds)) != duration:
            raise ValueError("phase rate or duration mismatch")
        session.add(
            getattr(stimuli, gen), duration=duration,
            ramp=float(_one(g, part, MUSIC.rampSeconds)),
            gain=float(_one(g, part, MUSIC.gain)),
            label=str(_one(g, part, MUSIC.phaseLabel)), **params)

    if set(g) != set(to_sstim_program_graph(session, base=base)):
        raise ValueError("program triples contradict executable definition")
    return session


def render_sstim_program(value: Graph | str | Path):
    """Render the encoded phase sequence after checking all its triples.

    Returns
    -------
    ndarray
        Mono or stereo samples, preserving the session timeline.
    """
    return from_sstim_program_graph(value).render()
