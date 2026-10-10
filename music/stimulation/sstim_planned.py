"""Conformant SSTIM 0.19 planned-session metadata around a MUSIC program.

A SessionSpecification is a *plan*, not evidence of delivery. SSTIM's
Full profile requires a preset reference, creation time, integer duration
60..7200 seconds, and masterVolume. A MUSIC phase program can supply
the underlying engine-specific sequence, but it does not by itself
satisfy this session-level contract.

The planned envelope is SSTIM; order, fades, and executable controls
remain MUSIC-owned. No SessionInstance, event, exposure or self-report
is synthesized.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
import json
import math
from pathlib import Path

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, RDF, RDFS, XSD

from .session import StimulationSession
from .sstim_io import (
    MUSIC, SSTIM, SSTIM_V, _decimal, _graph,
)
from .sstim_program import (
    from_sstim_program_graph, to_sstim_program_graph,
)


@dataclass(frozen=True)
class PlannedSession:
    """Validated SSTIM plan and locally executable MUSIC phase program."""

    session: StimulationSession
    preset_iri: str
    preset_label: str
    created_at: datetime
    master_volume: float


def _one(graph: Graph, subject, predicate):
    vals = list(graph.objects(subject, predicate))
    if len(vals) != 1:
        raise ValueError(f"expected exactly one {predicate}")
    return vals[0]


def to_sstim_planned_session_graph(
        session: StimulationSession, *, preset_iri: str,
        preset_label: str, created_at: datetime,
        master_volume: float = 1.0,
        base: str = "https://example.org/music/plan/") -> Graph:
    """Publish a properly scoped *planned* SSTIM SessionSpecification.

    Refuses nonintegral or outside-range sessions rather than inventing
    a normative duration. The caller identifies a real preset and passes
    the timestamp; this function never pretends to have observed delivery.
    Uses the conservative published reproEquivalentPresentation level.
    """
    if (not isinstance(preset_iri, str)
            or not preset_iri.startswith(("https://", "http://"))
            or preset_iri.startswith("https://w3id.org/sstim")):
        raise ValueError("preset_iri must identify an external preset")
    if not isinstance(preset_label, str) or not preset_label.strip():
        raise ValueError("preset_label must be nonempty")
    if (not isinstance(created_at, datetime)
            or created_at.tzinfo is None
            or created_at.utcoffset() is None):
        raise ValueError("created_at requires a timezone-aware datetime")
    if (not isinstance(master_volume, (float, int))
            or isinstance(master_volume, bool)
            or not math.isfinite(master_volume)
            or not 0 <= master_volume <= 1):
        raise ValueError("master_volume must be finite and in [0, 1]")
    duration = session.duration
    if (not math.isfinite(duration) or not 60 <= duration <= 7200
            or not math.isclose(duration, round(duration),
                                rel_tol=0, abs_tol=1e-8)):
        raise ValueError("SSTIM session duration must be integer 60..7200")
    program = to_sstim_program_graph(session, base=base)
    g = Graph()
    g += program
    root = URIRef(base + "session-specification")
    preset = URIRef(preset_iri)
    g.add((root, RDF.type, SSTIM.SessionSpecification))
    g.add((root, RDFS.label, Literal("MUSIC planned SSTIM session")))
    g.add((root, DCTERMS.created, Literal(
        created_at.isoformat(), datatype=XSD.dateTime)))
    g.add((root, SSTIM.referencesPreset, preset))
    g.add((preset, RDF.type, SSTIM.Preset))
    g.add((preset, RDFS.label, Literal(preset_label)))
    g.add((root, SSTIM.durationSeconds,
           Literal(round(duration), datatype=XSD.integer)))
    g.add((root, SSTIM.masterVolume, _decimal(master_volume)))
    # Recompute from the *serialized MUSIC configuration*, not from the
    # caller's sparse phase kwargs: default controls are materialized by
    # the exporter and must contribute to the configuration identity.
    lines = sorted(line.strip() for line in
                   program.serialize(format="nt").splitlines()
                   if line.strip())
    # A plan's playback level and selected preset are configuration too.
    # An acoustic-configuration digest must include those overrides.
    canonical = json.dumps({
        "program_ntriples": lines,
        "preset_iri": preset_iri,
        "master_volume": format(
            Decimal(str(float(master_volume))).normalize(), "f"),
        "duration_seconds": int(round(duration)),
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = sha256(canonical.encode("utf-8")).hexdigest()
    g.add((root, SSTIM.configurationDigest,
           Literal(digest, datatype=XSD.string)))
    g.add((root, SSTIM.digestAlgorithm, Literal(
        "sha256-music-plan-json-v1", datatype=XSD.string)))
    g.add((root, SSTIM.hasReproducibilityLevel,
           SSTIM_V.reproEquivalentPresentation))
    g.add((SSTIM_V.reproEquivalentPresentation, RDF.type,
           SSTIM.ReproducibilityLevel))
    g.add((root, MUSIC.hasProgram, URIRef(base + "program")))
    return g


def from_sstim_planned_session_graph(
        value: Graph | str | Path) -> PlannedSession:
    """Check the SSTIM session metadata and strict MUSIC program subgraph.

    This reader is for the exact, explicitly versioned MUSIC envelope.
    It does not claim to understand every third-party SessionSpecification
    or infer playback timing from an RDF statement order.
    """
    g = _graph(value)
    roots = list(g.subjects(RDF.type, SSTIM.SessionSpecification))
    if len(roots) != 1:
        raise ValueError("expected one planned SessionSpecification")
    root = roots[0]
    base = str(root)
    if not base.endswith("session-specification"):
        raise ValueError("invalid planned-session IRI")
    base = base[:-len("session-specification")]
    if _one(g, root, MUSIC.hasProgram) != URIRef(base + "program"):
        raise ValueError("planned session must reference its MUSIC program")
    preset = _one(g, root, SSTIM.referencesPreset)
    label = str(_one(g, preset, RDFS.label))
    if (preset, RDF.type, SSTIM.Preset) not in g:
        raise ValueError("referenced preset must be typed sstim:Preset")
    created = _one(g, root, DCTERMS.created)
    if not isinstance(created, Literal) or created.datatype != XSD.dateTime:
        raise ValueError("created timestamp must be xsd:dateTime")
    try:
        created_at = datetime.fromisoformat(
            str(created).replace("Z", "+00:00"))
        volume = float(_one(g, root, SSTIM.masterVolume))
        duration = int(_one(g, root, SSTIM.durationSeconds))
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid planned-session metadata") from exc
    if _one(g, root, SSTIM.hasReproducibilityLevel) != (
            SSTIM_V.reproEquivalentPresentation):
        raise ValueError("unsupported reproducibility claim")
    subgraph = Graph()
    for subject, predicate, obj in g:
        if (isinstance(subject, URIRef)
                and (str(subject) == base + "program"
                     or str(subject).startswith(base + "phase-"))):
            subgraph.add((subject, predicate, obj))
    session = from_sstim_program_graph(subgraph)
    if session.duration != duration:
        raise ValueError("planned duration contradicts program")
    expected = to_sstim_planned_session_graph(
        session, base=base, preset_iri=str(preset),
        preset_label=label, created_at=created_at,
        master_volume=volume)
    if set(g) != set(expected):
        raise ValueError("planned SSTIM session contradicts MUSIC plan")
    return PlannedSession(
        session, str(preset), label, created_at, volume)
