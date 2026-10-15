"""Sessions as SSTIM descriptions: programs, and plans around them."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
import json
import math
from numbers import Real

import numpy as np
from numpy.typing import NDArray
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, RDF, RDFS, XSD

from .. import stimuli
from ..session import StimulationSession
from ._stimuli import (
    STIMULI, checked, read_stimulus, stimulus_graph, under,
)
from ._vocabulary import (
    MUSIC, SSTIM, SSTIM_V, bind, check_base, decimal, difference, one,
    significant,
)

#: The algorithm a plan's configuration digest is computed with.
DIGEST_ALGORITHM = "sha256-music-plan-json-v1"


@dataclass(frozen=True)
class SessionPlan:
    """What SSTIM's SessionSpecification needs beyond the session itself.

    Attributes
    ----------
    preset_iri : str
        An http(s) IRI naming the preset, outside SSTIM's namespace.
    preset_label : str
        The preset's name, not empty.
    created_at : datetime
        When the plan was made, with its time zone.
    master_volume : float
        The playback level, from 0 to 1.
    """

    preset_iri: str
    preset_label: str
    created_at: datetime
    master_volume: float = 1.0


@dataclass(frozen=True)
class PlannedSession:
    """A session, and the plan a SessionSpecification made of it.

    Attributes
    ----------
    session : StimulationSession
        The session, as the program in the description gives it.
    plan : SessionPlan
        The preset, the creation time and the master volume.
    """

    session: StimulationSession
    plan: SessionPlan

    def render(self) -> NDArray[np.float64]:
        """The session's samples, as ``session.render()`` gives them."""
        return self.session.render()


def program_graph(session: StimulationSession, base: str) -> Graph:
    """`session` as a program in the package's terms, its stimuli in SSTIM's.

    SSTIM 0.19 has no terms for phases in order, their ramps or their
    gains, so the program is the package's; each phase's stimulus is an
    SSTIM StimulusSpecification. Nothing here was delivered, so it is
    never an ``sstim:SessionInstance``.
    """
    check_base(base)
    if not isinstance(session, StimulationSession):
        raise TypeError("expected a StimulationSession")
    if session.ramp_shape not in ("equal_power", "linear"):
        raise ValueError("the ramp shape must be equal_power or linear")
    if not math.isfinite(session.end_ramp) or session.end_ramp < 0:
        raise ValueError("end_ramp must be finite and not negative")
    if not 1 <= len(session.phases) <= 50:
        raise ValueError("a session must have from 1 to 50 phases")
    for phase in session.phases:
        name = getattr(phase.stimulus, "__name__", None)
        if name not in STIMULI or phase.stimulus is not getattr(stimuli,
                                                                name):
            raise ValueError(
                "a phase must be one of the seven stimuli, by its function: "
                "a rendered array, another callable or a custom table has "
                "no description")
        if (not all(math.isfinite(value) for value in
                    (phase.duration, phase.ramp, phase.gain))
                or phase.duration <= 0 or phase.ramp < 0 or phase.gain < 0):
            raise ValueError(
                "a phase's duration must be positive, and its ramp and gain "
                "finite and not negative")
    graph = bind(Graph())
    root = URIRef(base + "program")
    graph.add((root, RDF.type, MUSIC.StimulationProgram))
    graph.add((root, RDFS.label, Literal("MUSIC stimulus program")))
    graph.add((root, MUSIC.sampleRateHz,
               Literal(session.sample_rate, datatype=XSD.integer)))
    graph.add((root, MUSIC.rampShape, Literal(session.ramp_shape)))
    graph.add((root, MUSIC.endRampSeconds, decimal(session.end_ramp)))
    for index, phase in enumerate(session.phases, 1):
        stem = base + f"phase-{index}/"
        name = getattr(phase.stimulus, "__name__")
        graph += stimulus_graph(checked(name, phase.parameters,
                                        phase.duration,
                                        session.sample_rate), stem)
        part = URIRef(base + f"phase-{index}")
        graph.add((root, MUSIC.hasPhase, part))
        graph.add((part, RDF.type, MUSIC.StimulationPhase))
        graph.add((part, MUSIC.order, Literal(index, datatype=XSD.integer)))
        graph.add((part, MUSIC.phaseStimulus, URIRef(stem + "specification")))
        graph.add((part, MUSIC.phaseDurationSeconds, decimal(phase.duration)))
        graph.add((part, MUSIC.rampSeconds, decimal(phase.ramp)))
        graph.add((part, MUSIC.gain, decimal(phase.gain)))
        graph.add((part, MUSIC.phaseLabel, Literal(phase.label)))
    return graph


def read_program(graph: Graph) -> StimulationSession:
    """The session a program the package wrote gives, and checked."""
    roots = list(graph.subjects(RDF.type, MUSIC.StimulationProgram))
    if len(roots) != 1:
        raise ValueError("expected exactly one MUSIC StimulationProgram")
    root = roots[0]
    if not str(root).endswith("program"):
        raise ValueError(
            "this is not a program the package wrote: it is not named "
            "...program")
    base = str(root)[:-len("program")]
    try:
        rate = int(one(graph, root, MUSIC.sampleRateHz))
        shape = str(one(graph, root, MUSIC.rampShape))
        end_ramp = float(one(graph, root, MUSIC.endRampSeconds))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"the program's settings are unreadable: {exc}"
                         ) from exc
    session = StimulationSession(sample_rate=rate, ramp_shape=shape,
                                 end_ramp=end_ramp)
    parts = list(graph.objects(root, MUSIC.hasPhase))
    if not 1 <= len(parts) <= 50:
        raise ValueError("a session must have from 1 to 50 phases")
    ordered = sorted((int(one(graph, part, MUSIC.order)), str(part))
                     for part in parts)
    if [number for number, _ in ordered] != list(range(1, len(parts) + 1)):
        raise ValueError("the phases must be numbered 1, 2, 3 and so on")
    for number, name in ordered:
        part, stem = URIRef(name), base + f"phase-{number}/"
        if one(graph, part, MUSIC.phaseStimulus) != URIRef(
                stem + "specification"):
            raise ValueError(f"phase {number} names another phase's stimulus")
        scoped = Graph()
        for statement in graph:
            if str(statement[0]).startswith(stem):
                scoped.add(statement)
        stimulus = read_stimulus(scoped)
        if stimulus.sample_rate != rate or float(one(
                graph, part, MUSIC.phaseDurationSeconds)) != stimulus.duration:
            raise ValueError(
                f"phase {number}'s stimulus has another rate or length than "
                f"the phase")
        session.add(getattr(stimuli, stimulus.generator),
                    duration=stimulus.duration,
                    ramp=float(one(graph, part, MUSIC.rampSeconds)),
                    gain=float(one(graph, part, MUSIC.gain)),
                    label=str(one(graph, part, MUSIC.phaseLabel)),
                    **stimulus.parameters)
    found = under(significant(graph), base)
    expected = significant(program_graph(session, base))
    if found != expected:
        raise ValueError(
            "the program's statements are not those the package writes for "
            "its session: " + difference(found, expected))
    return session


def _check_plan(plan: SessionPlan) -> None:
    if not isinstance(plan, SessionPlan):
        raise TypeError("expected a SessionPlan")
    if (not isinstance(plan.preset_iri, str)
            or not plan.preset_iri.startswith(("https://", "http://"))
            or plan.preset_iri.startswith("https://w3id.org/sstim")):
        raise ValueError(
            "the preset must be named by an http(s) IRI outside SSTIM's")
    if not isinstance(plan.preset_label, str) or not plan.preset_label.strip():
        raise ValueError("the preset must have a name")
    if (not isinstance(plan.created_at, datetime)
            or plan.created_at.utcoffset() is None):
        raise ValueError("the creation time must be a datetime with its "
                         "time zone")
    if (isinstance(plan.master_volume, bool)
            or not isinstance(plan.master_volume, Real)
            or not math.isfinite(plan.master_volume)
            or not 0 <= plan.master_volume <= 1):
        raise ValueError("the master volume must be a number from 0 to 1")


def plan_graph(session: StimulationSession, plan: SessionPlan,
               base: str) -> Graph:
    """`session` as SSTIM's own SessionSpecification, around its program.

    SSTIM's Full profile takes a specification of a whole number of
    seconds, from 60 to 7,200, with a preset, a creation time and a master
    volume, which `plan` gives. It declares only
    ``reproEquivalentPresentation``, and a digest of the program, the
    preset, the volume and the duration. It is a plan: no
    ``SessionInstance``, event or exposure is written.
    """
    _check_plan(plan)
    duration = session.duration
    if (not math.isfinite(duration) or not 60 <= duration <= 7200
            or not math.isclose(duration, round(duration), rel_tol=0,
                                abs_tol=1e-8)):
        raise ValueError(
            "SSTIM plans a session of a whole number of seconds, from 60 to "
            f"7200; this one lasts {duration:g}")
    program = program_graph(session, base)
    graph = bind(Graph())
    graph += program
    root = URIRef(base + "session-specification")
    preset = URIRef(plan.preset_iri)
    graph.add((root, RDF.type, SSTIM.SessionSpecification))
    graph.add((root, RDFS.label, Literal("MUSIC planned SSTIM session")))
    graph.add((root, DCTERMS.created, Literal(
        plan.created_at.isoformat(), datatype=XSD.dateTime)))
    graph.add((root, SSTIM.referencesPreset, preset))
    graph.add((preset, RDF.type, SSTIM.Preset))
    graph.add((preset, RDFS.label, Literal(plan.preset_label)))
    graph.add((root, SSTIM.durationSeconds,
               Literal(round(duration), datatype=XSD.integer)))
    graph.add((root, SSTIM.masterVolume, decimal(plan.master_volume)))
    # The digest is of the program as written, defaults and all, and of
    # the plan's own choices, so it changes with any of them.
    lines = sorted(line.strip() for line in
                   program.serialize(format="nt").splitlines()
                   if line.strip())
    canonical = json.dumps({
        "program_ntriples": lines,
        "preset_iri": plan.preset_iri,
        "master_volume": format(
            Decimal(str(float(plan.master_volume))).normalize(), "f"),
        "duration_seconds": int(round(duration)),
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    graph.add((root, SSTIM.configurationDigest, Literal(
        sha256(canonical.encode("utf-8")).hexdigest(),
        datatype=XSD.string)))
    graph.add((root, SSTIM.digestAlgorithm,
               Literal(DIGEST_ALGORITHM, datatype=XSD.string)))
    graph.add((root, SSTIM.hasReproducibilityLevel,
               SSTIM_V.reproEquivalentPresentation))
    graph.add((root, MUSIC.hasProgram, URIRef(base + "program")))
    return graph


def read_plan(graph: Graph) -> PlannedSession:
    """The session and plan a SessionSpecification the package wrote gives."""
    roots = list(graph.subjects(RDF.type, SSTIM.SessionSpecification))
    if len(roots) != 1:
        raise ValueError("expected exactly one SSTIM SessionSpecification")
    root = roots[0]
    if not str(root).endswith("session-specification"):
        raise ValueError(
            "this is not a plan the package wrote: it is not named "
            "...session-specification")
    base = str(root)[:-len("session-specification")]
    if one(graph, root, MUSIC.hasProgram) != URIRef(base + "program"):
        raise ValueError("the plan must name its own program")
    preset = one(graph, root, SSTIM.referencesPreset)
    if (preset, RDF.type, SSTIM.Preset) not in graph:
        raise ValueError("the preset must be an sstim:Preset")
    created = one(graph, root, DCTERMS.created)
    if not isinstance(created, Literal) or created.datatype != XSD.dateTime:
        raise ValueError("the creation time must be an xsd:dateTime")
    try:
        plan = SessionPlan(
            str(preset), str(one(graph, preset, RDFS.label)),
            datetime.fromisoformat(str(created).replace("Z", "+00:00")),
            float(one(graph, root, SSTIM.masterVolume)))
        duration = int(one(graph, root, SSTIM.durationSeconds))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"the plan's settings are unreadable: {exc}"
                         ) from exc
    program = Graph()
    for statement in graph:
        subject = str(statement[0])
        if subject == base + "program" or subject.startswith(
                base + "phase-"):
            program.add(statement)
    session = read_program(program)
    if not math.isclose(session.duration, duration, rel_tol=0,
                        abs_tol=1e-8):
        raise ValueError("the plan's duration is not its program's")
    also = (DCTERMS.created,)
    found = under(significant(graph, also), base)
    expected = under(significant(plan_graph(session, plan, base), also), base)
    if found != expected:
        raise ValueError(
            "the plan's statements are not those the package writes for it: "
            + difference(found, expected))
    return PlannedSession(session, plan)
