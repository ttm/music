"""Explain auditable SSTIM interchange capabilities without speculative DSP.

Status is a *local renderer capability*, not an SSTIM validity judgment.
No unknown RDF program can dispatch Python code. MUSIC replay requires
the fully checked MUSIC extension; reference beats require only SSTIM
triples and an explicitly named, non-normative rendering convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rdflib import Graph
from rdflib.namespace import RDF

from .sstim_io import SSTIM, SSTIM_V, MUSIC, _graph


@dataclass(frozen=True)
class InteroperabilityCapability:
    """Description of a supported, partial or unsupported decode path."""

    mode: str
    technique: str
    can_render: bool
    exact_pcm_portable: bool
    missing_contract: tuple[str, ...]


def inspect_sstim_capabilities(
        value: Graph | str | Path) -> InteroperabilityCapability:
    """Negotiate renderer support without requiring the optional SSTIM client.

    Portable binaural and monaural beats can render only under MUSIC's
    explicitly selected analytic convention. Other stimuli are inspected
    for a known MUSIC-owned executable profile; if absent or contradictory,
    report the missing cross-engine contract rather than guessing values.
    Even exact MUSIC replay is not portable *across arbitrary engines*.
    """
    from .sstim_semantic import inspect_sstim_beat

    graph = _graph(value)
    programs = list(graph.subjects(RDF.type, MUSIC.StimulationProgram))
    if programs:
        try:
            from .sstim_program import from_sstim_program_graph
            from_sstim_program_graph(graph)
        except (ValueError, TypeError, KeyError):
            return InteroperabilityCapability(
                "unsupported", "program", False, False,
                ("invalid or incomplete MUSIC program",))
        return InteroperabilityCapability(
            "music-program", "program", True, False,
            ("MUSIC phase ordering, mixing, fading and engine version",))

    roots = list(graph.subjects(RDF.type, SSTIM.StimulusSpecification))
    if len(roots) != 1:
        return InteroperabilityCapability(
            "unsupported", "unknown", False, False,
            ("exactly one SSTIM StimulusSpecification",))

    # A graph that claims MUSIC executability must pass the MUSIC
    # validator first, even if its SSTIM-only beat projection is usable.
    # Otherwise contradictory numeric hints could be silently ignored.
    owned = [(sub, pred, obj) for sub, pred, obj in graph
             if str(pred).startswith(str(MUSIC))]
    engines = list(graph.objects(roots[0], MUSIC.generator))
    if owned:
        if len(engines) != 1:
            return InteroperabilityCapability(
                "unsupported", "unknown", False, False,
                ("incomplete or ambiguous MUSIC extension",))
        name = str(engines[0])
        try:
            if name in ("modulated_noise", "spatial_motion"):
                from .sstim_advanced import from_sstim_advanced_graph
                from_sstim_advanced_graph(graph)
            else:
                from .sstim_io import from_sstim_graph
                from_sstim_graph(graph)
        except (ValueError, TypeError, KeyError, AttributeError):
            return InteroperabilityCapability(
                "unsupported", name, False, False,
                ("invalid or contradictory MUSIC engine metadata",))
        return InteroperabilityCapability(
            "music-extension", name, True, False,
            ("MUSIC implementation and DSP algorithm identity",))

    try:
        beat = inspect_sstim_beat(graph)
    except (ValueError, TypeError):
        pass
    else:
        return InteroperabilityCapability(
            "portable-beat-reference", beat.technique, True, False,
            ("explicit carrier phase and gain convention",
             "physical output calibration"))

    # Read only *published* SSTIM mechanisms and signal shapes, not labels
    # that may contain arbitrary user-authored language or generator hints.
    shapes = set(graph.objects(None, SSTIM.hasSignalShape))
    mechanisms = set(graph.objects(None, SSTIM.hasRenderingMechanism))
    parameters = set(graph.objects(None, SSTIM.rendersOntoParameter))
    if SSTIM_V.shapeNoise in shapes:
        return InteroperabilityCapability(
            "descriptive-only", "noise", False, False,
            ("noise color/spectral model, random algorithm and seed",
             "envelope/ramp convention, calibrated gain"))
    if SSTIM_V.paramSpatialPosition in parameters:
        return InteroperabilityCapability(
            "descriptive-only", "spatial", False, False,
            ("spatial trajectory, geometry and localization model",))
    if SSTIM_V.mechanismFrequencyModulation in mechanisms:
        return InteroperabilityCapability(
            "descriptive-only", "frequency-modulation", False, False,
            ("frequency deviation, phase and modulation waveform",))
    if SSTIM_V.mechanismAmplitudeModulation in mechanisms:
        return InteroperabilityCapability(
            "descriptive-only", "amplitude-modulation", False, False,
            ("duty cycle or depth, ramp and waveform convention",))
    return InteroperabilityCapability(
        "unsupported", "unknown", False, False,
        ("recognized audio rendering mechanism and parameters",))
