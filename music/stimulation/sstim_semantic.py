"""Engine-independent inspection of a narrow SSTIM 0.19 beat profile.

This consumer reads SSTIM terms alone. It never relies on MUSIC generator
names or parametersJson, imports code from RDF, or executes graph-supplied
instructions. The reference renderer explicitly chooses zero carrier phase
and (for monaural beats) equal-amplitude tones. SSTIM does not guarantee
those choices, so sample-exact agreement with a third-party engine is
NOT a consequence of ontology conformance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rdflib import Graph
from rdflib.namespace import RDF

from .sstim_io import SSTIM, SSTIM_EX, SSTIM_V, _graph

__all__ = ["SemanticBeat", "inspect_sstim_beat", "render_semantic_beat"]

# Non-normative, explicitly named, reproducible analytic rendering convention.
_PROFILE = "zero-phase-equal-gain-beats-v1"


@dataclass(frozen=True)
class SemanticBeat:
    """A portable beat's interpretable parameters, without MUSIC hints.

    The SSTIM graph establishes carrier frequencies, difference/envelope
    rate, modality, delivery and mechanism. Exact amplitude and initial
    carrier phase are NOT established by those triples.
    """

    technique: str
    carrier_freq: float
    beat_freq: float
    duration: float
    sample_rate: int


def _one(g: Graph, subject, predicate):
    terms = list(g.objects(subject, predicate))
    if len(terms) != 1:
        raise ValueError(f"expected exactly one {predicate}")
    return terms[0]


def _number(g: Graph, node, predicate) -> float:
    try:
        value = float(str(_one(g, node, predicate)))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid numeric {predicate}") from exc
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {predicate}")
    return value


def inspect_sstim_beat(
        value: Graph | str | Path, *,
        sample_rate: int = 44100) -> SemanticBeat:
    """Inspect a supported mono/monaural or stereo/binaural SSTIM beat.

    The accepted profile requires one determinate sine signal, a fixed
    beat rate, exactly one rendering per channel, air-conducted auditory
    delivery, standard placements, correct physical/perceptual presence,
    and consistent carriers/duration. Reject all other techniques and
    ambiguous or incomplete RDF rather than guessing implementation
    parameters such as a duty cycle or modulation depth.

    The caller must supply the intended *output* sampling rate because
    SSTIM's engine-independent specification need not select one.
    """
    if (not isinstance(sample_rate, int) or isinstance(sample_rate, bool)
            or not 8000 <= sample_rate <= 192000):
        raise ValueError("sample_rate must be in [8000, 192000]")
    g = _graph(value)
    roots = list(g.subjects(RDF.type, SSTIM.StimulusSpecification))
    if len(roots) != 1:
        raise ValueError("expected one SSTIM stimulus specification")
    root = roots[0]
    if str(_one(g, root, SSTIM.stimulusRegime)) != "determinate":
        raise ValueError("only determinate beats are supported")
    signal = _one(g, root, SSTIM.hasSignal)
    if (signal, RDF.type, SSTIM.StimulationSignal) not in g:
        raise ValueError("stimulus signal class missing")
    if _one(g, signal, SSTIM.hasSignalShape) != SSTIM_V.shapeSine:
        raise ValueError("only a sine beat signal is supported")
    beat = _number(g, signal, SSTIM.hzMin)
    if beat <= 0 or _number(g, signal, SSTIM.hzMax) != beat:
        raise ValueError("beat must have a fixed positive rate")
    channels = list(g.objects(root, SSTIM.hasStimulusChannel))
    if len(channels) not in (1, 2):
        raise ValueError("expected one or two auditory channels")

    carriers = {}
    durations = set()
    methods = set()
    presences = set()
    for channel in channels:
        if (channel, RDF.type, SSTIM_EX.StimulusChannel) not in g:
            raise ValueError("channel class missing")
        if (_one(g, channel, SSTIM_EX.perceivedModality)
                != SSTIM_EX.modalityAuditory):
            raise ValueError("requires auditory modality")
        if (_one(g, channel, SSTIM_EX.deliveryMedium)
                != SSTIM_EX.mediumAirConductedSound):
            raise ValueError("requires air-conducted sound")
        duration = _number(g, channel, SSTIM.channelDurationSeconds)
        durations.add(duration)
        placement = _one(g, channel, SSTIM_EX.hasBodyPlacement)
        if placement in carriers:
            raise ValueError("duplicate channel placement")
        rendering = _one(g, channel, SSTIM.hasSignalRendering)
        if (rendering, RDF.type, SSTIM.SignalRendering) not in g:
            raise ValueError("rendering class missing")
        if _one(g, rendering, SSTIM.rendersSignal) != signal:
            raise ValueError("rendering targets another signal")
        if (_one(g, rendering, SSTIM.rendersOntoParameter)
                != SSTIM_V.paramAmplitude):
            raise ValueError("expected amplitude rendering")
        methods.add(_one(g, rendering, SSTIM.hasRenderingMechanism))
        presences.add(_one(g, rendering, SSTIM.hasRenderingPresence))
        carriers[placement] = _number(
            g, rendering, SSTIM.renderingCarrierHz)

    if len(durations) != 1:
        raise ValueError("channel durations differ")
    duration = durations.pop()
    if (duration <= 0 or int(duration * sample_rate) <= 0
            or int(duration * sample_rate) > 5_000_000):
        raise ValueError("invalid or excessive beat duration")

    if len(channels) == 2:
        if (set(carriers) != {SSTIM_EX.placementEarLeft,
                              SSTIM_EX.placementEarRight}
                or methods != {SSTIM_V.mechanismBinauralBeat}
                or presences != {SSTIM_V.presencePerceptual}):
            raise ValueError("unsupported binaural channel semantics")
        left, right = (carriers[SSTIM_EX.placementEarLeft],
                       carriers[SSTIM_EX.placementEarRight])
        if (left <= 0 or right >= sample_rate / 2 or left >= right
                or not math.isclose(right - left, beat,
                                    rel_tol=1e-9, abs_tol=1e-7)):
            raise ValueError("binaural carriers contradict beat rate")
        return SemanticBeat("binaural", (left + right) / 2,
                            beat, duration, sample_rate)

    if (set(carriers) != {SSTIM_EX.placementEars}
            or methods != {SSTIM_V.mechanismMonauralBeat}
            or presences != {SSTIM_V.presencePhysical}):
        raise ValueError("unsupported monaural channel semantics")
    carrier = carriers[SSTIM_EX.placementEars]
    if (carrier - beat / 2 <= 0
            or carrier + beat / 2 >= sample_rate / 2):
        raise ValueError("monaural carriers outside Nyquist")
    return SemanticBeat("monaural", carrier, beat, duration, sample_rate)


def render_semantic_beat(
        value: SemanticBeat | Graph | str | Path, *, profile: str,
        sample_rate: int = 44100) -> NDArray[np.float64]:
    """Analytic independent reference, NOT a universal SSTIM DSP engine.

    `profile='zero-phase-equal-gain-beats-v1'` declares assumptions:
    both sine carriers start at phase zero, and monaural audio is their
    equal-gain mean. Amplitude calibration and physical playback are not
    represented. Compare spectral/temporal properties across engines,
    not exact DAC samples or neural effects.
    """
    if profile != _PROFILE:
        raise ValueError("explicit zero-phase-equal-gain-beats-v1 required")
    spec = (value if isinstance(value, SemanticBeat)
            else inspect_sstim_beat(value, sample_rate=sample_rate))
    count = int(spec.duration * spec.sample_rate)
    if count <= 0 or count > 5_000_000:
        raise ValueError("invalid sample count")
    t = np.arange(count) / spec.sample_rate
    low = np.sin(2 * np.pi * (spec.carrier_freq - spec.beat_freq / 2) * t)
    high = np.sin(2 * np.pi * (spec.carrier_freq + spec.beat_freq / 2) * t)
    if spec.technique == "binaural":
        return np.vstack((low, high))
    if spec.technique == "monaural":
        return (low + high) / 2
    raise ValueError("unsupported semantic technique")
