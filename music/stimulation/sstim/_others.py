"""Descriptions in SSTIM's terms alone, as another tool writes them."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import inspect
import math
from numbers import Real
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF

from .. import stimuli
from ._stimuli import (
    BEATS, COLOURS, TONES, SstimStimulus, checked, portable, stimulus_graph,
)
from ._vocabulary import SSTIM, SSTIM_EX, SSTIM_V

#: The convention a beat in SSTIM's terms alone can be rendered by: both
#: tones are sines starting at phase zero, and a monaural beat is their
#: mean. SSTIM fixes neither, so this names the choice instead of
#: presenting it as SSTIM's.
BEAT_PROFILE = "zero-phase-equal-gain-beats-v1"

_SSTIM_SPACE = "https://w3id.org/sstim"
_EXPECTED = "https://example.org/music/expected/"

#: The channel terms that state a number, each functional or not.
_VALUES = {
    SSTIM_EX.hasCarrierFrequencyHz: True,
    SSTIM_EX.hasToneFrequencyHz: True,
    SSTIM_EX.hasBeatFrequencyHz: True,
    SSTIM_EX.hasModulationFrequencyHz: False,
    SSTIM_EX.hasFrequencyHz: False,
    SSTIM_EX.hasDutyCycle: True,
}

#: Each kind of node a stimulus is made of: its class, the terms read
#: from it, and the terms that say something about it but nothing about
#: what it sounds like, which are left alone. Any other SSTIM term on it
#: is refused: it may change the stimulus in a way the package would not
#: render.
_KINDS = {
    "stimulus": (SSTIM.StimulusSpecification,
                 {SSTIM.stimulusRegime, SSTIM.hasSignal,
                  SSTIM.hasStimulusChannel},
                 {SSTIM.describesStimulation, SSTIM.hasStimulationTarget}),
    "signal": (SSTIM.StimulationSignal,
               {SSTIM.hasSignalShape, SSTIM.hzMin, SSTIM.hzMax},
               {SSTIM.signalCoversBand, SSTIM.signalOverlapsBand,
                SSTIM.signalWithinBand}),
    "channel": (SSTIM_EX.StimulusChannel,
                {SSTIM.channelDurationSeconds, SSTIM.hasSignalRendering,
                 SSTIM_EX.perceivedModality, SSTIM_EX.deliveryMedium,
                 SSTIM_EX.hasBodyPlacement, SSTIM_EX.hasStimulusPattern,
                 *_VALUES},
                {SSTIM_EX.channelRole, SSTIM_EX.hasComfortBoundary,
                 SSTIM_EX.requiresDeviceCapability}),
    "rendering": (SSTIM.SignalRendering,
                  {SSTIM.rendersSignal, SSTIM.rendersOntoParameter,
                   SSTIM.hasRenderingMechanism, SSTIM.hasRenderingPresence,
                   SSTIM.renderingCarrierHz},
                  set()),
}

_PREFIXES = ((str(SSTIM_EX), "sstim-ex:"), (str(SSTIM_V), "sstim-v:"),
             (str(SSTIM), "sstim:"))


def _term(term: Any) -> str:
    """`term` as a reader writes it: sstim-ex:hasDutyCycle, or <...>."""
    for space, prefix in _PREFIXES:
        if isinstance(term, URIRef) and str(term).startswith(space):
            return prefix + str(term)[len(space):]
    return term.n3()


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def _sstim(term: Any) -> bool:
    return isinstance(term, URIRef) and str(term).startswith(_SSTIM_SPACE)


def _number(term: Any, what: str) -> float:
    if isinstance(term, Literal):
        value = term.toPython()
        if (isinstance(value, (int, float, Decimal))
                and not isinstance(value, bool)
                and math.isfinite(float(value))):
            return float(value)
    raise ValueError(f"{what} is not a finite number: {_term(term)}")


def _objects(graph: Graph, node: Any, predicate: URIRef, where: str,
             required: bool = True) -> list:
    values = list(graph.objects(node, predicate))
    if required and not values:
        raise ValueError(f"{where} states no {_term(predicate)}")
    return values


def _single(graph: Graph, node: Any, predicate: URIRef, where: str,
            required: bool = True) -> Any:
    values = _objects(graph, node, predicate, where, required)
    if len(values) > 1:
        raise ValueError(f"{where} states more than one {_term(predicate)}")
    return values[0] if values else None


def _check_node(graph: Graph, node: Any, kind: str, where: str) -> None:
    """Refuse a node that is not of its kind, or says what isn't read."""
    cls, read, aside = _KINDS[kind]
    classes = set(graph.objects(node, RDF.type))
    if cls not in classes:
        raise ValueError(f"{where} is not typed {_term(cls)}")
    others = sorted((c for c in classes if _sstim(c) and c != cls),
                    key=str)
    if others:
        raise ValueError(f"{where} is also typed {_term(others[0])}, which "
                         f"the package does not render")
    unread = sorted((p for p in set(graph.predicates(node, None))
                     if _sstim(p) and p not in read and p not in aside),
                    key=str)
    if unread:
        raise ValueError(
            f"{where} states {', '.join(_term(p) for p in unread)}, which "
            f"the package cannot render as stated")


@dataclass(frozen=True)
class _Rendering:
    parameter: URIRef
    mechanism: URIRef
    presence: URIRef
    carrier: float | None


@dataclass(frozen=True)
class _Channel:
    modality: URIRef
    medium: URIRef
    duration: float | None
    renderings: dict[URIRef, _Rendering]
    values: dict[URIRef, tuple[float, ...]]
    patterns: frozenset


@dataclass(frozen=True)
class _Stimulus:
    regime: str
    signals: dict[URIRef, tuple[float, float]]
    channels: dict[URIRef, _Channel]


def _root(graph: Graph) -> Any:
    roots = list(graph.subjects(RDF.type, SSTIM.StimulusSpecification))
    if len(roots) != 1:
        raise ValueError("expected exactly one SSTIM StimulusSpecification")
    return roots[0]


def _value(graph: Graph, node: Any, predicate: URIRef, where: str) -> float:
    """The one number `node` states for `predicate`."""
    return _number(_single(graph, node, predicate, where),
                   f"{where}'s {_term(predicate)}")


def _optional(graph: Graph, node: Any, predicate: URIRef,
              where: str) -> float | None:
    """The one number `node` states for `predicate`, if it states one."""
    term = _single(graph, node, predicate, where, required=False)
    return None if term is None else _number(
        term, f"{where}'s {_term(predicate)}")


def _project(graph: Graph, root: Any) -> _Stimulus:
    """The stimulus at `root` in SSTIM's terms, apart from its nodes' names.

    Signals are told apart by their shape, channels by their placement and
    a channel's renderings by the shape of the signal each renders: the
    package's stimuli never have two of one, so a description that does
    is none of them.
    """
    _check_node(graph, root, "stimulus", "the stimulus")
    regime = _single(graph, root, SSTIM.stimulusRegime, "the stimulus")
    shapes: dict[Any, URIRef] = {}
    signals: dict[URIRef, tuple[float, float]] = {}
    for node in _objects(graph, root, SSTIM.hasSignal, "the stimulus"):
        shape = _single(graph, node, SSTIM.hasSignalShape, "a signal")
        where = f"the {_term(shape)} signal"
        _check_node(graph, node, "signal", where)
        if shape in signals:
            raise ValueError(f"the stimulus has two signals of shape "
                             f"{_term(shape)}, which none of the package's "
                             f"stimuli has")
        shapes[node] = shape
        signals[shape] = (_value(graph, node, SSTIM.hzMin, where),
                          _value(graph, node, SSTIM.hzMax, where))
    channels: dict[URIRef, _Channel] = {}
    for node in _objects(graph, root, SSTIM.hasStimulusChannel,
                         "the stimulus"):
        placement = _single(graph, node, SSTIM_EX.hasBodyPlacement,
                            "a channel")
        where = f"the {_term(placement)} channel"
        _check_node(graph, node, "channel", where)
        if placement in channels:
            raise ValueError(f"the stimulus has two channels at "
                             f"{_term(placement)}")
        renderings: dict[URIRef, _Rendering] = {}
        for rendering in _objects(graph, node, SSTIM.hasSignalRendering,
                                  where):
            within = f"a rendering of {where}"
            _check_node(graph, rendering, "rendering", within)
            signal = _single(graph, rendering, SSTIM.rendersSignal, within)
            if signal not in shapes:
                raise ValueError(f"{where} renders a signal the stimulus "
                                 f"does not have")
            if shapes[signal] in renderings:
                raise ValueError(f"{where} renders its "
                                 f"{_term(shapes[signal])} signal twice")
            renderings[shapes[signal]] = _Rendering(
                _single(graph, rendering, SSTIM.rendersOntoParameter, within),
                _single(graph, rendering, SSTIM.hasRenderingMechanism,
                        within),
                _single(graph, rendering, SSTIM.hasRenderingPresence, within),
                _optional(graph, rendering, SSTIM.renderingCarrierHz,
                          within))
        values = {}
        for predicate, functional in _VALUES.items():
            found = [_number(term, f"{where}'s {_term(predicate)}")
                     for term in graph.objects(node, predicate)]
            if functional and len(found) > 1:
                raise ValueError(f"{where} states more than one "
                                 f"{_term(predicate)}")
            if found:
                values[predicate] = tuple(found)
        channels[placement] = _Channel(
            _single(graph, node, SSTIM_EX.perceivedModality, where),
            _single(graph, node, SSTIM_EX.deliveryMedium, where),
            _optional(graph, node, SSTIM.channelDurationSeconds, where),
            renderings, values,
            frozenset(graph.objects(node, SSTIM_EX.hasStimulusPattern)))
    return _Stimulus(str(regime), signals, channels)


def _generator(found: _Stimulus) -> str:
    """Which of the package's stimuli `found` is, by its SSTIM terms."""
    renderings = [r for c in found.channels.values()
                  for r in c.renderings.values()]
    mechanisms = {r.mechanism for r in renderings}
    if SSTIM_V.shapeNoise in found.signals:
        return "modulated_noise"
    if SSTIM_V.paramSpatialPosition in {r.parameter for r in renderings}:
        return "spatial_motion"
    for generator, (mechanism, shape, _, _) in TONES.items():
        if (SSTIM_V[mechanism] in mechanisms
                and SSTIM_V[shape] in found.signals):
            return generator
    raise ValueError(
        "the description's signals and mechanisms are none of the "
        "package's stimuli: it renders "
        + ", ".join(sorted(_term(m) for m in mechanisms))
        + " with signals of shape "
        + ", ".join(sorted(_term(s) for s in found.signals)))


def _stated(generator: str, found: _Stimulus) -> dict[str, Any]:
    """The controls of `generator` that `found` states in SSTIM's terms."""
    signals, channels = found.signals, found.channels
    values: dict[str, Any] = {}

    def carrier(placement: URIRef) -> float | None:
        """The carrier the channel at `placement` states, if any."""
        channel = channels.get(placement)
        if channel is None:
            return None
        for rendering in channel.renderings.values():
            if rendering.carrier is not None:
                return rendering.carrier
        stated = channel.values.get(SSTIM_EX.hasCarrierFrequencyHz)
        return stated[0] if stated else None

    if generator in TONES:
        _, shape, _, rate_name = TONES[generator]
        rate = signals[SSTIM_V[shape]][0]
        values[rate_name] = rate
        if generator == "binaural_beats":
            left = carrier(SSTIM_EX.placementEarLeft)
            right = carrier(SSTIM_EX.placementEarRight)
            if (left is not None and right is not None
                    and not _close(right - left, rate)):
                raise ValueError(
                    f"its ears' carriers, {left:g} and {right:g} Hz, are "
                    f"not its {rate:g} Hz beat apart")
            centre = ((left + right) / 2 if left is not None
                      and right is not None else
                      left + rate / 2 if left is not None else
                      right - rate / 2 if right is not None else None)
        else:
            # Whichever channel states it: one at the wrong placement is
            # refused for its placement, not for a carrier it does state.
            centre = next((c for c in map(carrier, channels)
                           if c is not None), None)
        if centre is not None:
            values["carrier_freq"] = centre
        if generator == "isochronic_tones":
            duty = [c.values[SSTIM_EX.hasDutyCycle][0]
                    for c in channels.values()
                    if SSTIM_EX.hasDutyCycle in c.values]
            if duty:
                values["duty_cycle"] = duty[0]
    elif generator == "modulated_noise":
        values["min_freq"], values["max_freq"] = signals[SSTIM_V.shapeNoise]
        modulation = signals.get(SSTIM_V.shapeSine)
        values["modulation_freq"] = modulation[0] if modulation else 0.0
        patterns = set().union(*(c.patterns for c in channels.values()))
        named = [name for name, term in COLOURS.items() if term in patterns]
        if len(named) == 1:
            values["noise_type"] = named[0]
    else:
        if SSTIM_V.shapeSine in signals:
            values["carrier_freq"] = signals[SSTIM_V.shapeSine][0]
        if SSTIM_V.shapeTriangle in signals:
            values["motion_rate"] = signals[SSTIM_V.shapeTriangle][0]
    return values


def _entailed(values: dict[URIRef, tuple[float, ...]]
              ) -> dict[URIRef, tuple[float, ...]]:
    """`values`, with what each implies of the terms above it.

    A beat frequency is a modulation frequency, and a carrier, a tone or a
    modulation frequency is a frequency the channel presents, so a reader
    that states the more general term states no contradiction.
    """
    def of(predicate: URIRef) -> tuple[float, ...]:
        """What `values` states for `predicate` itself."""
        return values.get(predicate, ())
    modulation = of(SSTIM_EX.hasModulationFrequencyHz) + \
        of(SSTIM_EX.hasBeatFrequencyHz)
    return {
        SSTIM_EX.hasCarrierFrequencyHz: of(SSTIM_EX.hasCarrierFrequencyHz),
        SSTIM_EX.hasToneFrequencyHz: of(SSTIM_EX.hasToneFrequencyHz),
        SSTIM_EX.hasBeatFrequencyHz: of(SSTIM_EX.hasBeatFrequencyHz),
        SSTIM_EX.hasModulationFrequencyHz: modulation,
        SSTIM_EX.hasFrequencyHz: (of(SSTIM_EX.hasCarrierFrequencyHz)
                                  + of(SSTIM_EX.hasToneFrequencyHz)
                                  + modulation),
        SSTIM_EX.hasDutyCycle: of(SSTIM_EX.hasDutyCycle),
    }


def _differences(found: _Stimulus, expected: _Stimulus) -> list[str]:
    """Where `found` says what the package would not render for it.

    What `found` leaves out of the terms SSTIM makes optional -- a
    carrier, a channel's frequencies -- is no difference. Its channels'
    durations were compared already, with each other and with any given.
    """
    problems = []
    if found.regime != expected.regime:
        problems.append(f"its sstim:stimulusRegime is {found.regime!r}, not "
                        f"{expected.regime!r}")
    if set(found.signals) != set(expected.signals):
        problems.append(
            "its signals are of shape "
            + ", ".join(sorted(_term(s) for s in found.signals))
            + ", not " + ", ".join(sorted(_term(s) for s in expected.signals)))
    else:
        for shape, (low, high) in sorted(found.signals.items()):
            span = expected.signals[shape]
            if not (_close(low, span[0]) and _close(high, span[1])):
                problems.append(
                    f"its {_term(shape)} signal spans {low:g} to {high:g} "
                    f"Hz, not {span[0]:g} to {span[1]:g}")
    if set(found.channels) != set(expected.channels):
        problems.append(
            "its channels are at "
            + ", ".join(sorted(_term(p) for p in found.channels))
            + ", not "
            + ", ".join(sorted(_term(p) for p in expected.channels)))
        return problems
    for placement, channel in sorted(found.channels.items()):
        want = expected.channels[placement]
        where = f"the {_term(placement)} channel"
        for name in ("modality", "medium"):
            if getattr(channel, name) != getattr(want, name):
                problems.append(f"{where}'s {name} is "
                                f"{_term(getattr(channel, name))}, not "
                                f"{_term(getattr(want, name))}")
        if set(channel.renderings) != set(want.renderings):
            problems.append(
                f"{where} renders signals of shape "
                + ", ".join(sorted(_term(s) for s in channel.renderings))
                + ", not "
                + ", ".join(sorted(_term(s) for s in want.renderings)))
            continue
        for shape, rendering in sorted(channel.renderings.items()):
            target = want.renderings[shape]
            for name in ("parameter", "mechanism", "presence"):
                if getattr(rendering, name) != getattr(target, name):
                    problems.append(
                        f"{where} renders its {_term(shape)} signal with "
                        f"{name} {_term(getattr(rendering, name))}, not "
                        f"{_term(getattr(target, name))}")
            if rendering.carrier is not None and (
                    target.carrier is None
                    or not _close(rendering.carrier, target.carrier)):
                problems.append(
                    f"{where} renders its {_term(shape)} signal on a "
                    f"{rendering.carrier:g} Hz carrier, "
                    + ("but the package renders it on none"
                       if target.carrier is None else
                       f"not {target.carrier:g}"))
        allowed = _entailed(want.values)
        for predicate, numbers in channel.values.items():
            for number in numbers:
                if not any(_close(number, value)
                           for value in allowed[predicate]):
                    problems.append(f"{where} states {_term(predicate)} "
                                    f"{number:g}, which the package does "
                                    f"not render")
        for pattern in sorted(channel.patterns - want.patterns):
            problems.append(f"{where} states the pattern {_term(pattern)}, "
                            f"which the package does not render")
    return problems


@dataclass(frozen=True)
class _Reading:
    generator: str
    stated: dict[str, Any]
    duration: float | None
    found: _Stimulus


def _read(graph: Graph) -> _Reading:
    """What a description says in SSTIM's terms, of which stimulus."""
    found = _project(graph, _root(graph))
    generator = _generator(found)
    durations = [c.duration for c in found.channels.values()
                 if c.duration is not None]
    if any(not _close(d, durations[0]) for d in durations):
        raise ValueError("the stimulus's channels last different times")
    return _Reading(generator, _stated(generator, found),
                    durations[0] if durations else None, found)


def _agree(given: Any, stated: Any) -> bool:
    numbers = all(isinstance(v, Real) and not isinstance(v, bool)
                  for v in (given, stated))
    return _close(float(given), float(stated)) if numbers else given == stated


def _moot(reading: _Reading) -> dict[str, Any]:
    """Controls that cannot change what `reading` renders, at defaults.

    An unmodulated noise is the same whatever its modulation's depth, so
    the depth is neither asked for nor taken as stated.
    """
    if (reading.generator != "modulated_noise"
            or reading.stated["modulation_freq"]):
        return {}
    default = inspect.signature(stimuli.modulated_noise).parameters
    return {"modulation_depth": default["modulation_depth"].default}


def _needs(reading: _Reading, controls: Mapping[str, Any],
           sample_rate: Any, duration: Any) -> list[str]:
    """What a caller must give that `reading` does not state."""
    needs = [] if sample_rate is not None else ["sample_rate"]
    if reading.duration is None and duration is None:
        needs.append("duration")
    moot = _moot(reading)
    return needs + [name for name in portable(reading.generator)
                    if name not in reading.stated and name not in controls
                    and name not in moot]


def _asking(reading: _Reading, needs: list[str]) -> str:
    """Why `reading` cannot be rendered yet, and what to give."""
    given = " and ".join(n for n in needs if n in ("sample_rate",
                                                     "duration"))
    rest = ", ".join(n for n in needs if n not in ("sample_rate",
                                                     "duration"))
    ask = (given + (", and " if given else "") + "controls for " + rest
           if rest else given)
    return (f"the description is {reading.generator} in SSTIM's terms "
            f"alone, which leave {', '.join(needs)} unsaid: give {ask}")


def _check(reading: _Reading, controls: Mapping[str, Any] | None,
           sample_rate: Any, duration: Any
           ) -> tuple[SstimStimulus | None, list[str]]:
    """The stimulus `reading` and what is given make, or what is missing.

    The caller gives only what SSTIM has no term for, or the description
    leaves out; what both give must agree. Then the package's own
    description of the stimulus must state everything the given one does.
    That is checked even while something is missing, with the generator's
    defaults standing in for it, so a description no choice could render
    is refused for what it says rather than for what it leaves out.
    """
    controls = {} if controls is None else controls
    if not isinstance(controls, Mapping):
        raise TypeError("controls must be a mapping of argument names")
    for name, value in controls.items():
        if name in reading.stated and not _agree(value,
                                                 reading.stated[name]):
            raise ValueError(
                f"controls give {name} as {value!r}, but the description "
                f"states {reading.stated[name]!r}")
    if (duration is not None and reading.duration is not None
            and not _agree(duration, reading.duration)):
        raise ValueError(f"duration is given as {duration!r}, but the "
                         f"description states {reading.duration!r}")
    needs = _needs(reading, controls, sample_rate, duration)
    defaults = inspect.signature(getattr(stimuli, reading.generator)
                                 ).parameters
    stand_ins = {name: defaults[name].default for name in needs
                 if name not in ("sample_rate", "duration")}
    seconds = (reading.duration if reading.duration is not None else
               duration if duration is not None else
               defaults["duration"].default)
    try:
        stimulus = checked(reading.generator,
                           {**_moot(reading), **stand_ins, **reading.stated,
                            **controls},
                           seconds,
                           192000 if sample_rate is None else sample_rate)
    except ValueError:
        if not needs:
            raise
        # A stand-in may be what does not fit: ask for what is missing.
        return None, needs
    problems = _differences(reading.found, _project(
        stimulus_graph(stimulus, _EXPECTED),
        URIRef(_EXPECTED + "specification")))
    if problems:
        raise ValueError(
            f"the description says what the package would not render for "
            f"it as {reading.generator}: " + "; ".join(problems[:3]))
    return (None if needs else stimulus), needs


def read_other(graph: Graph, *, controls: Mapping[str, Any] | None,
               sample_rate: Any, duration: Any) -> SstimStimulus:
    """A stimulus in SSTIM's terms alone, with what they cannot say."""
    reading = _read(graph)
    stimulus, needs = _check(reading, controls, sample_rate, duration)
    if stimulus is None:
        raise ValueError(_asking(reading, needs))
    return stimulus


def needs_of(graph: Graph) -> tuple[str, tuple[str, ...]]:
    """Which stimulus a description is, and what rendering it needs."""
    reading = _read(graph)
    return reading.generator, tuple(_check(reading, None, None, None)[1])


@dataclass(frozen=True)
class SstimBeat:
    """A beat as SSTIM's terms give it, to render by :data:`BEAT_PROFILE`.

    SSTIM states a beat's tones, its rate and its length; it does not fix
    the tones' phase or gain, so this renders them by a named convention,
    not by the package's generator: both start at phase zero, and a
    monaural beat is their mean. Another engine following the convention
    renders the same samples, to within the rounding of its sines.

    Attributes
    ----------
    generator : str
        ``binaural_beats`` or ``monaural_beats``.
    carrier_freq : float
        The frequency midway between the two tones, in Hz.
    beat_freq : float
        The difference between them, in Hz.
    duration : float
        Seconds.
    sample_rate : int
        Samples per second.
    """

    generator: str
    carrier_freq: float
    beat_freq: float
    duration: float
    sample_rate: int

    def render(self) -> NDArray[np.float64]:
        """The samples, by :data:`BEAT_PROFILE`.

        Returns
        -------
        ndarray
            ``(2, nsamples)`` for a binaural beat, one tone in each ear;
            mono for a monaural one.
        """
        if self.generator not in BEATS:
            raise ValueError("a beat is binaural_beats or monaural_beats")
        beat = checked(self.generator, {"carrier_freq": self.carrier_freq,
                                        "beat_freq": self.beat_freq},
                       self.duration, self.sample_rate)
        carrier = beat.parameters["carrier_freq"]
        rate = beat.parameters["beat_freq"]
        t = np.arange(int(beat.duration * beat.sample_rate)) / \
            beat.sample_rate
        low = np.sin(2 * np.pi * (carrier - rate / 2) * t)
        high = np.sin(2 * np.pi * (carrier + rate / 2) * t)
        if self.generator == "binaural_beats":
            return np.vstack((low, high))
        return (low + high) / 2


def read_beat(graph: Graph, *, sample_rate: Any, duration: Any) -> SstimBeat:
    """A binaural or monaural beat, by its SSTIM terms alone."""
    reading = _read(graph)
    if reading.generator not in BEATS:
        raise ValueError(
            f"{BEAT_PROFILE} renders a binaural or a monaural beat; this "
            f"description is {reading.generator}")
    stimulus, needs = _check(reading, None, sample_rate, duration)
    if "carrier_freq" in needs:
        raise ValueError(f"{BEAT_PROFILE} renders a beat whose carriers the "
                         f"description states; this one states none")
    if stimulus is None:
        raise ValueError(_asking(reading, needs))
    return SstimBeat(stimulus.generator, stimulus.parameters["carrier_freq"],
                     stimulus.parameters["beat_freq"], stimulus.duration,
                     stimulus.sample_rate)
