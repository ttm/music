"""The seven stimuli as SSTIM descriptions, and read back."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import math
from numbers import Real
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from .. import stimuli
from ._vocabulary import (
    MUSIC, SSTIM, SSTIM_EX, SSTIM_V, bind, check_base, decimal, difference,
    one, significant,
)

#: The stimuli a description can be written for, by name.
STIMULI = ("binaural_beats", "monaural_beats", "isochronic_tones",
           "amplitude_modulation", "frequency_modulation", "modulated_noise",
           "spatial_motion")

#: Each tone stimulus in SSTIM's terms: its mechanism, its signal's shape,
#: the parameter rendered, and the argument that is its rate.
TONES = {
    "binaural_beats": ("mechanismBinauralBeat", "shapeSine",
                       "paramAmplitude", "beat_freq"),
    "monaural_beats": ("mechanismMonauralBeat", "shapeSine",
                       "paramAmplitude", "beat_freq"),
    "isochronic_tones": ("mechanismAmplitudeModulation", "shapeSquare",
                         "paramAmplitude", "pulse_rate"),
    "amplitude_modulation": ("mechanismAmplitudeModulation", "shapeSine",
                             "paramAmplitude", "modulation_freq"),
    "frequency_modulation": ("mechanismFrequencyModulation", "shapeSine",
                             "paramFrequency", "modulation_freq"),
}
BEATS = ("binaural_beats", "monaural_beats")

#: SSTIM's names for the package's noise colours. Black, which SSTIM takes
#: for silence, and a slope given in decibels an octave have none.
COLOURS = {
    "white": SSTIM_EX.audioNoiseWhite,
    "pink": SSTIM_EX.audioNoisePink,
    "brown": SSTIM_EX.audioNoiseBrownRed,
    "blue": SSTIM_EX.audioNoiseBlue,
    "violet": SSTIM_EX.audioNoiseVioletPurple,
}
_NOISES = set(COLOURS) | {"black"}

#: Arguments a description never carries: the length and rate are its own,
#: and a table or a sound is not a number SSTIM, or JSON, can hold.
_RESERVED = {"duration", "number_of_samples", "sample_rate", "waveform_table",
             "modulation_waveform_table", "sonic_vector"}


@dataclass(frozen=True)
class SstimStimulus:
    """One stimulus as a description gives it.

    Attributes
    ----------
    generator : str
        Which of :data:`STIMULI` it is.
    parameters : dict
        Every one of its arguments but its length, rate and tables.
    duration : float
        Seconds.
    sample_rate : int
        Samples per second.
    """

    generator: str
    parameters: dict[str, Any]
    duration: float
    sample_rate: int

    def render(self, *, oversampling_factor: int | None = None
               ) -> NDArray[np.float64]:
        """The samples, as the stimulus renders them.

        Parameters
        ----------
        oversampling_factor : int, optional
            Render through :func:`music.render_oversampled` at this many
            times the rate, from 2 to 16, which needs SciPy; None renders
            directly.

        Returns
        -------
        ndarray
            Mono, or ``(2, nsamples)`` for the binaural beats and spatial
            motion.
        """
        pinned = checked(self.generator, self.parameters, self.duration,
                         self.sample_rate)
        generator = getattr(stimuli, pinned.generator)
        if oversampling_factor is None:
            return generator(duration=pinned.duration,
                             sample_rate=pinned.sample_rate,
                             **pinned.parameters)
        from ...core.synths.oversampling import render_oversampled
        return render_oversampled(
            generator, duration=pinned.duration,
            sample_rate=pinned.sample_rate, factor=oversampling_factor,
            **pinned.parameters)


def portable(generator: str) -> tuple[str, ...]:
    """The arguments of `generator` a description carries."""
    fields = inspect.signature(getattr(stimuli, generator)).parameters
    return tuple(name for name in fields if name not in _RESERVED)


def _number(name: str, value: Any) -> float:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value)):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    return float(value)


def checked(generator: Any, parameters: Mapping[str, Any], duration: Any,
             sample_rate: Any) -> SstimStimulus:
    """`generator` with `parameters`, checked, and every default written out.

    A description that leaves an argument to the package's default would
    render differently once that default changed, so every one is pinned.
    """
    if generator not in STIMULI:
        raise ValueError(
            f"not a stimulus a description can be written for: "
            f"{generator!r}; they are {', '.join(STIMULI)}")
    if (isinstance(sample_rate, bool) or not isinstance(sample_rate, int)
            or not 8000 <= sample_rate <= 192000):
        raise ValueError(
            f"sample_rate must be an integer from 8000 to 192000; got "
            f"{sample_rate!r}")
    if (isinstance(duration, (bool, np.bool_))
            or not isinstance(duration, Real)
            or not math.isfinite(duration) or duration <= 0):
        raise ValueError(
            f"duration must be a positive number of seconds; got "
            f"{duration!r}")
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping of argument names")
    fields = inspect.signature(getattr(stimuli, generator)).parameters
    unknown = sorted(name for name in parameters
                     if name not in fields or name in _RESERVED)
    if unknown:
        raise ValueError(
            f"{generator} takes no argument a description carries named "
            f"{', '.join(unknown)}")
    values: dict[str, Any] = {}
    for name in portable(generator):
        value = parameters.get(name, fields[name].default)
        if name == "seed":
            if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, np.integer))
                    or not 0 <= value < 2 ** 64):
                raise ValueError(
                    f"seed must be None or an integer from 0 to 2**64 - 1; "
                    f"got {value!r}")
            values[name] = None if value is None else int(value)
        elif name == "noise_type" and isinstance(value, str):
            if value not in _NOISES:
                raise ValueError(
                    f"noise_type must be one of {', '.join(sorted(_NOISES))} "
                    f"or a slope in decibels an octave; got {value!r}")
            values[name] = value
        else:
            values[name] = _number(name, value)
    _check(generator, values, sample_rate)
    return SstimStimulus(generator, values, float(duration), sample_rate)


def _check(generator: str, values: dict[str, Any], sample_rate: int) -> None:
    """Refuse arguments the stimulus cannot render as described."""
    nyquist = sample_rate / 2
    if generator in TONES:
        carrier, rate = values["carrier_freq"], values[TONES[generator][3]]
        if not 0 < carrier < nyquist or rate <= 0:
            raise ValueError(
                "the carrier must lie between 0 and the Nyquist frequency, "
                "and the rate must be positive")
        if generator in BEATS and not (
                0 < carrier - rate / 2 and carrier + rate / 2 < nyquist):
            raise ValueError(
                "both tones of the beat must lie between 0 and the Nyquist "
                "frequency")
        if generator == "isochronic_tones" and not (
                0 < values["duty_cycle"] <= 1
                and values["ramp_duration"] >= 0):
            raise ValueError(
                "duty_cycle must be in (0, 1] and ramp_duration not negative")
        if generator == "amplitude_modulation" and not (
                0 <= values["modulation_depth"] <= 1):
            raise ValueError("modulation_depth must be in [0, 1]")
    elif generator == "modulated_noise":
        if not (0 <= values["modulation_depth"] <= 1
                and values["modulation_freq"] >= 0
                and 0 <= values["min_freq"] < values["max_freq"] <= nyquist):
            raise ValueError(
                "modulation_depth must be in [0, 1], modulation_freq not "
                "negative, and the band rise from 0 to the Nyquist frequency")
    elif not (0 < values["carrier_freq"] < nyquist
              and values["motion_rate"] >= 0 and values["dist"] > 0
              and values["zeta"] > 0 and -100 < values["air_temp"] < 100):
        raise ValueError(
            "the carrier must lie below the Nyquist frequency, the motion "
            "rate not be negative, dist and zeta be positive, and air_temp "
            "lie between -100 and 100")


def _signal(graph: Graph, node: URIRef, label: str, shape: str,
            low: float, high: float) -> None:
    graph.add((node, RDF.type, SSTIM.StimulationSignal))
    graph.add((node, RDFS.label, Literal(label)))
    graph.add((node, SSTIM.hasSignalShape, SSTIM_V[shape]))
    graph.add((node, SSTIM.hzMin, decimal(low)))
    graph.add((node, SSTIM.hzMax, decimal(high)))


def _channel(graph: Graph, node: URIRef, placement: str,
             duration: float) -> None:
    graph.add((node, RDF.type, SSTIM_EX.StimulusChannel))
    graph.add((node, RDFS.label, Literal(placement)))
    graph.add((node, SSTIM.channelDurationSeconds, decimal(duration)))
    graph.add((node, SSTIM_EX.perceivedModality, SSTIM_EX.modalityAuditory))
    graph.add((node, SSTIM_EX.deliveryMedium,
               SSTIM_EX.mediumAirConductedSound))
    graph.add((node, SSTIM_EX.hasBodyPlacement, SSTIM_EX[placement]))


def _rendering(graph: Graph, channel: URIRef, node: URIRef, label: str,
               signal: URIRef, parameter: str, mechanism: str,
               presence: str) -> None:
    graph.add((channel, SSTIM.hasSignalRendering, node))
    graph.add((node, RDF.type, SSTIM.SignalRendering))
    graph.add((node, RDFS.label, Literal(label)))
    graph.add((node, SSTIM.rendersSignal, signal))
    graph.add((node, SSTIM.rendersOntoParameter, SSTIM_V[parameter]))
    graph.add((node, SSTIM.hasRenderingMechanism, SSTIM_V[mechanism]))
    graph.add((node, SSTIM.hasRenderingPresence, SSTIM_V[presence]))


def stimulus_graph(stimulus: SstimStimulus, base: str) -> Graph:
    """`stimulus` described in SSTIM's terms, with the package's beside.

    SSTIM's terms say what the stimulus is, in the words any consumer
    reads: its signals, its channels and what each presents, at which
    frequencies, and how each signal is rendered. The package's say what
    SSTIM has no words for -- a modulation's depth, a noise's seed, a
    motion's geometry -- and which of its stimuli renders it.
    """
    check_base(base)
    generator, values = stimulus.generator, stimulus.parameters
    graph = bind(Graph())
    root = URIRef(base + "specification")
    graph.add((root, RDF.type, SSTIM.StimulusSpecification))
    graph.add((root, MUSIC.generator, Literal(generator)))
    graph.add((root, MUSIC.parametersJson, Literal(
        json.dumps(values, sort_keys=True, separators=(",", ":")))))
    graph.add((root, MUSIC.sampleRateHz,
               Literal(stimulus.sample_rate, datatype=XSD.integer)))
    if generator in TONES:
        _tone(graph, root, stimulus, base)
    elif generator == "modulated_noise":
        _noise(graph, root, stimulus, base)
    else:
        _motion(graph, root, stimulus, base)
    return graph


def _tone(graph: Graph, root: URIRef, stimulus: SstimStimulus,
          base: str) -> None:
    generator, values = stimulus.generator, stimulus.parameters
    mechanism, shape, parameter, rate_name = TONES[generator]
    rate, carrier = values[rate_name], values["carrier_freq"]
    graph.add((root, RDFS.label, Literal(f"{generator}: {rate:g} Hz")))
    graph.add((root, SSTIM.stimulusRegime, Literal("determinate")))
    signal = URIRef(base + "signal")
    graph.add((root, SSTIM.hasSignal, signal))
    _signal(graph, signal, f"{rate:g} Hz {shape[5:].lower()} signal",
            shape, rate, rate)
    binaural = generator == "binaural_beats"
    placements = (("placementEarLeft", carrier - rate / 2),
                  ("placementEarRight", carrier + rate / 2)) if binaural \
        else (("placementEars", carrier),)
    for index, (placement, frequency) in enumerate(placements, 1):
        channel = URIRef(base + f"channel-{index}")
        rendering = URIRef(base + f"rendering-{index}")
        graph.add((root, SSTIM.hasStimulusChannel, channel))
        _channel(graph, channel, placement, stimulus.duration)
        _rendering(graph, channel, rendering, f"{generator}: {placement}",
                   signal, parameter, mechanism,
                   "presencePerceptual" if binaural else "presencePhysical")
        graph.add((rendering, SSTIM.renderingCarrierHz, decimal(frequency)))
        graph.add((channel, SSTIM_EX.hasCarrierFrequencyHz,
                   decimal(frequency)))
        graph.add((channel, SSTIM_EX.hasBeatFrequencyHz
                   if generator in BEATS else
                   SSTIM_EX.hasModulationFrequencyHz, decimal(rate)))
        if generator == "isochronic_tones":
            graph.add((channel, SSTIM_EX.hasDutyCycle,
                       decimal(values["duty_cycle"])))


def _noise(graph: Graph, root: URIRef, stimulus: SstimStimulus,
           base: str) -> None:
    values = stimulus.parameters
    colour = values["noise_type"]
    named = colour if isinstance(colour, str) else \
        f"{colour:g} dB an octave"
    graph.add((root, RDFS.label, Literal(f"modulated_noise: {named}")))
    graph.add((root, SSTIM.stimulusRegime, Literal("stochastic")))
    band = URIRef(base + "signal-noise")
    graph.add((root, SSTIM.hasSignal, band))
    _signal(graph, band, f"{named} noise band", "shapeNoise",
            values["min_freq"], values["max_freq"])
    renderings = [(band, "paramAmplitude", "mechanismDirectPresentation")]
    rate = values["modulation_freq"]
    if rate > 0:
        modulation = URIRef(base + "signal-modulation")
        graph.add((root, SSTIM.hasSignal, modulation))
        _signal(graph, modulation, f"{rate:g} Hz noise amplitude modulation",
                "shapeSine", rate, rate)
        renderings.append((modulation, "paramAmplitude",
                           "mechanismAmplitudeModulation"))
    channel = URIRef(base + "channel-1")
    graph.add((root, SSTIM.hasStimulusChannel, channel))
    _channel(graph, channel, "placementEars", stimulus.duration)
    for index, (signal, parameter, mechanism) in enumerate(renderings, 1):
        _rendering(graph, channel, URIRef(base + f"rendering-1-{index}"),
                   f"placementEars: {mechanism}", signal, parameter,
                   mechanism, "presencePhysical")
    if colour in COLOURS:
        graph.add((channel, SSTIM_EX.hasStimulusPattern, COLOURS[colour]))
    if rate > 0:
        graph.add((channel, SSTIM_EX.hasModulationFrequencyHz, decimal(rate)))


def _motion(graph: Graph, root: URIRef, stimulus: SstimStimulus,
            base: str) -> None:
    values = stimulus.parameters
    carrier, rate = values["carrier_freq"], values["motion_rate"]
    graph.add((root, RDFS.label, Literal(
        f"spatial_motion: {carrier:g} Hz, {rate:g} Hz")))
    graph.add((root, SSTIM.stimulusRegime, Literal("determinate")))
    tone, motion = URIRef(base + "signal-tone"), URIRef(base + "signal-motion")
    graph.add((root, SSTIM.hasSignal, tone))
    graph.add((root, SSTIM.hasSignal, motion))
    _signal(graph, tone, f"{carrier:g} Hz acoustic source tone",
            "shapeSine", carrier, carrier)
    _signal(graph, motion, f"{rate:g} Hz triangular azimuth motion",
            "shapeTriangle", rate, rate)
    for index, placement in enumerate(
            ("placementEarLeft", "placementEarRight"), 1):
        channel = URIRef(base + f"channel-{index}")
        graph.add((root, SSTIM.hasStimulusChannel, channel))
        _channel(graph, channel, placement, stimulus.duration)
        for number, (signal, parameter) in enumerate(
                ((tone, "paramAmplitude"),
                 (motion, "paramSpatialPosition")), 1):
            _rendering(graph, channel,
                       URIRef(base + f"rendering-{index}-{number}"),
                       f"{placement}: {parameter}", signal, parameter,
                       "mechanismDirectPresentation", "presencePhysical")
        graph.add((channel, SSTIM_EX.hasToneFrequencyHz, decimal(carrier)))
        if rate > 0:
            graph.add((channel, SSTIM_EX.hasModulationFrequencyHz,
                       decimal(rate)))


def under(statements: set, base: str) -> set:
    """Those of `statements` about the resources named under `base`."""
    return {t for t in statements if str(t[0]).startswith(base)}


def read_stimulus(graph: Graph) -> SstimStimulus:
    """The stimulus a description the package wrote gives, and checked.

    Its statements in SSTIM's terms and the package's, about the resources
    it names, must be exactly those the package would write for it: a
    graph that says something else about the stimulus is refused rather
    than half believed. Statements in other vocabularies, such as labels
    or provenance, are left alone.
    """
    roots = list(graph.subjects(RDF.type, SSTIM.StimulusSpecification))
    if len(roots) != 1:
        raise ValueError("expected exactly one SSTIM StimulusSpecification")
    root = roots[0]
    if not str(root).endswith("specification"):
        raise ValueError(
            "this is not a description the package wrote: its stimulus is "
            "not named ...specification")
    base = str(root)[:-len("specification")]
    generator = str(one(graph, root, MUSIC.generator))
    text = str(one(graph, root, MUSIC.parametersJson))
    if len(text) > 2048:
        raise ValueError("the parameters are longer than any stimulus has")
    try:
        parameters = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("the parameters are not JSON") from exc
    if not isinstance(parameters, dict):
        raise ValueError("the parameters must be a JSON object")
    rate = one(graph, root, MUSIC.sampleRateHz)
    if not isinstance(rate, Literal) or rate.datatype != XSD.integer:
        raise ValueError("the sample rate must be an xsd:integer")
    # The package names its first channel channel-1; the comparison below
    # checks every other channel against it.
    durations: list[Any] = list(graph.objects(
        URIRef(base + "channel-1"), SSTIM.channelDurationSeconds))
    if len(durations) != 1:
        raise ValueError("the stimulus's first channel, channel-1, states "
                         "no single duration")
    try:
        duration = float(durations[0])
    except (TypeError, ValueError) as exc:
        raise ValueError("the channel's duration is not a number") from exc
    stimulus = checked(generator, parameters, duration, int(rate))
    found = under(significant(graph), base)
    expected = significant(stimulus_graph(stimulus, base))
    if found != expected:
        raise ValueError(
            "the description's statements are not those the package "
            "writes for its stimulus: " + difference(found, expected))
    return stimulus
