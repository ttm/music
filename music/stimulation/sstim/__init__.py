"""Stimuli and sessions as SSTIM descriptions, and back.

SSTIM, the Sensory Stimulation Vocabulary, describes a stimulus in terms
any tool can read: its signals, the channels that present them, and how
each channel renders each signal, at which frequencies and for how long.
This module writes the package's stimuli and sessions in those terms,
reads them back, and reads what other tools write.

========================  ===================================================
:func:`to_sstim_graph`    A stimulus or a session, as a description.
:func:`from_sstim_graph`  A description, as what renders it.
:func:`render_sstim`      A description's samples.
:func:`inspect_sstim`     What the package can do with a description.
:func:`validate_sstim`    Whether a description conforms to SSTIM.
========================  ===================================================

A description is read in one of three ways:

* **The package's own.** It states every control in the package's terms
  beside SSTIM's, and renders exactly as written. Its SSTIM statements
  must be exactly those the package writes for those controls, or it is
  refused rather than half believed.
* **In SSTIM's terms alone,** as another tool writes it. Its mechanisms
  and signals say which of the seven stimuli it is; its carriers, rates,
  duty cycle, noise colour and length are read from it. What SSTIM has no
  term for -- the sample rate, a modulation's depth, a noise's seed, a
  motion's geometry -- is given as ``sample_rate`` and ``controls``, and
  :func:`inspect_sstim` lists it. The package's own description of the
  result must then state everything the given one does.
* **By a named convention.** A binaural or monaural beat in SSTIM's terms
  can be rendered by :data:`BEAT_PROFILE` instead of the package's
  generator: zero-phase, equal-gain sines, which another engine can
  reproduce to within the rounding of its sines.

The package's exact samples are its own: SSTIM states what a stimulus
is, not how an engine computes it, so another engine renders the same
description into different samples. A session is a program in the
package's terms, since SSTIM 0.19 has none for phases in order, each
phase an SSTIM stimulus; given a :class:`SessionPlan`, it is wrapped in
SSTIM's own SessionSpecification. Nothing written is a record of
delivery: no SessionInstance, event or exposure is made.

Reading never follows a URL or a link in what it reads, and never runs
anything a description names: a generator is one of :data:`STIMULI`, by
name. Descriptions need RDFLib; :func:`validate_sstim` needs SSTIM's
validator. Both come with ``pip install 'music[sstim]'``.
"""
from __future__ import annotations

try:
    import rdflib  # noqa: F401
except ImportError as exc:  # pragma: no cover - tested in a subprocess
    raise ImportError("music.stimulation.sstim needs RDFLib: pip install "
                      "'music[sstim]'") from exc

import inspect
import math
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray
from rdflib import Graph
from rdflib.namespace import RDF

from .. import stimuli
from ..session import StimulationSession
from ._capabilities import SESSION_ALONE, SstimCapability, inspect_sstim
from ._others import BEAT_PROFILE, SstimBeat, read_beat, read_other
from ._sessions import (
    PlannedSession, SessionPlan, plan_graph, program_graph, read_plan,
    read_program,
)
from ._stimuli import (
    STIMULI, SstimStimulus, checked, read_stimulus, stimulus_graph,
)
from ._vocabulary import (
    MAX_SAMPLES, MUSIC, SSTIM, SSTIM_EX, SSTIM_V, SSTIM_VERSION, read_graph,
    uses_package_terms,
)

__all__ = [
    "BEAT_PROFILE",
    "MAX_SAMPLES",
    "MUSIC",
    "PlannedSession",
    "SSTIM",
    "SSTIM_EX",
    "SSTIM_V",
    "SSTIM_VERSION",
    "STIMULI",
    "SessionPlan",
    "SstimBeat",
    "SstimCapability",
    "SstimStimulus",
    "from_sstim_graph",
    "inspect_sstim",
    "render_sstim",
    "to_sstim_graph",
    "validate_sstim",
]

Source = Graph | str | Path
Described = SstimStimulus | SstimBeat | StimulationSession | PlannedSession


def to_sstim_graph(source: Callable[..., Any] | str | SstimStimulus
                   | StimulationSession | PlannedSession, /, *,
                   base: str | None = None, plan: SessionPlan | None = None,
                   duration: float | None = None,
                   sample_rate: int | None = None,
                   **parameters: Any) -> Graph:
    """Describe a stimulus or a session in SSTIM's terms.

    A stimulus is described as the call ``source(duration=duration,
    sample_rate=sample_rate, **parameters)`` renders it, with every
    argument left to its default written out, so that a later default
    cannot change what the description renders. Its SSTIM terms state its
    signals, its channels and their frequencies; the package's state what
    SSTIM has no term for, and which stimulus renders it.

    A session is a program of its phases in the package's terms, each an
    SSTIM stimulus. With a `plan`, it is also SSTIM's SessionSpecification:
    a plan of a whole number of seconds, from 60 to 7,200, that names a
    preset and declares ``reproEquivalentPresentation``.

    Parameters
    ----------
    source : callable, str, SstimStimulus, StimulationSession or \
PlannedSession
        One of the seven stimuli, by its function or its name (see
        :data:`STIMULI`); a stimulus :func:`from_sstim_graph` read; a
        session whose phases are each one of the seven by its function;
        or a planned session.
    base : str, optional
        Your own http(s) IRI, ending in / or #, the description's
        resources are named under. Defaults to one under
        ``https://example.org/music/``, which is fine to try things but
        not to publish.
    plan : SessionPlan, optional
        For a session: the preset, creation time and master volume that
        make it a SessionSpecification.
    duration : float, optional
        For a stimulus: seconds. Defaults to the generator's.
    sample_rate : int, optional
        For a stimulus: samples per second, from 8,000 to 192,000.
        Defaults to the generator's. SSTIM has no term for it, so it is
        written in the package's.
    **parameters
        For a stimulus: its arguments, but for its length, rate, source
        sound and tables, which are not numbers a description holds. A
        noise's colour is named or a slope; its seed is None or an integer
        below 2**64.

    Returns
    -------
    rdflib.Graph
        The description. Check it with :func:`validate_sstim`.

    Raises
    ------
    ValueError
        If the stimulus, an argument or `base` cannot be described, or a
        planned session is not a whole number of seconds from 60 to 7,200.
    TypeError
        If a stimulus is given a plan, or a session given arguments.

    Examples
    --------
    >>> from music.stimulation import binaural_beats
    >>> graph = to_sstim_graph(binaural_beats, carrier_freq=200,
    ...                        beat_freq=10, duration=60)
    >>> from_sstim_graph(graph).parameters
    {'carrier_freq': 200.0, 'beat_freq': 10.0}
    """
    if isinstance(source, PlannedSession):
        if plan is not None:
            raise TypeError("a planned session has its plan")
        plan = source.plan
        source = source.session
    if isinstance(source, StimulationSession):
        if duration is not None or sample_rate is not None or parameters:
            raise TypeError("a session has its own phases, rate and length; "
                            "it takes no duration, sample_rate or arguments")
        base = "https://example.org/music/session/" if base is None else base
        if plan is None:
            return program_graph(source, base)
        return plan_graph(source, plan, base)
    if plan is not None:
        raise TypeError("a plan is for a session, not a stimulus")
    base = "https://example.org/music/stimulus/" if base is None else base
    if isinstance(source, SstimStimulus):
        if duration is not None or sample_rate is not None or parameters:
            raise TypeError("a stimulus read from a description has its own "
                            "arguments, rate and length")
        return stimulus_graph(checked(source.generator, source.parameters,
                                      source.duration, source.sample_rate),
                              base)
    name = source if isinstance(source, str) else getattr(
        source, "__name__", None)
    if name not in STIMULI or not (isinstance(source, str)
                                   or source is getattr(stimuli, name)):
        raise ValueError(
            f"expected one of the seven stimuli, by its function or name, a "
            f"stimulus, or a session; got {source!r}; they are "
            f"{', '.join(STIMULI)}")
    defaults = inspect.signature(getattr(stimuli, name)).parameters
    return stimulus_graph(checked(
        name, parameters,
        defaults["duration"].default if duration is None else duration,
        defaults["sample_rate"].default if sample_rate is None
        else sample_rate), base)


def _fixed(what: str, rate: int, seconds: float, sample_rate: Any,
           duration: Any) -> None:
    """Refuse a rate or length given for a description that fixes them."""
    if sample_rate is not None and (isinstance(sample_rate, bool)
                                    or sample_rate != rate):
        raise ValueError(f"{what} fixes its sample rate at {rate} Hz; "
                         f"sample_rate is given as {sample_rate!r}")
    if duration is not None and (
            isinstance(duration, bool) or not isinstance(duration, Real)
            or not math.isclose(duration, seconds, rel_tol=1e-9,
                                abs_tol=1e-9)):
        raise ValueError(f"{what} lasts {seconds:g} s; duration is given "
                         f"as {duration!r}")


def from_sstim_graph(value: Source, /, *,
                     controls: Mapping[str, Any] | None = None,
                     profile: str | None = None,
                     sample_rate: int | None = None,
                     duration: float | None = None) -> Described:
    """Read a description: what renders it, checked.

    A description the package wrote gives back what was described: a
    :class:`SstimStimulus`, a :class:`music.StimulationSession`
    or a :class:`PlannedSession`. One in SSTIM's terms alone gives the
    :class:`SstimStimulus` it describes, given what SSTIM cannot say; or,
    for a beat with `profile`, an :class:`SstimBeat`. See the module's
    introduction for how each is checked.

    Parameters
    ----------
    value : rdflib.Graph, str or Path
        The description: a graph, a local Turtle file, or Turtle text.
    controls : mapping, optional
        For a description in SSTIM's terms alone: the stimulus's
        arguments it does not state, by name. Those it states may be given
        too, and must agree.
    profile : str, optional
        :data:`BEAT_PROFILE`, to read a beat as an :class:`SstimBeat`.
    sample_rate : int, optional
        The rate to render at, which a description in SSTIM's terms alone
        needs: SSTIM fixes none. One the package wrote fixes its own,
        which this must match if given.
    duration : float, optional
        Seconds, for a description whose channels state no length; a
        length stated must match if given.

    Returns
    -------
    SstimStimulus, SstimBeat, StimulationSession or PlannedSession
        What renders the description, with its ``render()``.

    Raises
    ------
    ValueError
        If the description is not one the package can render as stated,
        or lacks or contradicts what is given: the message says which.
    """
    graph = read_graph(value)
    if profile is not None and profile != BEAT_PROFILE:
        raise ValueError(f"the only profile is {BEAT_PROFILE!r}; got "
                         f"{profile!r}")
    own = uses_package_terms(graph)
    planned = (None, RDF.type, SSTIM.SessionSpecification) in graph
    if planned or (None, RDF.type, MUSIC.StimulationProgram) in graph:
        if not own:
            raise ValueError(SESSION_ALONE)
        if controls or profile is not None:
            raise ValueError("a session is read as the package wrote it, "
                             "with no controls or profile")
        read = read_plan(graph) if planned else read_program(graph)
        session = read.session if isinstance(read, PlannedSession) else read
        _fixed("the session", session.sample_rate, session.duration,
               sample_rate, duration)
        return read
    if own:
        if controls:
            raise ValueError("the package wrote this description, which "
                             "states every control: it takes no controls")
        stimulus = read_stimulus(graph)
        _fixed("the description", stimulus.sample_rate, stimulus.duration,
               sample_rate, duration)
        if profile is None:
            return stimulus
        return read_beat(graph, sample_rate=stimulus.sample_rate,
                         duration=None)
    if profile is not None:
        if controls:
            raise ValueError(f"{BEAT_PROFILE} renders a beat from its SSTIM "
                             f"terms alone, with no controls")
        return read_beat(graph, sample_rate=sample_rate, duration=duration)
    return read_other(graph, controls=controls, sample_rate=sample_rate,
                      duration=duration)


def render_sstim(value: Source, /, *,
                 controls: Mapping[str, Any] | None = None,
                 profile: str | None = None, sample_rate: int | None = None,
                 duration: float | None = None,
                 oversampling_factor: int | None = None,
                 max_samples: int = MAX_SAMPLES) -> NDArray[np.float64]:
    """Render a description, as :func:`from_sstim_graph` reads it.

    Parameters
    ----------
    value : rdflib.Graph, str or Path
        The description: a graph, a local Turtle file, or Turtle text.
    controls, profile, sample_rate, duration
        As for :func:`from_sstim_graph`.
    oversampling_factor : int, optional
        For one of the package's stimuli: render through
        :func:`music.render_oversampled` at this many times the rate, from
        2 to 16, which needs SciPy. None renders directly.
    max_samples : int
        The most samples a channel may have, so that a description from
        elsewhere cannot exhaust memory by accident. Raise it to render a
        longer one.

    Returns
    -------
    ndarray
        Mono, or ``(2, nsamples)`` for the binaural beats, spatial motion
        and a session containing either.

    Raises
    ------
    ValueError
        As :func:`from_sstim_graph` does; or if the render would be longer
        than `max_samples`, or `oversampling_factor` is given for a
        session or a beat rendered by its profile.
    """
    if (isinstance(max_samples, bool) or not isinstance(max_samples, int)
            or max_samples < 1):
        raise ValueError(f"max_samples must be a positive integer; got "
                         f"{max_samples!r}")
    described = from_sstim_graph(value, controls=controls, profile=profile,
                                 sample_rate=sample_rate, duration=duration)
    timeline = (described.session if isinstance(described, PlannedSession)
                else described)
    count = int(timeline.duration * timeline.sample_rate)
    if count > max_samples:
        raise ValueError(
            f"the description renders {count} samples a channel, more than "
            f"max_samples, {max_samples}: raise it to render a longer one")
    if oversampling_factor is None:
        return described.render()
    if not isinstance(described, SstimStimulus):
        raise ValueError("oversampling_factor is for one of the package's "
                         "stimuli, not a session or a beat by its profile")
    return described.render(oversampling_factor=oversampling_factor)


def validate_sstim(value: Source, /, *, version: str = SSTIM_VERSION) -> Any:
    """Check a description against SSTIM's Full profile, by SSTIM's validator.

    The validator fetches the release's modules the first time, checks
    their checksums and caches them, so it needs the network once. Its
    checks are SSTIM's, and stricter than reading a description: one the
    package reads may still not conform, and one that conforms may still
    be one the package cannot render.

    Parameters
    ----------
    value : rdflib.Graph, str or Path
        The description: a graph, a local Turtle file, or Turtle text.
    version : str
        The SSTIM release to validate against.

    Returns
    -------
    sstim.Report
        Whether it conforms, with any terms it uses that SSTIM does not
        define or that are minted in SSTIM's namespace.

    Raises
    ------
    ImportError
        If SSTIM's validator is not installed.
    """
    try:
        import sstim
    except ImportError as exc:
        raise ImportError(
            "validating needs SSTIM's validator: pip install "
            "'music[sstim]'") from exc
    return sstim.validate(read_graph(value), profile="full", version=version)
