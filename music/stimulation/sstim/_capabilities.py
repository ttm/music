"""What the package can do with a description, before anything renders."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rdflib import Graph
from rdflib.namespace import RDF

from ._others import BEAT_PROFILE, needs_of
from ._sessions import read_plan, read_program
from ._stimuli import BEATS, read_stimulus
from ._vocabulary import MUSIC, SSTIM, read_graph, uses_package_terms

#: Why a session in SSTIM's terms alone is not rendered.
SESSION_ALONE = (
    "a SessionSpecification in SSTIM's terms alone names a preset, which "
    "only its own engine renders: SSTIM has no terms for a session's phases")


@dataclass(frozen=True)
class SstimCapability:
    """What the package can do with a description.

    Attributes
    ----------
    kind : str
        ``stimulus``, ``session`` (a program the package wrote) or
        ``planned-session`` (an SSTIM SessionSpecification around one), or
        ``unsupported``.
    written_by_package : bool
        Whether it says anything in the package's own terms. Such a
        description carries every control and renders as written, and is
        refused if those terms and SSTIM's disagree; any other is read by
        SSTIM's terms alone.
    generator : str or None
        Which of :data:`STIMULI` renders a stimulus.
    needs : tuple of str
        What :func:`render_sstim` must still be given: ``sample_rate``,
        ``duration``, and by name the controls SSTIM has no term for.
    profiles : tuple of str
        The named conventions it can also be rendered by without them but
        for ``sample_rate``: :data:`BEAT_PROFILE`, for a beat.
    reason : str
        Why it is unsupported; empty otherwise.
    """

    kind: str
    written_by_package: bool
    generator: str | None
    needs: tuple[str, ...]
    profiles: tuple[str, ...]
    reason: str

    @property
    def can_render(self) -> bool:
        """Whether :func:`render_sstim` renders it given nothing more."""
        return self.kind != "unsupported" and not self.needs


def inspect_sstim(value: Graph | str | Path, /) -> SstimCapability:
    """Say what the package can do with a description, rendering nothing.

    A description the package wrote is checked whole, as
    :func:`from_sstim_graph` checks it. One in SSTIM's terms alone is read
    as one of the seven stimuli, and checked with the generator's
    defaults standing in for the controls it leaves unsaid, which
    ``needs`` then lists. Needs RDFLib, not SSTIM's validator.

    Parameters
    ----------
    value : rdflib.Graph, str or Path
        The description: a graph, a local Turtle file, or Turtle text.

    Returns
    -------
    SstimCapability
        What it is, and what rendering it still needs.

    Raises
    ------
    ValueError
        If `value` is not a local Turtle description within the budgets.
    """
    graph = read_graph(value)
    own = uses_package_terms(graph)
    needs: tuple[str, ...] = ()
    try:
        if (None, RDF.type, SSTIM.SessionSpecification) in graph:
            if not own:
                raise ValueError(SESSION_ALONE)
            read_plan(graph)
            return SstimCapability("planned-session", True, None, (), (), "")
        if (None, RDF.type, MUSIC.StimulationProgram) in graph:
            read_program(graph)
            return SstimCapability("session", True, None, (), (), "")
        if own:
            generator = read_stimulus(graph).generator
        else:
            generator, needs = needs_of(graph)
    except (ValueError, TypeError) as exc:
        return SstimCapability("unsupported", own, None, (), (), str(exc))
    return SstimCapability("stimulus", own, generator, needs,
                           (BEAT_PROFILE,) if generator in BEATS else (), "")
