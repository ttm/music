"""The vocabularies a description is written in, and reading one safely."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

SSTIM = Namespace("https://w3id.org/sstim#")
SSTIM_V = Namespace("https://w3id.org/sstim/vocab#")
SSTIM_EX = Namespace("https://w3id.org/sstim/exposure#")
#: The package's own terms, for what SSTIM has no term for. Never minted
#: under SSTIM's namespaces.
MUSIC = Namespace("https://github.com/ttm/music/terms#")

#: The SSTIM release the descriptions are written in and validated against.
SSTIM_VERSION = "0.19.0"

#: How many samples a channel :func:`render_sstim` renders at most, by
#: default: about 113 s at 44.1 kHz. A description is only numbers, so
#: writing and reading one are not capped; rendering one from elsewhere
#: could exhaust memory by accident.
MAX_SAMPLES = 5_000_000
#: A description read from Turtle is capped at these. They stop a large
#: file being read by accident; they are not a sandbox for a hostile one,
#: which the parser has read before the triples can be counted.
MAX_TURTLE_BYTES = 2 * 1024 * 1024
MAX_TRIPLES = 15_000

_OURS = (str(SSTIM), str(SSTIM_V), str(SSTIM_EX), str(MUSIC))


def decimal(value: float) -> Literal:
    """`value` as an xsd:decimal, written as the float it is read back as.

    Every reader takes these back as floats and rebuilds the graph to
    compare, so an int 15, written "15", was rebuilt "15.0", and the graph
    refused as altered.
    """
    return Literal(str(Decimal(str(float(value)))), datatype=XSD.decimal)


def one(graph: Graph, subject: Any, predicate: URIRef) -> Any:
    """The single object of `subject`'s `predicate`, or a ValueError."""
    values = list(graph.objects(subject, predicate))
    if len(values) != 1:
        raise ValueError(f"expected exactly one {predicate}")
    return values[0]


def check_base(base: Any) -> str:
    """`base`, if it is an http(s) IRI of the caller's, ending / or #."""
    if (not isinstance(base, str)
            or not base.startswith(("http://", "https://"))
            or not base.endswith(("/", "#"))
            or base.startswith("https://w3id.org/sstim")):
        raise ValueError(
            "base must be your own http(s) IRI, ending in / or #, outside "
            "SSTIM's namespace")
    return base


def bind(graph: Graph) -> Graph:
    """`graph`, with the prefixes a description is written with."""
    for prefix, namespace in (("sstim", SSTIM), ("sstim-v", SSTIM_V),
                              ("sstim-ex", SSTIM_EX), ("music-impl", MUSIC)):
        graph.bind(prefix, namespace)
    return graph


def uses_package_terms(graph: Graph) -> bool:
    """Whether `graph` says anything in the package's own terms."""
    space = str(MUSIC)
    return any(str(p).startswith(space) for p in graph.predicates()) or any(
        str(o).startswith(space) for o in graph.objects(None, RDF.type))


def ours(term: Any) -> bool:
    """Whether `term` is in SSTIM's vocabularies or the package's."""
    return isinstance(term, URIRef) and str(term).startswith(_OURS)


def significant(graph: Graph, also: Iterable[URIRef] = ()) -> set:
    """The statements in `graph` made in SSTIM's terms or the package's.

    Labels, comments and anything in another vocabulary say nothing a
    render depends on, and a caller may annotate a description with them.
    An ``rdf:type`` counts when its class is ours. `also` names other
    predicates that count, such as the creation time of a plan.
    """
    extra = set(also)
    return {(s, p, o) for s, p, o in graph
            if ours(p) or p in extra or (p == RDF.type and ours(o))}


def difference(found: set, expected: set) -> str:
    """A few of the statements that differ, to say why a graph is refused."""
    def shown(triples):
        """Up to three of `triples`, as N-Triples, in a fixed order."""
        return "; ".join(sorted(" ".join(term.n3() for term in triple)
                                for triple in triples)[:3])
    missing, unexpected = expected - found, found - expected
    parts = []
    if missing:
        parts.append(f"missing {shown(missing)}")
    if unexpected:
        parts.append(f"unexpected {shown(unexpected)}")
    return ", ".join(parts)


def _turtle(value: str | Path) -> str:
    """The Turtle `value` is, or names the local file of."""
    if isinstance(value, str):
        trimmed = value.lstrip()
        if trimmed.startswith(("http://", "https://", "file://", "ftp://")):
            raise ValueError("only local Turtle; URI RDF input denied")
        if ("\n" in value or trimmed.startswith(
                ("@prefix", "@base", "PREFIX", "BASE", "<"))):
            return value
        value = Path(value)
    try:
        if not value.is_file():
            raise ValueError("expected an existing local Turtle file")
        if value.stat().st_size > MAX_TURTLE_BYTES:
            raise ValueError("Turtle input exceeds byte budget")
        return value.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("cannot read local Turtle file") from exc


def read_graph(value: Graph | str | Path) -> Graph:
    """A graph, from a graph, a local Turtle file, or Turtle text.

    Never from a URL, and never following a link in what it reads: within
    :data:`MAX_TURTLE_BYTES` and :data:`MAX_TRIPLES`.
    """
    if isinstance(value, Graph):
        graph = value
    elif isinstance(value, (str, Path)):
        text = _turtle(value)
        if len(text.encode("utf-8")) > MAX_TURTLE_BYTES:
            raise ValueError("Turtle input exceeds byte budget")
        graph = Graph()
        graph.parse(data=text, format="turtle")
    else:
        raise TypeError("expected RDF Graph, local Turtle or local path")
    if len(graph) > MAX_TRIPLES:
        raise ValueError("RDF graph exceeds triple budget")
    return graph
