"""All SSTIM importers share bounded, local-only RDF graph loading."""

from pathlib import Path

import pytest

pytest.importorskip("rdflib")
from rdflib import Graph, Literal, URIRef  # noqa: E402
from rdflib.namespace import RDF  # noqa: E402

from music.stimulation.sstim_io import (  # noqa: E402
    _MAX_RDF_TRIPLES, _MAX_TURTLE_BYTES, _graph, SSTIM,
    to_sstim_graph,
)


def fixture():
    return to_sstim_graph(
        "monaural_beats", duration=.02,
        parameters={"carrier_freq": 200, "beat_freq": 10},
        base="https://example.org/import-guard/")


def test_graph_and_serialized_turtle_and_file_inputs(tmp_path):
    g = fixture()
    turtle = g.serialize(format="turtle")
    path = tmp_path / "input.ttl"
    path.write_text(turtle)
    assert set(_graph(g)) == set(g)
    assert set(_graph(turtle)) == set(g)
    assert set(_graph(path)) == set(g)
    assert set(_graph(str(path))) == set(g)


@pytest.mark.parametrize("uri", [
    "https://example.org/a.ttl",
    "http://example.org/x",
    "ftp://example.org/a.ttl",
    "file:///etc/passwd",
])
def test_explicit_uri_sources_are_never_dereferenced(uri):
    with pytest.raises(ValueError, match="URI RDF input"):
        _graph(uri)


def test_reject_missing_or_unsupported_sources(tmp_path):
    with pytest.raises(ValueError, match="existing local"):
        _graph(tmp_path / "not-here.ttl")
    with pytest.raises(ValueError, match="existing local"):
        _graph("missing-file.ttl")
    with pytest.raises(TypeError, match="expected RDF Graph"):
        _graph(3)
    with pytest.raises(ValueError, match="existing local"):
        _graph(tmp_path)


def test_reject_large_turtle_source_bytes(tmp_path):
    large = "@prefix ex: <urn:ex:> .\n" + "#" * _MAX_TURTLE_BYTES
    with pytest.raises(ValueError, match="byte budget"):
        _graph(large)
    p = tmp_path / "huge.ttl"
    p.write_text(large)
    with pytest.raises(ValueError, match="byte budget"):
        _graph(p)


def test_reject_excessive_graph_triples_and_compact_expansion():
    g = Graph()
    subject = URIRef("urn:many")
    pred = URIRef("urn:count")
    for i in range(_MAX_RDF_TRIPLES + 1):
        g.add((subject, pred, Literal(i)))
    with pytest.raises(ValueError, match="triple budget"):
        _graph(g)
    compact = ("@prefix ex: <urn:ex:> .\n"
               "ex:many ex:count " +
               ", ".join(f'"{i}"' for i in range(_MAX_RDF_TRIPLES + 1)) +
               " .")
    assert len(compact.encode("utf-8")) < _MAX_TURTLE_BYTES
    with pytest.raises(ValueError, match="triple budget"):
        _graph(compact)


def test_local_nonexistent_file_then_existing_file(tmp_path):
    path = Path(tmp_path / "audio.ttl")
    with pytest.raises(ValueError):
        _graph(path)
    path.write_text(
        '<urn:a> <urn:p> <urn:b> .\n', encoding="utf-8")
    g = _graph(path)
    assert len(g) == 1
    assert not list(g.subjects(RDF.type, SSTIM.StimulusSpecification))


def test_invalid_utf8_turtle_file_fails_before_rdf_parser(tmp_path):
    path = tmp_path / "invalid.ttl"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="cannot read local Turtle"):
        _graph(path)
