"""Descriptions are read only from local Turtle, and within budgets."""

from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("rdflib")
from rdflib import Graph, Literal, URIRef  # noqa: E402
from rdflib.namespace import RDF  # noqa: E402

from music.stimulation.sstim import (  # noqa: E402
    SSTIM, from_sstim_graph, inspect_sstim, render_sstim, to_sstim_graph,
)
from music.stimulation.sstim._vocabulary import (  # noqa: E402
    MAX_TRIPLES, MAX_TURTLE_BYTES, read_graph,
)


def fixture():
    return to_sstim_graph("monaural_beats", duration=.02,
                          base="https://example.org/import-guard/")


def test_a_graph_turtle_text_a_path_and_a_path_string(tmp_path):
    graph = fixture()
    turtle = graph.serialize(format="turtle")
    path = tmp_path / "input.ttl"
    path.write_text(turtle)
    assert read_graph(graph) is graph
    assert set(read_graph(turtle)) == set(graph)
    assert set(read_graph(path)) == set(graph)
    assert set(read_graph(str(path))) == set(graph)


@pytest.mark.parametrize("uri", [
    "https://example.org/a.ttl",
    "http://example.org/x",
    "ftp://example.org/a.ttl",
    "file:///etc/passwd",
    "   https://example.org/indented.ttl",
])
@pytest.mark.parametrize("reader", [
    read_graph, from_sstim_graph, render_sstim, inspect_sstim])
def test_a_url_is_never_fetched(uri, reader):
    with pytest.raises(ValueError, match="URI RDF input denied"):
        reader(uri)


def test_missing_and_unsupported_sources_are_refused(tmp_path):
    with pytest.raises(ValueError, match="existing local"):
        read_graph(tmp_path / "not-here.ttl")
    with pytest.raises(ValueError, match="existing local"):
        read_graph("missing-file.ttl")
    with pytest.raises(TypeError, match="expected RDF Graph"):
        read_graph(3)
    with pytest.raises(ValueError, match="existing local"):
        read_graph(tmp_path)


def test_too_many_bytes_are_refused(tmp_path):
    large = "@prefix ex: <urn:ex:> .\n" + "#" * MAX_TURTLE_BYTES
    with pytest.raises(ValueError, match="byte budget"):
        read_graph(large)
    path = tmp_path / "huge.ttl"
    path.write_text(large)
    with pytest.raises(ValueError, match="byte budget"):
        read_graph(path)


def test_too_many_triples_are_refused_however_compact():
    graph = Graph()
    for i in range(MAX_TRIPLES + 1):
        graph.add((URIRef("urn:many"), URIRef("urn:count"), Literal(i)))
    with pytest.raises(ValueError, match="triple budget"):
        read_graph(graph)
    compact = ("@prefix ex: <urn:ex:> .\nex:many ex:count "
               + ", ".join(f'"{i}"' for i in range(MAX_TRIPLES + 1)) + " .")
    assert len(compact.encode("utf-8")) < MAX_TURTLE_BYTES
    with pytest.raises(ValueError, match="triple budget"):
        read_graph(compact)


def test_a_file_written_after_a_failed_read_is_read(tmp_path):
    path = Path(tmp_path / "audio.ttl")
    with pytest.raises(ValueError):
        read_graph(path)
    path.write_text('<urn:a> <urn:p> <urn:b> .\n', encoding="utf-8")
    graph = read_graph(path)
    assert len(graph) == 1
    assert not list(graph.subjects(RDF.type, SSTIM.StimulusSpecification))


def test_a_file_that_is_not_utf8_is_refused_before_parsing(tmp_path):
    path = tmp_path / "invalid.ttl"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="cannot read local Turtle"):
        read_graph(path)


def test_without_rdflib_the_module_says_what_to_install():
    code = ("import sys; sys.modules['rdflib'] = None; "
            "import music.stimulation.sstim")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "needs RDFLib: pip install 'music[sstim]'" in result.stderr


def test_importing_music_imports_no_rdf():
    code = ("import sys, music, music.stimulation; "
            "print('rdflib' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
