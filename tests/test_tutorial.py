"""Run the tutorial.

Documentation that is never executed drifts from the code it documents.
Every ``code-block:: python`` in ``docs/tutorial.rst`` is extracted and run
here, in order and in one shared namespace, exactly as a reader working
down the page would run them.  Doctest-style blocks are checked against
the output they claim.
"""

import doctest
import pathlib
import re
import textwrap

import pytest

TUTORIAL = (pathlib.Path(__file__).parent.parent
            / "docs" / "tutorial.rst")

#: ``.. code-block:: python`` followed by an indented body.
_BLOCK = re.compile(
    r"^\.\. code-block:: python\n\n((?:(?:[ \t]+[^\n]*)?\n)+)",
    re.MULTILINE,
)


def _blocks():
    """Every python block on the page, in the order it appears."""
    return [textwrap.dedent(match.group(1)).strip("\n")
            for match in _BLOCK.finditer(TUTORIAL.read_text())]


BLOCKS = _blocks()


def test_the_page_has_the_blocks_this_test_thinks_it_has():
    """Guard the extraction: a change to the directive's formatting must
    not quietly reduce this file to testing nothing."""
    assert len(BLOCKS) >= 14
    assert any("PlainChanges" in block for block in BLOCKS)
    assert any("write_wav_stereo" in block for block in BLOCKS)


def test_every_tutorial_block_runs(tmp_path, monkeypatch):
    """The blocks build on each other, so they share one namespace and run
    in page order -- which also checks that a name a later block uses was
    really defined by an earlier one."""
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}

    for index, block in enumerate(BLOCKS):
        if block.lstrip().startswith(">>>"):
            continue
        try:
            exec(compile(block, f"tutorial.rst[block {index}]", "exec"),
                 namespace)
        except Exception as error:  # pragma: no cover - the failure report
            pytest.fail(f"block {index} of the tutorial failed with "
                        f"{type(error).__name__}: {error}\n\n{block}")

    assert "music" in namespace, "the page never showed the import"
    assert (tmp_path / "piece.wav").exists(), (
        "the closing example did not write its file"
    )


def test_the_doctest_blocks_produce_the_output_they_claim():
    """A block written as a session promises a specific result."""
    import music

    sessions = [block for block in BLOCKS
                if block.lstrip().startswith(">>>")]
    assert sessions, "no doctest-style block found on the page"

    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    parser = doctest.DocTestParser()
    for index, session in enumerate(sessions):
        test = parser.get_doctest(session + "\n", {"music": music},
                                  f"tutorial-session-{index}", None, 0)
        runner.run(test)
    assert runner.failures == 0, f"{runner.failures} doctest failure(s)"
