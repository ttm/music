"""Run the tutorial.

Documentation that is never executed drifts from the code it documents.
Every ``code-block:: python`` in ``docs/tutorial.rst`` is extracted and run
here, in order and in one shared namespace, exactly as a reader working
down the page would run them.  Doctest-style blocks are checked against
the output they claim.  The blocks that sing need espeak-ng and
Parselmouth, which the test jobs do not install; where they are missing,
those blocks are left out, and CI's singing job runs them.
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


def _can_sing():
    from music.singing import psola

    return not psola.missing_requirements()


SINGS = _can_sing()


def test_the_page_has_the_blocks_this_test_thinks_it_has():
    """Guard the extraction: a change to the directive's formatting must
    not quietly reduce this file to testing nothing."""
    assert len(BLOCKS) >= 14
    assert any("PlainChanges" in block for block in BLOCKS)
    assert any("write_wav_stereo" in block for block in BLOCKS)
    assert sum("music.sing(" in block for block in BLOCKS) == 2


def test_every_tutorial_block_runs(tmp_path, monkeypatch):
    """The blocks build on each other, so they share one namespace and run
    in page order -- which also checks that a name a later block uses was
    really defined by an earlier one."""
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}

    for index, block in enumerate(BLOCKS):
        if block.lstrip().startswith(">>>"):
            continue
        if "music.sing(" in block and not SINGS:
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
    if SINGS:
        assert (tmp_path / "twinkle.wav").exists()
        assert (tmp_path / "frere.wav").exists()


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


def _cycles(sound):
    """Where each rising zero crossing of `sound` falls, interpolated, and
    the frequency of each cycle between two of them, at 44,100 Hz."""
    import numpy as np

    negative = np.signbit(sound)
    crossings = np.where(~negative[1:] & negative[:-1])[0]
    exact = crossings + sound[crossings] / (sound[crossings]
                                            - sound[crossings + 1])
    return exact, 44100 / np.diff(exact)


def test_the_figures_the_tutorial_quotes_are_what_its_code_measures():
    """Its prose quotes three measurements of the blocks beside it, and
    nothing checked them: the glissando's midpoint said 441 Hz, where the
    code measures 440.1."""
    import numpy as np

    import music

    text = " ".join(TUTORIAL.read_text().split())
    _, vibrato = _cycles(music.note_with_vibrato(
        freq=440, duration=2, vibrato_freq=5, max_pitch_dev=2))
    assert f"``{vibrato.min():.1f} .. {vibrato.max():.1f} Hz``" in text
    distinct = len(np.unique(np.round(vibrato, 1)))
    assert f"takes {distinct} distinct values, to a tenth of a hertz" in text

    sweep = music.note_with_glissando(220, 880, duration=2, method="exp")
    edges, glissando = _cycles(sweep)
    halfway = glissando[np.searchsorted(edges, len(sweep) / 2) - 1]
    assert f"it is at {halfway:.1f} Hz" in text

