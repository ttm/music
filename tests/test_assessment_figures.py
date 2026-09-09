"""The self-measuring documents, and the anchors that keep them checkable.

`tools/assessment_figures.py` measures ASSESSMENT.md's figures and reads
the version each living document stamps itself with.  Every one of those
checks is a regex over prose that people rewrite, so a check is only as
good as its anchor: reflow a table row and the figure it carried stops
being read, silently, which is the failure this whole arrangement exists
to stop.

The script does refuse to run when an anchor is gone.  But CI runs it
with `--fast`, and `--fast` never builds the seven expectations that need
pytest, coverage and ruff to answer -- so those anchors were checked only
by the release gate, which is the last place to learn that a heading was
reworded three weeks ago.  These tests check every anchor on every push,
and measure nothing: the numbers are the script's job, and the placeholder
figures below are never compared against anything.

The version stamps are here for a second reason.  All three documents
said `music` 1.4.0 while the package was 1.7.0, with every figure around
them correct and current, because a stamp was the one thing nothing read.
Checking it here runs it on every supported Python and inside the source
distribution, rather than in the single CI job that runs the script.
"""

import re

import pytest

from tools.assessment_figures import (ASSESSMENT, DISCREPANCIES,
                                      RECONCILIATION, declared_version,
                                      expectations, stamp_the_date)


class AnyFigure(dict):
    """Zero for any figure the script asks about.

    These tests check where the script looks, not what it finds, so the
    values never matter.  Seeding `tests` is what does matter:
    `expectations` builds the slow half only when that key is present,
    and the slow half is exactly the part `--fast` leaves unchecked.
    """

    def __missing__(self, key):
        return 0


def test_every_figure_the_script_checks_is_still_where_it_looks():
    """Including the seven that only the release gate ever builds."""
    orphaned = []
    for label, pattern, _want, path in expectations(AnyFigure(tests=0)):
        if not path.exists():        # pragma: no cover - an sdist only
            continue
        if not re.search(pattern, path.read_text()):
            orphaned.append(f"{label} ({path.name})")

    assert not orphaned, (
        "these figures are no longer where tools/assessment_figures.py "
        f"looks for them: {', '.join(orphaned)}. The file was rewritten "
        "around them, so the check silently stopped running. Fix the "
        "pattern rather than deleting the check.")


@pytest.mark.parametrize("document", [ASSESSMENT, RECONCILIATION,
                                      DISCREPANCIES],
                         ids=lambda document: document.name)
def test_the_living_documents_stamp_the_version_they_describe(document):
    """A document naming a release the package has left behind is stale."""
    if not document.exists():        # pragma: no cover - an sdist only
        pytest.skip(f"{document.name} is missing")

    stamped = re.findall(r"(?<=`music` )\d+\.\d+\.\d+", document.read_text())
    assert stamped, (
        f"{document.name} no longer stamps a version. It is a living "
        "record of a package that keeps changing; without a version it "
        "does not say which one it describes.")
    assert set(stamped) == {declared_version()}, (
        f"{document.name} says {sorted(set(stamped))} and pyproject.toml "
        f"says {declared_version()}. `python tools/assessment_figures.py "
        "--write` corrects it.")


def test_the_date_is_rewritten_rather_than_quietly_skipped():
    """The one write in the script that no check has already anchored."""
    text = "*Last measured **2019-01-01**, `music` 1.0.0: 3 modules.*"
    assert "2019-01-01" not in stamp_the_date(text)
    assert re.search(r"Last measured \*\*\d{4}-\d\d-\d\d\*\*",
                     stamp_the_date(text))

    with pytest.raises(SystemExit, match="not where this script looks"):
        stamp_the_date("a file that no longer carries a stamp")
