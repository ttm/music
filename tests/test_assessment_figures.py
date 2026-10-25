"""The self-measuring documents, and the anchors that keep them checkable.

`tools/assessment_figures.py` writes ASSESSMENT.md's figures as one table,
between two markers, and checks the few figures prose still quotes: the
version each living document stamps itself with, and the reconciliation
register's counts. A quoted figure is checked by a regex over prose that
people rewrite, so a check is only as good as its anchor: reflow a
sentence and the figure it carried stops being read, silently, which is
the failure this whole arrangement exists to stop. The table is written
whole, so what can go wrong with it is its markers, or rows the script
no longer writes.

The script refuses to run when an anchor is gone, but only a full run
measures the table, and that is the release gate's -- the last place to
learn that a heading was reworded three weeks ago. These tests check
every anchor, and the table's shape, on every push, and measure nothing:
the numbers are the script's job, and the placeholder figures below are
never compared against anything.

The version stamps are here for a second reason. All three documents
said `music` 1.4.0 while the package was 1.7.0, with every figure around
them correct and current, because a stamp was the one thing nothing read.
Checking it here runs it on every supported Python and inside the source
distribution, rather than in the single CI job that runs the script.
"""

import re

import pytest

from tools.assessment_figures import (ASSESSMENT, BEGIN, DISCREPANCIES, END,
                                      RECONCILIATION, current_table,
                                      declared_version, quoted, register,
                                      table, table_date, table_rows)


class AnyFigure(dict):
    """Zero for any figure the script asks about.

    These tests check where the script looks and what shape it writes,
    not what it finds, so the values never matter.
    """

    def __missing__(self, key):
        return 0


def test_every_quoted_figure_is_still_where_the_script_looks():
    orphaned = []
    for label, pattern, _want, path in quoted(AnyFigure(), "0.0.0"):
        if not path.exists():        # pragma: no cover - an sdist only
            continue
        if not re.search(pattern, path.read_text()):
            orphaned.append(f"{label} ({path.name})")

    assert not orphaned, (
        "these figures are no longer where tools/assessment_figures.py "
        f"looks for them: {', '.join(orphaned)}. The file was rewritten "
        "around them, so the check silently stopped running. Fix the "
        "pattern rather than deleting the check.")


def _assessment():
    if not ASSESSMENT.exists():      # pragma: no cover - an sdist only
        pytest.skip("ASSESSMENT.md is missing")
    return ASSESSMENT.read_text()


def test_the_table_is_between_its_markers_once():
    block = current_table(_assessment())
    assert block.startswith(BEGIN) and block.endswith(END)


def test_the_table_has_the_rows_the_script_writes():
    """A row added by hand, or one the script stopped writing, is a row
    no run would ever correct."""
    written = table_rows(table(AnyFigure(), "0.0.0", "2019-01-01"))
    assert set(table_rows(current_table(_assessment()))) == set(written)
    assert len(written) >= 10


def test_the_table_says_when_it_was_measured():
    assert re.fullmatch(r"\d{4}-\d\d-\d\d",
                        table_date(current_table(_assessment())))


def test_a_table_without_its_markers_is_refused_rather_than_written():
    with pytest.raises(SystemExit, match="exactly once"):
        current_table("a file whose table was written by hand")
    with pytest.raises(SystemExit, match="exactly once"):
        current_table(f"{BEGIN}\n{END}\n{BEGIN}\n{END}")


def test_a_table_without_a_date_is_refused():
    with pytest.raises(SystemExit, match="no measurement date"):
        table_date(f"{BEGIN}\n{END}")


def test_writing_the_table_replaces_only_the_table():
    old = table(AnyFigure(modules=1), "1.0.0", "2019-01-01")
    new = table(AnyFigure(modules=2), "1.0.1", "2019-01-02")
    text = f"before\n\n{old}\n\nafter\n"
    replaced = text.replace(current_table(text), new)
    assert replaced == f"before\n\n{new}\n\nafter\n"
    assert table_rows(new)["Package"].startswith("2 modules")


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


def test_every_register_row_is_counted_in_one_state():
    """A state worded differently would drop out of all three counts and
    leave the quoted figures agreeing with a register that was short."""
    counts = register()
    assert counts["mass_total"] == (counts["mass_exact"]
                                    + counts["mass_divergent"]
                                    + counts["mass_broken"])
    assert counts["mass_total"] > 30
