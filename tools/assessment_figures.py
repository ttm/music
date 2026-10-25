#!/usr/bin/env python3
"""Keep ASSESSMENT.md's numbers true, by measuring them.

ASSESSMENT.md opens by saying that a snapshot nobody updates
misrepresents the code it describes. It then went stale four times in
two days, and was corrected by hand each time -- once while preparing a
release, with the wrong test count already committed. Issue #70 asks for
the practice of keeping it current; a practice that depends on someone
remembering is the thing that keeps failing.

So the figures the package can measure about itself are one table in
ASSESSMENT.md, between two markers, and this script writes it. They were
once quoted through the prose and checked by regular expressions over
the sentences holding them, on every push: that kept them true, and made
every change that added a test or a line also a change to that file, and
a reflowed sentence a check that silently stopped running. The table is
written whole, the prose refers to it rather than restating it, and the
release gate checks it, so every released copy is measured from its
release.

What prose still quotes changes only when something meaningful does: the
version each of the three living documents stamps itself with, in
ASSESSMENT.md, RECONCILIATION.md and DISCREPANCIES.md -- all three said
1.4.0 at 1.7.0, with every number around them correct, because a stamp
was the one figure nothing read -- and the reconciliation register's
counts, wherever they are quoted. Those are checked on every push.

Usage
-----
::

    python tools/assessment_figures.py                 # report any drift
    python tools/assessment_figures.py --write         # correct it
    python tools/assessment_figures.py --fast          # quoted figures
    python tools/assessment_figures.py --write --fast  # correct those

``--fast`` measures nothing: it checks the quoted figures, which is what CI
runs on every push. The table needs the whole suite to run under coverage,
and ruff, which take a few minutes between them.

What is deliberately not checked: the measurement date, which is written
rather than checked, since a date that is not today is not drift but a
table nothing has had to correct since; and the history in the opening
paragraphs of ASSESSMENT.md, which is about how the file went wrong once
and must not be rewritten to match the present.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent.parent
ASSESSMENT = ROOT / "ASSESSMENT.md"
README = ROOT / "README.md"
RECONCILIATION = ROOT / "RECONCILIATION.md"
DISCREPANCIES = ROOT / "DISCREPANCIES.md"
ZENODO = ROOT / ".zenodo.json"

#: The lines the table sits between. The first says who writes it.
BEGIN = ("<!-- BEGIN FIGURES: written by `python tools/assessment_figures.py "
         "--write`; change the script, not the table. -->")
END = "<!-- END FIGURES -->"


def declared_version():
    """The version in pyproject.toml, which is the one that counts.

    `tools/release.py` says the same of it, and refuses to publish while
    CITATION.cff or the changelog disagrees.
    """
    match = re.search(r"^version = '([^']+)'",
                      (ROOT / "pyproject.toml").read_text(), re.MULTILINE)
    if not match:
        raise SystemExit("no version found in pyproject.toml")
    return match.group(1)


def package_files():
    return sorted(p for p in (ROOT / "music").glob("**/*.py")
                  if "ecantorix" not in str(p))


def loc(paths):
    return sum(len(p.read_text().splitlines()) for p in paths)


def register():
    """How many routines the reconciliation register puts in each state.

    The register is RECONCILIATION.md's table, which
    `tests/test_mass_reconciliation.py` re-establishes against the package
    on every push, so counting its rows needs no MASS checkout. The counts
    are quoted in four other places, and the Zenodo abstract said 26
    sample-exact for three releases after it had become 23.
    """
    states = re.findall(r"^\| `[^|]+` \| `[^|]+` \| ([a-z ]+?) \|",
                        RECONCILIATION.read_text(), re.MULTILINE)
    return {"mass_total": len(states),
            "mass_exact": states.count("exact"),
            "mass_divergent": states.count("divergent"),
            "mass_broken": states.count("reference does not run")}


def scan():
    """The figures an AST walk can settle, without running anything."""
    sys.path.insert(0, str(ROOT))
    import music

    exported = set(music.__all__)
    annotated = total = 0
    exported_annotated = exported_total = 0
    documented = public = 0

    for path in package_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = (node.args.posonlyargs + node.args.args
                        + node.args.kwonlyargs)
                # `s` as well as `self`: the legacy synths spell the
                # instance parameter that way.
                ok = (all(a.annotation for a in args
                          if a.arg not in ("self", "s"))
                      and node.returns is not None)
                total += 1
                annotated += ok
                if node.name in exported:
                    exported_total += 1
                    exported_annotated += ok
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                if not node.name.startswith("_"):
                    public += 1
                    documented += bool(ast.get_docstring(node))

    return {
        "modules": len(package_files()),
        "package_loc": loc(package_files()),
        "tests_loc": loc(sorted((ROOT / "tests").glob("**/*.py"))),
        "examples": len(sorted((ROOT / "examples").glob("*.py"))),
        "api_names": len(music.__all__),
        "annotated": annotated,
        "functions": total,
        "annotated_pct": round(100 * annotated / total),
        "exported_annotated": exported_annotated,
        "exported_functions": exported_total,
        "exported_pct": round(100 * exported_annotated / exported_total),
        "documented": documented,
        "public_defs": public,
        "documented_pct": round(100 * documented / public),
        "legacy_loc": loc(sorted((ROOT / "music" / "legacy").glob("**/*.py"))),
    }


def measured_by_running():
    """The figures that need pytest, coverage and ruff to answer."""
    # Collected rather than passed. Some tests skip when an optional
    # external resource is absent -- the KEMAR measurements, the singing
    # engine -- so the number that passes depends on the machine, and a
    # figure that does is not a figure. Collection is the same everywhere.
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT, capture_output=True, text=True)
    count = re.search(r"(\d+) tests? collected", collected.stdout)
    if not count:
        raise SystemExit("could not read the collection:\n"
                         + collected.stdout[-2000:])
    # The docstring examples are the items collected from the package
    # itself rather than from tests/.
    doctests = len(re.findall(r"^music/\S+::", collected.stdout, re.M))

    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--cov=music", "--cov-branch",
         "--cov-report=term"],
        cwd=ROOT, capture_output=True, text=True)
    passed = re.search(r"(\d+) passed", tests.stdout)
    total_line = re.search(r"^TOTAL\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)%",
                           tests.stdout, re.M)
    if not (passed and total_line):
        raise SystemExit("could not read the test run:\n"
                         + tests.stdout[-2000:])

    lint = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "ALL", "music"],
        cwd=ROOT, capture_output=True, text=True)
    found = re.search(r"Found (\d+) error", lint.stdout)
    # The three largest kinds, which the table names beside the total.
    statistics = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "ALL",
         "--statistics", "music"],
        cwd=ROOT, capture_output=True, text=True).stdout
    kinds = {code: int(number) for number, code in re.findall(
        r"^\s*(\d+)\s+([A-Z]+\d+)\b", statistics, re.M)}

    return {
        "tests": int(count.group(1)),
        "doctests": doctests,
        "passed": int(passed.group(1)),
        "statements": int(total_line.group(1)),
        "missed": int(total_line.group(2)),
        "branches": int(total_line.group(3)),
        "partial": int(total_line.group(4)),
        "coverage_pct": int(total_line.group(5)),
        "ruff_all": int(found.group(1)) if found else 0,
        "ruff_quotes": kinds.get("Q000", 0),
        "ruff_untyped_arguments": kinds.get("ANN001", 0),
        "ruff_untyped_returns": kinds.get("ANN201", 0),
    }


def thousands(n):
    return f"{n:,}"


def table(figures, version, date):
    """The figures, as the block ASSESSMENT.md carries between markers."""
    f, t = figures, thousands
    rows = [
        ("Package", f"{f['modules']} modules, {t(f['package_loc'])} lines"),
        ("Public API", f"{f['api_names']} names"),
        ("Tests", f"{t(f['tests'])} collected by `pytest`, in "
                  f"{t(f['tests_loc'])} lines"),
        ("Docstring examples", f"{f['doctests']} docstrings, each run as a "
                               f"doctest"),
        ("Examples", f"{f['examples']} scripts, every one run by "
                     f"`tools/run_examples.py`"),
        ("Line and branch coverage",
         f"{f['coverage_pct']} % of {t(f['statements'])} statements and "
         f"{t(f['branches'])} branches: {f['missed']} statements missed, "
         f"{f['partial']} branches taken only one way"),
        ("Annotation coverage",
         f"{f['annotated']} / {f['functions']} functions "
         f"({f['annotated_pct']} %); {f['exported_annotated']} / "
         f"{f['exported_functions']} exported ({f['exported_pct']} %)"),
        ("Docstring coverage",
         f"{f['documented']} / {f['public_defs']} public definitions "
         f"({f['documented_pct']} %)"),
        ("Extended lint, `ruff check --select ALL music`",
         f"{t(f['ruff_all'])} findings, the largest kinds "
         f"{t(f['ruff_quotes'])} quote-style, "
         f"{t(f['ruff_untyped_arguments'])} missing argument annotations "
         f"and {t(f['ruff_untyped_returns'])} missing return annotations"),
        ("`legacy/`", f"{t(f['legacy_loc'])} lines"),
    ]
    return "\n".join(
        [BEGIN, f"Measured **{date}**, at `music` {version}.", "",
         "| Figure | Value |", "|---|---|"]
        + [f"| {label} | {value} |" for label, value in rows]
        + [END])


def current_table(text):
    """The block between the markers in `text`, markers included."""
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise SystemExit(
            "ASSESSMENT.md must hold the figures table between its two "
            "markers exactly once; they are what --write replaces. Put "
            f"them back rather than writing the table by hand:\n{BEGIN}\n"
            f"{END}")
    start = text.index(BEGIN)
    return text[start:text.index(END, start) + len(END)]


def table_date(block):
    """The measurement date the table carries."""
    match = re.search(r"Measured \*\*(\d{4}-\d\d-\d\d)\*\*", block)
    if not match:
        raise SystemExit("the figures table carries no measurement date")
    return match.group(1)


def table_rows(block):
    """{label: value} for each row of a figures table."""
    return dict(re.findall(r"^\| (.+?) \| (.+?) \|$", block, re.M)[1:])


def quoted(figures, version):
    """Each figure prose still quotes, as (label, regex, wanted text, file).

    RECONCILIATION.md and DISCREPANCIES.md are here for their version
    stamp. Neither is measured from here -- the reconciliation needs a
    MASS checkout -- but each is re-established against whatever the
    package currently is on every push, by
    `tests/test_mass_reconciliation.py` and by the tests each discrepancy
    names. So the version they stamp is the current one, and the date
    they carry is not: that belongs to the run against the checkout, and
    stays where it is.
    """
    exact, divergent, broken, total = (
        str(figures[key]) for key in ("mass_exact", "mass_divergent",
                                      "mass_broken", "mass_total"))
    return [
        ("version stamp, RECONCILIATION.md",
         r"(?<=`music` )\d+\.\d+\.\d+(?=\s+on every push)",
         version, RECONCILIATION),
        ("version stamp, DISCREPANCIES.md",
         r"(?<=`music` )\d+\.\d+\.\d+(?=\s+on every push)",
         version, DISCREPANCIES),
        ("register summary, sample-exact",
         r"(?<=\*\*)\d+(?= sample-exact, \d+ divergent)", exact,
         RECONCILIATION),
        ("register summary, divergent",
         r"(?<=sample-exact, )\d+(?= divergent, )", divergent, RECONCILIATION),
        ("register summary, not runnable",
         r"(?<=divergent, )\d+(?= where the reference does not run\.)",
         broken, RECONCILIATION),
        ("MASS sample-exact",
         r"(?<=\| \*\*)\d+(?= of \d+ routines sample-exact)", exact,
         ASSESSMENT),
        ("MASS routines", r"(?<= of )\d+(?= routines sample-exact)", total,
         ASSESSMENT),
        ("MASS divergent", r"(?<=sample-exact\*\*; )\d+(?= divergent)",
         divergent, ASSESSMENT),
        ("MASS not runnable",
         r"(?<=stated reason, )\d+(?= where the reference does not run)",
         broken, ASSESSMENT),
        ("README MASS sample-exact",
         r"(?<=routine by routine — )\d+(?= of \d+ are reproduced)", exact,
         README),
        ("README MASS routines", r"(?<= of )\d+(?= are reproduced sample)",
         total, README),
        ("README MASS divergent", r"\d+(?= diverge for reasons it states)",
         divergent, README),
        ("README MASS not runnable",
         r"(?<=and in )\d+(?= the reference itself)", broken, README),
        ("Zenodo abstract, sample-exact",
         r"(?<=itself: )\d+(?= of the reference’s)", exact, ZENODO),
        ("Zenodo abstract, routines",
         r"(?<=of the reference’s )\d+(?= routines)", total, ZENODO),
        ("Zenodo abstract, divergent", r"\d+(?= diverge for stated reasons)",
         divergent, ZENODO),
        ("Zenodo abstract, not runnable",
         r"(?<=and in )\d+(?= the reference itself does not run)", broken,
         ZENODO),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="correct the files rather than reporting")
    parser.add_argument("--fast", action="store_true",
                        help="check the quoted figures alone, measuring "
                             "nothing")
    args = parser.parse_args(argv)

    version = declared_version()
    checks = quoted(register(), version)
    texts = {path: path.read_text() for _l, _p, _w, path in checks}
    texts.setdefault(ASSESSMENT, ASSESSMENT.read_text())
    drifted = []
    for label, pattern, want, path in checks:
        found = re.findall(pattern, texts[path])
        if not found:
            raise SystemExit(
                f"the {label} figure is not where this script looks for it. "
                f"{path.name} has been rewritten around it; fix the "
                "pattern in tools/assessment_figures.py rather than "
                "deleting the check.")
        for have in set(found):
            if have != want:
                drifted.append((label, have, want))
        if args.write:
            texts[path] = re.sub(pattern, want, texts[path])

    checked = len(checks)
    if not args.fast:
        block = current_table(texts[ASSESSMENT])
        date = (datetime.date.today().isoformat() if args.write
                else table_date(block))
        fresh = table(scan() | measured_by_running(), version, date)
        have, want = table_rows(block), table_rows(fresh)
        for label in sorted(set(have) | set(want)):
            if have.get(label) != want.get(label):
                drifted.append((f"table, {label}", have.get(label),
                                want.get(label)))
        stamped = re.search(r"at `music` (\S+)\.", block)
        if not stamped or stamped.group(1) != version:
            drifted.append(("table, version",
                            stamped.group(1) if stamped else None, version))
        if args.write:
            texts[ASSESSMENT] = texts[ASSESSMENT].replace(block, fresh)
        checked += len(want) + 1

    if args.write:
        for path, text in texts.items():
            path.write_text(text)

    if not drifted:
        files = ", ".join(sorted({path.name for *_x, path in checks}))
        print(f"{files} agree with the package, on {checked} figures.")
        return 0

    for label, have, want in drifted:
        print(f"  {label}: says {have}, should say {want}")
    if args.write:
        print(f"\ncorrected {len(drifted)}.")
        return 0
    print(f"\n{len(drifted)} figure(s) drifted. "
          "`python tools/assessment_figures.py --write` corrects them.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
