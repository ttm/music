#!/usr/bin/env python3
"""Keep ASSESSMENT.md's numbers true, by measuring them.

ASSESSMENT.md opens by saying that a snapshot nobody updates
misrepresents the code it describes. It then went stale four times in
two days, and was corrected by hand each time -- once while preparing a
release, with the wrong test count already committed. Issue #70 asks for
the practice of keeping it current; a practice that depends on someone
remembering is the thing that keeps failing.

Every figure in that file comes from running something. So this runs
them and compares, and the release gate refuses to publish a file that
disagrees with the package it describes.

Usage
-----
::

    python tools/assessment_figures.py           # report any drift
    python tools/assessment_figures.py --write   # correct it
    python tools/assessment_figures.py --fast    # skip the slow measures

``--fast`` skips the three figures that need pytest, coverage and ruff
to run, and checks only the ones an AST scan can settle. The others take
about a minute between them.

What is deliberately not checked: the wall-clock time in the test-suite
row, which is a property of the machine rather than of the package, and
the history in the opening paragraphs, which is about how the file went
wrong once and must not be rewritten to match the present.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent.parent
ASSESSMENT = ROOT / "ASSESSMENT.md"


def package_files():
    return sorted(p for p in (ROOT / "music").glob("**/*.py")
                  if "ecantorix" not in str(p))


def loc(paths):
    return sum(len(p.read_text().splitlines()) for p in paths)


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
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--cov=music",
         "--cov-report=term"],
        cwd=ROOT, capture_output=True, text=True)
    passed = re.search(r"(\d+) passed", tests.stdout)
    total_line = re.search(r"^TOTAL\s+(\d+)\s+(\d+)\s+(\d+)%",
                           tests.stdout, re.M)
    if not (passed and total_line):
        raise SystemExit("could not read the test run:\n"
                         + tests.stdout[-2000:])

    lint = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "ALL", "music"],
        cwd=ROOT, capture_output=True, text=True)
    found = re.search(r"Found (\d+) error", lint.stdout)

    return {
        "tests": int(passed.group(1)),
        "statements": int(total_line.group(1)),
        "missed": int(total_line.group(2)),
        "coverage_pct": int(total_line.group(3)),
        "ruff_all": int(found.group(1)) if found else 0,
    }


def thousands(n):
    return f"{n:,}"


def expectations(figures):
    """Each figure, as (label, regex over ASSESSMENT.md, wanted text).

    The regexes are anchored on enough surrounding words to miss the
    opening history, which quotes figures from when this file was wrong
    and must keep quoting them.
    """
    wanted = [
        ("modules", r"(?<=: )\d+(?= modules,)", str(figures["modules"])),
        ("package LOC", r"(?<=modules, )[\d,]+(?= LOC package)",
         thousands(figures["package_loc"])),
        ("tests LOC", r"(?<=LOC package \+ )[\d,]+(?= LOC tests)",
         thousands(figures["tests_loc"])),
        ("public API names", r"(?<=LOC tests, )\d+(?= names)",
         str(figures["api_names"])),
        ("annotation coverage",
         r"(?<=AST scan \| \*\*)\d+ / \d+ functions \(\d+ %\)",
         f"{figures['annotated']} / {figures['functions']} functions "
         f"({figures['annotated_pct']} %)"),
        ("exported annotation coverage",
         r"(?<=%\)\*\*; )\d+ / \d+ exported \(\d+ %\)",
         f"{figures['exported_annotated']} / "
         f"{figures['exported_functions']} exported "
         f"({figures['exported_pct']} %)"),
        ("docstring coverage",
         r"(?<=AST scan \| \*\*)\d+ / \d+ public defs \(\d+ %\)",
         f"{figures['documented']} / {figures['public_defs']} public defs "
         f"({figures['documented_pct']} %)"),
        ("annotation coverage, restated",
         r"(?<=Annotation coverage at )\d+(?= %;)",
         str(figures["annotated_pct"])),
        ("annotation coverage, restated again",
         r"(?<=\*\*Annotation coverage is )\d+(?= %\*\*)",
         str(figures["annotated_pct"])),
        ("exported annotation, restated",
         r"(?<=and )\d+(?= % across the exported API)",
         str(figures["exported_pct"])),
        ("legacy LOC", r"(?<=\*\*`legacy/` is )[\d,]+(?= LOC\*\*)",
         thousands(figures["legacy_loc"])),
    ]
    if "tests" in figures:
        wanted += [
            ("test count", r"(?<=\| \*\*)\d+(?= passed\*\*)",
             str(figures["tests"])),
            ("coverage", r"(?<=\| \*\*)\d+ %(?=\*\* \([\d,]+ stmts)",
             f"{figures['coverage_pct']} %"),
            ("statements", r"(?<=\*\* \()[\d,]+(?= stmts)",
             thousands(figures["statements"])),
            ("statements missed", r"(?<= stmts, )\d+(?= missed\))",
             str(figures["missed"])),
            ("extended lint",
             r"(?<=`ruff check --select ALL music` \| )[\d,]+(?= findings)",
             thousands(figures["ruff_all"])),
            ("extended lint, restated",
             r"(?<=extended lint set reports )[\d,]+(?= findings\*\*)",
             thousands(figures["ruff_all"])),
        ]
    return wanted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="correct ASSESSMENT.md rather than reporting")
    parser.add_argument("--fast", action="store_true",
                        help="skip the figures that need pytest and ruff")
    args = parser.parse_args(argv)

    figures = scan()
    if not args.fast:
        figures |= measured_by_running()

    text = ASSESSMENT.read_text()
    drifted = []
    for label, pattern, want in expectations(figures):
        found = re.findall(pattern, text)
        if not found:
            raise SystemExit(
                f"the {label} figure is not where this script looks for it. "
                "ASSESSMENT.md has been rewritten around it; fix the "
                "pattern in tools/assessment_figures.py rather than "
                "deleting the check.")
        for have in set(found):
            if have != want:
                drifted.append((label, have, want))
        if args.write:
            text = re.sub(pattern, want, text)

    if args.write:
        ASSESSMENT.write_text(text)

    if not drifted:
        print(f"ASSESSMENT.md agrees with the package, on "
              f"{len(expectations(figures))} figures.")
        return 0

    for label, have, want in drifted:
        print(f"  {label}: says {have}, measures {want}")
    if args.write:
        print(f"\ncorrected {len(drifted)}.")
        return 0
    print(f"\n{len(drifted)} figure(s) drifted. "
          "`python tools/assessment_figures.py --write` corrects them.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
