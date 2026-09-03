#!/usr/bin/env python3
"""Run every example, and fail if any of them stops working.

The examples are the only part of this repository that nothing ran
automatically, and it showed. Deferring the ``music.structures`` import
removed the submodule attribute that three of them use, and the break
survived a full test suite at 100% coverage, a clean mypy, a clean ruff
and a docs build, because every one of those checks looks at the package
and none of them looks at a caller. The examples are the only callers
this repository has.

They are also the code most likely to be copied. A reader who wants to
know how to use this package opens ``examples/`` before the API
reference, so an example that does not run is worse than a missing one.

Usage
-----
::

    python tools/run_examples.py           # run them all
    python tools/run_examples.py --list    # just say what would run

Each example is run in a scratch directory, because they write WAV files
next to themselves and the repository should not collect them.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).parent.parent
EXAMPLES = ROOT / "examples"

#: Examples that cannot run unattended, and why. Anything not named here
#: is expected to run to completion with a zero exit status, so a new
#: example is covered the moment it is added rather than when someone
#: remembers to list it.
SKIP = {
    "singing_demo.py":
        "needs the external eCantorix engine, which setup_engine() clones "
        "at runtime and which needs git, make, perl and espeak",
}


def examples() -> list[pathlib.Path]:
    """Every example, in a stable order."""
    return sorted(EXAMPLES.glob("*.py"))


def run(path: pathlib.Path) -> tuple[bool, str, float]:
    """Run one example in a scratch directory.

    Returns
    -------
    tuple
        Whether it succeeded, its combined output, and how long it took.
    """
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as scratch:
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=scratch,
            capture_output=True,
            text=True,
            # The examples import `music`, and CI installs the package,
            # but running from a checkout should work too.
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
    return (result.returncode == 0,
            (result.stdout + result.stderr).strip(),
            time.monotonic() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true",
                        help="print what would run and exit")
    args = parser.parse_args()

    found = examples()
    if not found:
        print("no examples found", file=sys.stderr)
        return 1

    if args.list:
        for path in found:
            mark = "skip" if path.name in SKIP else "run "
            print(f"{mark}  {path.name}")
        return 0

    failures = []
    for path in found:
        if path.name in SKIP:
            print(f"SKIP  {path.name}  ({SKIP[path.name]})")
            continue
        ok, output, seconds = run(path)
        if ok:
            print(f"PASS  {path.name}  ({seconds:.1f}s)")
        else:
            print(f"FAIL  {path.name}  ({seconds:.1f}s)")
            print("\n".join("      " + line
                            for line in output.splitlines()[-15:]))
            failures.append(path.name)

    ran = len(found) - len(SKIP)
    if failures:
        print(f"\n{len(failures)} of {ran} examples failed: "
              f"{', '.join(failures)}")
        return 1
    print(f"\nall {ran} examples ran; {len(SKIP)} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
