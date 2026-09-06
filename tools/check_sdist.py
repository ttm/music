#!/usr/bin/env python3
"""Build the source distribution, unpack it, and run its tests.

The sdist ships ``tests/``. For 1.5.0 and every release before it, it
shipped them without ``conftest.py``, ``pytest.ini``, ``tools/``,
``docs/`` or ``tests/fixtures/``, so all thirty-eight test files failed to
collect: a redistributor building from source and running the suite --
which is the reason to ship it -- got two collection errors and no tests.

Nothing noticed, because every check in this repository runs against the
working tree, where those files are present. This is the one that runs
against what actually leaves the machine.

    python tools/check_sdist.py            # build, unpack, test
    python tools/check_sdist.py --keep     # and leave the tree behind

It is slow -- a build and a full suite -- so it is part of the release
gate rather than of CI on every push. `MANIFEST.in` is what it checks.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Files the suite reads from the repository root rather than from the
#: package. Each is checked for by name before the tests run, so a missing
#: one is reported as itself instead of as a collection error thirty lines
#: into pytest's output.
NEEDED = (
    'conftest.py',
    'pytest.ini',
    'docs/api.rst',
    'docs/tutorial.rst',
    '.zenodo.json',
    'tools/mass_reconcile.py',
    'tools/mass_reference.py',
    'tests/fixtures/mass_reference.npz',
)


def build(into: Path) -> Path:
    """Build the sdist into `into`, and return the tarball.

    Clearing the egg-info first is not tidiness. setuptools reuses the
    ``SOURCES.txt`` it left there last time, so a file dropped from
    MANIFEST.in keeps being shipped and this check keeps passing --
    which it did, the first time it was pointed at a deliberately
    broken manifest.
    """
    shutil.rmtree(ROOT / 'music.egg-info', ignore_errors=True)
    subprocess.run(
        (sys.executable, '-m', 'build', '--sdist', '--outdir', str(into)),
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    tarballs = sorted(into.glob('*.tar.gz'))
    if len(tarballs) != 1:
        raise SystemExit(f'expected one sdist in {into}, found {tarballs}')
    return tarballs[0]


def unpack(tarball: Path, into: Path) -> Path:
    with tarfile.open(tarball) as archive:
        archive.extractall(into, filter='data')
    roots = [path for path in into.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise SystemExit(f'expected one directory in {tarball}, got {roots}')
    return roots[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keep', action='store_true',
                        help='leave the unpacked tree in place')
    args = parser.parse_args()

    scratch = Path(tempfile.mkdtemp(prefix='music-sdist-'))
    try:
        tarball = build(scratch)
        print(f'  built {tarball.name}')
        tree = unpack(tarball, scratch / 'unpacked')

        missing = [name for name in NEEDED if not (tree / name).exists()]
        if missing:
            print(f'\nthe sdist is missing what its tests need: {missing}')
            print('add them to MANIFEST.in')
            return 1
        print(f'  carries all {len(NEEDED)} files the suite reads')

        shipped = len(list((tree / 'tests').glob('test_*.py')))
        print(f'  running {shipped} test files from the unpacked tree')
        result = subprocess.run(
            (sys.executable, '-m', 'pytest', '-q'), cwd=tree)
        if result.returncode:
            print('\nthe sdist ships tests that do not pass in it')
            return 1
        print('\nthe source distribution tests itself')
        return 0
    finally:
        if args.keep:
            print(f'left in {scratch}')
        else:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == '__main__':
    raise SystemExit(main())
