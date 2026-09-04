"""Load the MASS reference implementation for comparison.

The reference is `src/aux/functions.py` in https://github.com/ttm/mass.  It is
GPL-3 and this package is MIT, so it is **not** vendored here: this module
reads whatever checkout the caller points at, and the only thing that ever
enters this repository is the numbers that come out of running it.

The file was written for Python 2 and NumPy 1.x and does not import under
either current runtime.  Loading it therefore needs source patches.  Each one
is listed in `PATCHES` with the reason, so that "the reference does not run"
stays a finding on the record rather than a detail buried in a loader.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

#: Substitutions applied to the reference source before executing it, as
#: (pattern, replacement, why) triples.  Every one is a defect in the
#: reference rather than a difference in convention.
PATCHES: tuple[tuple[str, str, str], ...] = (
    (r'^from HRTF import \*$', '',
     'HRTF.py sits beside the file but is not importable as written'),
    (r'\bn\.linspace\(-1, 1, Lt/2\b', 'n.linspace(-1, 1, int(Lt/2)',
     'true division makes the table length a float; NumPy rejects it'),
    (r'\bn\.int\b', 'int',
     'np.int was removed in NumPy 1.24'),
    (r'\bn\.float\b', 'float',
     'np.float was removed in NumPy 1.24'),
)

DEFAULT_LOCATIONS = ('../mass', '~/repos/mass', '~/rep/mass')


class ReferenceNotFound(Exception):
    """Raised when no MASS checkout can be located."""


def locate(explicit: str | None = None) -> Path:
    """Return the path to the MASS reference file.

    Looks at `explicit`, then `$MASS_SRC`, then a few conventional checkout
    locations.  The path may name either the repository root or the file.
    """
    candidates = [explicit, os.environ.get('MASS_SRC'), *DEFAULT_LOCATIONS]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        for probe in (path, path / 'src' / 'aux' / 'functions.py'):
            if probe.is_file():
                return probe
    raise ReferenceNotFound(
        'no MASS checkout found; pass --mass PATH or set MASS_SRC. '
        'Clone it from https://github.com/ttm/mass'
    )


def load(explicit: str | None = None) -> dict:
    """Execute the reference and return its namespace.

    The namespace holds the 36 reference routines and the waveform tables
    they default to.
    """
    path = locate(explicit)
    source = path.read_text()
    for pattern, replacement, _why in PATCHES:
        source = re.sub(pattern, replacement, source, flags=re.MULTILINE)
    namespace: dict = {'__name__': 'mass_reference', '__file__': str(path)}
    exec(compile(source, str(path), 'exec'), namespace)
    return namespace
