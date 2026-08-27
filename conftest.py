"""Shared pytest setup.

Puts the repository root on ``sys.path`` so ``import music`` resolves to the
working tree, whether or not the package is installed.  Keeping it here means
the test modules import the package the same way a user does, instead of each
one reaching for ``sys.path`` or loading modules by file path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
