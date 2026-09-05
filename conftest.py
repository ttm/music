"""Shared pytest setup.

Puts the repository root on ``sys.path`` so ``import music`` resolves to the
working tree, whether or not the package is installed.  Keeping it here means
the test modules import the package the same way a user does, instead of each
one reaching for ``sys.path`` or loading modules by file path.

It also makes the docstring examples runnable.  They are written the way a
reader meets them -- ``write_wav_mono(note())``, with the package's names in
scope, not ``music.core.io.write_wav_mono(music.core.synths.notes.note())``
-- so running them as doctests needs that namespace supplied and a scratch
directory to write into.  See :func:`_doctest_namespace`.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def _doctest_namespace(request, doctest_namespace, tmp_path):
    """Give every docstring example the namespace a reader would have.

    An example in ``music/core/synths/notes.py`` calls ``write_wav_mono``,
    which that module does not import: it is written for someone who has
    done ``from music import *``, which is how the published API reference
    renders it and how a reader will copy it. Supplying that namespace is
    what lets the examples be checked at all, rather than being prose that
    happens to be formatted as code.

    Examples that write files run with the working directory set to a
    scratch path, so ``write_wav_mono(note())`` leaves its ``asound.wav``
    there instead of in the repository.
    """
    if not isinstance(request.node, pytest.DoctestItem):
        yield
        return

    import numpy
    import music

    for name in music.__all__:
        doctest_namespace[name] = getattr(music, name)
    doctest_namespace["music"] = music
    doctest_namespace["np"] = numpy
    doctest_namespace["numpy"] = numpy
    doctest_namespace["Path"] = Path

    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(previous)
