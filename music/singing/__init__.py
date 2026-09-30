"""Singing: a lyric sung to a melody, by eCantorix or by espeak-ng and PSOLA.

:func:`sing` is the entry point. Its default backend, eCantorix, is a Perl
engine :func:`setup_engine` clones; ``backend="psola"`` is
:mod:`music.singing.psola`, which needs espeak-ng and
``pip install 'music[singing]'``.
"""

from .bootstrap import get_engine, make_test_song, setup_engine
from .perform import sing

__all__ = [
    'get_engine',
    'setup_engine',
    'make_test_song',
    'sing',
]
