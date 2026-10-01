"""Singing: a lyric sung to a melody, by espeak-ng and PSOLA or by eCantorix.

:func:`sing` is the entry point. Its default backend,
:mod:`music.singing.psola`, needs espeak-ng and
``pip install 'music[singing]'``. ``backend="ecantorix"`` is a Perl engine
:func:`setup_engine` clones, kept as the reference the default is compared
with, and for its effects.
"""

from .bootstrap import get_engine, make_test_song, setup_engine
from .perform import sing

__all__ = [
    'get_engine',
    'setup_engine',
    'make_test_song',
    'sing',
]
