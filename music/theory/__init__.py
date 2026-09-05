"""Scales, chords and the harmonic series, from the MASS companion paper.

`notesInMusic.tex` states the material this subpackage implements: the
diatonic modes and the rotation that generates them, the three minor
scales, the harmonic series, and the triads and tetrads of tonal harmony.
Everything is counted in semitones from a tonic or root of zero, which is
what :func:`~music.pitch_to_freq` takes.

What it does not implement is the rest of that paper: harmonic expansion,
chromatic mediants, modulation and counterpoint are described there and
not here. See `ASSESSMENT.md`.
"""

from .chords import (CHORDS, SEVENTHS, TRIADS, add_seventh, chord, invert)
from .scales import (DIATONIC_STEPS, HARMONIC_SERIES_AS_PRINTED,
                     MINOR_SCALES, MODES, SCALES, harmonic_series,
                     mode_by_rotation, scale)

__all__ = [
    'CHORDS',
    'DIATONIC_STEPS',
    'HARMONIC_SERIES_AS_PRINTED',
    'MINOR_SCALES',
    'MODES',
    'SCALES',
    'SEVENTHS',
    'TRIADS',
    'add_seventh',
    'chord',
    'harmonic_series',
    'invert',
    'mode_by_rotation',
    'scale',
]
