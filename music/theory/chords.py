"""Chords, as the MASS companion paper builds them.

`notesInMusic.tex` gives the four triads as sets of semitones above a root
(its equation ``triades``), then says what to add to reach the tetrads: a
further third, 10 for a minor seventh or 11 for a major one, and
``+/- 12`` on any note for an inversion or an open position. That is what
this module is.

Everything counts semitones from a root of zero, like
:mod:`music.theory.scales`, so a chord becomes frequencies through
:func:`~music.pitch_to_freq` and a sound through
:func:`~music.note`::

    >>> [round(f, 2) for f in pitch_to_freq(220.0, chord('major'))]
    [220.0, 277.18, 329.63]

References
----------
.. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
       representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)
"""
from __future__ import annotations

__all__ = ['TRIADS', 'SEVENTHS', 'CHORDS', 'chord', 'add_seventh', 'invert']

#: The four triads of ``triades``: a root, a third that is major or minor,
#: and a fifth that is perfect, diminished or augmented.
TRIADS: dict[str, tuple[int, ...]] = {
    'major': (0, 4, 7),
    'minor': (0, 3, 7),
    'diminished': (0, 3, 6),
    'augmented': (0, 4, 8),
}

#: The tetrads the article reaches by adding one more third: 10 semitones
#: for a minor seventh, 11 for a major one.
SEVENTHS: dict[str, tuple[int, ...]] = {
    'dominant seventh': TRIADS['major'] + (10,),
    'major seventh': TRIADS['major'] + (11,),
    'minor seventh': TRIADS['minor'] + (10,),
    'minor major seventh': TRIADS['minor'] + (11,),
    'half diminished': TRIADS['diminished'] + (10,),
    'diminished seventh': TRIADS['diminished'] + (9,),
}

#: Every chord this module knows, by name.
CHORDS: dict[str, tuple[int, ...]] = {**TRIADS, **SEVENTHS}


def chord(name: str = 'major', root: int = 0) -> tuple[int, ...]:
    """The semitones of a named chord, counted from `root`.

    Parameters
    ----------
    name : string
        One of :data:`CHORDS`: the four triads or the six tetrads.
    root : integer
        Semitones to shift the whole chord by.

    Returns
    -------
    tuple of integers
        The chord's notes, in ascending order.

    Raises
    ------
    ValueError
        If `name` is not a chord this module knows.

    See Also
    --------
    add_seventh : builds a tetrad from a triad, as the article describes.
    invert : moves a note by an octave, for inversions and open positions.

    Examples
    --------
    >>> chord('major')
    (0, 4, 7)
    >>> chord('minor seventh', root=3)
    (3, 6, 10, 13)
    """
    try:
        notes = CHORDS[name]
    except KeyError:
        raise ValueError(
            f'{name!r} is not a chord here; try one of '
            f'{sorted(CHORDS)}') from None
    return tuple(note + root for note in notes)


def add_seventh(notes: tuple[int, ...], major: bool = False) \
        -> tuple[int, ...]:
    """One more third on top of a triad.

    The article: "it is sufficient to include 10 as the highest note to
    achieve a tetrad with a minor seventh, or include 11 in order to
    achieve a tetrad with a major seventh".

    Parameters
    ----------
    notes : tuple of integers
        The triad, as :func:`chord` returns one.
    major : boolean
        11 semitones rather than 10.

    Returns
    -------
    tuple of integers
        The tetrad.

    See Also
    --------
    chord : the tetrads this makes, reachable by name.

    Examples
    --------
    >>> add_seventh(chord('major'))
    (0, 4, 7, 10)
    >>> add_seventh(chord('major'), major=True) == chord('major seventh')
    True
    """
    root = notes[0]
    return tuple(notes) + (root + (11 if major else 10),)


def invert(notes: tuple[int, ...], degree: int = 0, octaves: int = 1) \
        -> tuple[int, ...]:
    """Move one note of a chord by whole octaves.

    The article: "Inversions and open positions can be obtained with the
    addition of +/- 12 to the selected note."

    Parameters
    ----------
    notes : tuple of integers
        The chord.
    degree : integer
        Which note to move, indexed from the bottom.
    octaves : integer
        How many octaves, and which way. Positive moves it up.

    Returns
    -------
    tuple of integers
        The chord with that note moved, sorted again so the result reads
        from the bottom up.

    Raises
    ------
    IndexError
        If `degree` is not a note of `notes`.

    Examples
    --------
    >>> invert(chord('major'), degree=0)
    (4, 7, 12)
    >>> invert(chord('major'), degree=2, octaves=-1)
    (-5, 0, 4)
    >>> invert(chord('major'), degree=2, octaves=1)  # an open position
    (0, 4, 19)
    """
    moved = list(notes)
    moved[degree] = moved[degree] + 12 * octaves
    return tuple(sorted(moved))
