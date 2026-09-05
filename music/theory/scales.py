"""Scales, modes and the harmonic series, as the MASS companion paper gives
them.

`notesInMusic.tex` states the seven diatonic modes as sets of semitone
offsets from a tonic (``eq:escalas``), the rotation of one step pattern that
generates all of them (``eq:relacaoDia``), the three minor scales
(``eq:escalasMenores``) and the first twenty partials of the harmonic series
in semitones (``eq:serieHarmonica``). This module is those, and the
routines for reaching a scale by name or by rotation.

Everything here counts semitones from a tonic of zero, which is what
:func:`~music.pitch_to_freq` takes, so a scale becomes frequencies in one
step::

    >>> [round(f, 2)
    ...  for f in pitch_to_freq(start_freq=220.0, semitones=scale('ionian'))]
    [220.0, 246.94, 277.18, 293.66, 329.63, 369.99, 415.3]

References
----------
.. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
       representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)
"""
from __future__ import annotations

import math

__all__ = ['DIATONIC_STEPS', 'MODES', 'MINOR_SCALES', 'SCALES',
           'HARMONIC_SERIES_AS_PRINTED', 'scale', 'mode_by_rotation',
           'harmonic_series']

#: The step pattern every diatonic mode is a rotation of, from
#: ``eq:relacaoDia``: tone, tone, semitone, tone, tone, tone, semitone.
DIATONIC_STEPS: tuple[int, ...] = (2, 2, 1, 2, 2, 2, 1)

#: The seven modes of ``eq:escalas``, each as semitones above its tonic.
#: The rotation of :data:`DIATONIC_STEPS` that produces each is the
#: ``kappa`` of ``eq:relacaoDia``, and :func:`mode_by_rotation` is the other
#: way to reach them.
MODES: dict[str, tuple[int, ...]] = {
    'dorian': (0, 2, 3, 5, 7, 9, 10),
    'phrygian': (0, 1, 3, 5, 7, 8, 10),
    'lydian': (0, 2, 4, 6, 7, 9, 11),
    'mixolydian': (0, 2, 4, 5, 7, 9, 10),
    'aeolian': (0, 2, 3, 5, 7, 8, 10),
    'locrian': (0, 1, 3, 5, 6, 8, 10),
    'ionian': (0, 2, 4, 5, 7, 9, 11),
}

#: The three minor scales of ``eq:escalasMenores``. The melodic minor is
#: fifteen degrees rather than seven: it rises one way and falls another,
#: and the article writes both directions as one sequence.
MINOR_SCALES: dict[str, tuple[int, ...]] = {
    'natural minor': (0, 2, 3, 5, 7, 8, 10),
    'harmonic minor': (0, 2, 3, 5, 7, 8, 11),
    'melodic minor': (0, 2, 3, 5, 7, 9, 11, 12, 10, 8, 7, 5, 3, 2, 0),
}

#: Every scale this module knows, by name. The two synonyms the article
#: gives -- ionian for the major scale, aeolian for the natural minor --
#: are both present and are the same tuple.
SCALES: dict[str, tuple[int, ...]] = {
    **MODES,
    **MINOR_SCALES,
    'major': MODES['ionian'],
    'minor': MODES['aeolian'],
}

#: The first twenty partials as ``eq:serieHarmonica`` prints them, in
#: semitones above the fundamental. They are not integers: the seventh
#: partial is a third of a semitone flat of the minor seventh, and the
#: eleventh is nearly half a semitone from anything on the keyboard, which
#: is why the series and equal temperament are different things.
#:
#: Nineteen of the twenty are ``12 * log2(n)`` to within the two decimals
#: they are printed at. The sixth is ``31 + 0.2`` where the exact value is
#: ``31.02``, which ``31 + 0.02`` would give: a typo in the paper rather
#: than a different claim. :func:`harmonic_series` computes the exact value
#: instead; this table is kept as printed so the two can be compared, which
#: `tests/test_theory.py` does. See `DISCREPANCIES.md`.
HARMONIC_SERIES_AS_PRINTED: tuple[float, ...] = (
    0, 12, 19 + 0.02, 24, 28 - 0.14, 31 + 0.2, 34 - 0.31,
    36, 38 + 0.04, 40 - 0.14, 42 - 0.49, 43 + 0.02,
    44 + 0.41, 46 - 0.31, 47 - 0.12,
    48, 49 + 0.05, 50 + 0.04, 51 - 0.02, 52 - 0.14,
)


def scale(name: str = 'major', tonic: int = 0) -> tuple[int, ...]:
    """The semitones of a named scale, counted from `tonic`.

    Parameters
    ----------
    name : string
        One of :data:`SCALES`: the seven modes, the three minor scales, or
        ``"major"`` and ``"minor"`` for the two the article names twice.
    tonic : integer
        Semitones to shift the whole scale by. Zero leaves it on its own
        tonic, which is how the article writes it.

    Returns
    -------
    tuple of integers
        The degrees, in ascending order for every scale but the melodic
        minor, which the article writes as a rise and a fall.

    Raises
    ------
    ValueError
        If `name` is not a scale this module knows.

    See Also
    --------
    mode_by_rotation : the same seven modes, reached by rotating the steps.
    pitch_to_freq : turns these semitones into frequencies.

    Examples
    --------
    >>> scale('major')
    (0, 2, 4, 5, 7, 9, 11)
    >>> scale('minor') == scale('aeolian')
    True
    >>> scale('dorian', tonic=2)
    (2, 4, 5, 7, 9, 11, 12)
    """
    try:
        degrees = SCALES[name]
    except KeyError:
        raise ValueError(
            f'{name!r} is not a scale here; try one of '
            f'{sorted(SCALES)}') from None
    return tuple(degree + tonic for degree in degrees)


def mode_by_rotation(kappa: int = 6) -> tuple[int, ...]:
    """A diatonic mode, built by rotating the steps as ``eq:relacaoDia`` does.

    ``e_0 = 0`` and ``e_i = d[(i + kappa) % 7] + e_(i-1)``, with ``d`` the
    seven steps of :data:`DIATONIC_STEPS`. Each rotation is one of the seven
    modes: ``kappa = 6`` is ionian, and 0 through 5 are dorian, phrygian,
    lydian, mixolydian, aeolian and locrian.

    Parameters
    ----------
    kappa : integer
        Which rotation. Taken modulo 7, so any integer names a mode.

    Returns
    -------
    tuple of integers
        The seven degrees of that mode.

    See Also
    --------
    scale : the same modes, by name.

    Examples
    --------
    >>> mode_by_rotation(6)
    (0, 2, 4, 5, 7, 9, 11)
    >>> mode_by_rotation(6) == scale('ionian')
    True
    >>> [mode_by_rotation(k)[1] for k in range(3)]
    [2, 1, 2]

    Notes
    -----
    That every mode is one rotation of one pattern is the content of the
    equation; naming them is bookkeeping on top of it.
    """
    degrees = [0]
    for i in range(1, len(DIATONIC_STEPS)):
        degrees.append(DIATONIC_STEPS[(i + kappa) % len(DIATONIC_STEPS)]
                       + degrees[-1])
    return tuple(degrees)


def harmonic_series(partials: int = 20) -> tuple[float, ...]:
    """The first `partials` harmonics, in semitones above the fundamental.

    The n-th partial is at ``12 * log2(n)``, which is what this computes.
    :data:`HARMONIC_SERIES_AS_PRINTED` is the article's table of the first
    twenty, rounded to two decimals and with one typo.

    Parameters
    ----------
    partials : integer
        How many, from 1 upwards. The article tabulates twenty; nothing
        here stops at that.

    Returns
    -------
    tuple of floats
        Semitones, with the fractions that make the series inequal to any
        tempered scale.

    Raises
    ------
    ValueError
        If `partials` is less than 1.

    See Also
    --------
    midi_to_hz_interval : turns one of these semitone counts into a ratio.

    Examples
    --------
    >>> [round(h, 2) for h in harmonic_series(4)]
    [0.0, 12.0, 19.02, 24.0]
    >>> round(harmonic_series(7)[-1], 2)
    33.69

    Notes
    -----
    Every octave is exact -- partials 1, 2, 4, 8 and 16 land on 0, 12, 24,
    36 and 48 -- and nothing else does. The third partial is 2 cents sharp
    of a tempered fifth and the seventh is 31 cents flat of a tempered
    minor seventh, which is the size of the disagreement between this
    series and the tuning of :func:`scale`.
    """
    if partials < 1:
        raise ValueError(f'partials must be at least 1; got {partials}')
    return tuple(12 * math.log2(n) for n in range(1, partials + 1))
