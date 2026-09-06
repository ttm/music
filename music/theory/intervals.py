"""Intervals by their traditional names, as the MASS companion paper gives
them.

`notesInMusic.tex` tabulates the simple intervals against their semitone
counts (its ``eq:intervalos``) and states the rules that generate the rest:
a degree is one more than the number of steps it spans, an interval wider
than an octave is the simple one plus a multiple of seven degrees, a major
interval lowered by a semitone is minor, and a perfect or major one raised
by a semitone is augmented.

The rest of `music.theory` counts semitones, which is what
:func:`~music.pitch_to_freq` takes. This is the other direction: it turns
``"M3"`` into 4 and 4 back into ``"M3"``, so an interval can be written the
way a musician writes it and still become a frequency::

    >>> pitch_to_freq(220.0, [interval("P1"), interval("M3"), interval("P5")])
    [220.0, 277.1826309768721, 329.6275569128699]

References
----------
.. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
       representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)
"""
from __future__ import annotations

import math
import re

__all__ = ['SIMPLE_INTERVALS', 'CONSONANCE', 'interval', 'interval_names',
           'consonance', 'interval_between']

#: The simple intervals -- an octave wide at most -- by traditional
#: notation, from the table of ``eq:intervalos``. The tritone has three
#: names for one interval, which is the article's own point about it.
SIMPLE_INTERVALS: dict[str, int] = {
    'P1': 0, 'm2': 1, 'M2': 2, 'm3': 3, 'M3': 4, 'P4': 5,
    'aug4': 6, 'dim5': 6, 'TT': 6, 'tri': 6,
    'P5': 7, 'm6': 8, 'M6': 9, 'm7': 10, 'M7': 11, 'P8': 12,
}

#: How the article classifies each simple interval. "Perfect fourth is a
#: special case, as it is a perfect consonance when considered as an
#: inversion of the perfect fifth and a dissonance or an imperfect
#: consonance otherwise"; the tritone is a dissonance in Western music and
#: "consonant in some cultures". Both are recorded as what they are --
#: unsettled -- rather than forced into one of the four.
CONSONANCE: dict[int, str] = {
    0: 'perfect consonance',
    1: 'strong dissonance',
    2: 'weak dissonance',
    3: 'imperfect consonance',
    4: 'imperfect consonance',
    5: 'context dependent',
    6: 'context dependent',
    7: 'perfect consonance',
    8: 'imperfect consonance',
    9: 'imperfect consonance',
    10: 'weak dissonance',
    11: 'strong dissonance',
    12: 'perfect consonance',
}

#: The degrees that take perfect rather than major or minor, within an
#: octave: unison, fourth, fifth and octave.
_PERFECT_DEGREES = (1, 4, 5, 8)

_NOTATION = re.compile(r'^(?P<quality>P|M|m|aug|A|dim|d)(?P<degree>\d+)$')


def _simple(degree: int) -> tuple[int, int]:
    """A degree as (its simple form, how many octaves were taken off).

    "An interval wider than an octave ... can be achieved by adding a
    multiple of 7 to the simple interval", so this undoes that. A degree of
    8 is the octave itself and stays whole rather than becoming a unison.
    """
    if degree == 8:
        return 8, 0
    return ((degree - 1) % 7) + 1, (degree - 1) // 7


def interval(name: str) -> int:
    """The number of semitones in an interval, from its traditional name.

    Parameters
    ----------
    name : string
        ``"P5"``, ``"m3"``, ``"aug4"``, ``"TT"``, or a compound such as
        ``"M9"`` or ``"m16"``. Quality is ``P``, ``M``, ``m``, ``aug``
        (or ``A``) and ``dim`` (or ``d``).

    Returns
    -------
    integer
        Semitones above the lower note.

    Raises
    ------
    ValueError
        If the name is not an interval, or asks for a quality the degree
        cannot take -- there is no major fifth.

    See Also
    --------
    interval_names : the same, backwards.
    consonance : how the article classifies it.
    midi_to_hz_interval : turns these semitones into a frequency ratio.

    Examples
    --------
    >>> interval("M3"), interval("P5"), interval("m7")
    (4, 7, 10)
    >>> interval("M9"), interval("P11"), interval("m16")
    (14, 17, 25)
    >>> interval("aug4") == interval("dim5") == interval("TT")
    True
    >>> interval("aug3")  # a major third raised by a semitone
    5
    """
    if name in SIMPLE_INTERVALS:
        return SIMPLE_INTERVALS[name]

    match = _NOTATION.match(name.strip())
    if not match:
        raise ValueError(
            f'{name!r} is not an interval; write a quality and a degree, '
            f'as in "P5", "m3", "aug4" or "M9"')
    quality = match.group('quality')
    degree = int(match.group('degree'))
    if degree < 1:
        raise ValueError(f'{name!r} has no degree; they start at 1 (unison)')

    simple, octaves = _simple(degree)
    perfect = simple in _PERFECT_DEGREES

    if quality == 'P':
        if not perfect:
            raise ValueError(
                f'{name!r}: a {simple} is major or minor, not perfect. The '
                f'perfect degrees are {_PERFECT_DEGREES}')
        base = SIMPLE_INTERVALS[f'P{simple}']
    elif quality in ('M', 'm'):
        if perfect:
            raise ValueError(
                f'{name!r}: a {simple} is perfect, not major or minor')
        base = SIMPLE_INTERVALS[f'{quality}{simple}']
    else:
        # "A perfect interval, or a major interval, increased by one
        # semitone results in an augmented interval"; a perfect or a minor
        # one decreased by a semitone is diminished.
        widened = quality in ('aug', 'A')
        if perfect:
            reference = SIMPLE_INTERVALS[f'P{simple}']
        else:
            reference = SIMPLE_INTERVALS[f'{"M" if widened else "m"}{simple}']
        base = reference + (1 if widened else -1)

    return base + 12 * octaves


def interval_names(semitones: int) -> tuple[str, ...]:
    """Every traditional name for an interval of this many semitones.

    Parameters
    ----------
    semitones : integer
        Semitones above the lower note, from 0 upwards.

    Returns
    -------
    tuple of strings
        One name for most intervals and three for the tritone, which the
        article names ``aug4``, ``dim5`` and ``TT``. Compound intervals
        are named as the simple one raised by octaves.

    Raises
    ------
    ValueError
        If `semitones` is negative. An interval is measured upward; use
        the two notes the other way round.

    See Also
    --------
    interval : the same, forwards.

    Examples
    --------
    >>> interval_names(4)
    ('M3',)
    >>> interval_names(6)
    ('aug4', 'dim5', 'TT')
    >>> interval_names(14)
    ('M9',)
    """
    if semitones < 0:
        raise ValueError(
            f'an interval is measured upward from the lower note; got '
            f'{semitones}. Swap the notes and name the interval between '
            f'them instead')
    octaves, within = divmod(semitones, 12)
    if within == 0 and octaves:
        within, octaves = 12, octaves - 1

    names = [name for name, count in SIMPLE_INTERVALS.items()
             if count == within and name not in ('tri',)]
    if not octaves:
        return tuple(names)

    compound = []
    for name in names:
        match = _NOTATION.match(name)
        if not match:                      # TT carries no degree to raise
            continue
        degree = int(match.group('degree')) + 7 * octaves
        compound.append(f"{match.group('quality')}{degree}")
    return tuple(compound)


def consonance(interval_or_semitones: str | int) -> str:
    """How the article classifies an interval.

    Parameters
    ----------
    interval_or_semitones : string or integer
        An interval name, or its semitone count.

    Returns
    -------
    string
        One of ``"perfect consonance"``, ``"imperfect consonance"``,
        ``"weak dissonance"``, ``"strong dissonance"``, or
        ``"context dependent"`` for the two the article declines to settle:
        the perfect fourth, "a perfect consonance when considered as an
        inversion of the perfect fifth and a dissonance or an imperfect
        consonance otherwise", and the tritone, "consonant in some
        cultures".

    Raises
    ------
    ValueError
        If the interval is negative, or its name is not one.

    See Also
    --------
    CONSONANCE : the classification, as a table.

    Examples
    --------
    >>> consonance("P5")
    'perfect consonance'
    >>> consonance("m2")
    'strong dissonance'
    >>> consonance("P4"), consonance("TT")
    ('context dependent', 'context dependent')

    Notes
    -----
    A compound interval "is classified in terms of the simple interval
    between the same notes but in the same octave", so a major ninth is
    classified as the major second it is an octave above.
    """
    semitones = (interval(interval_or_semitones)
                 if isinstance(interval_or_semitones, str)
                 else int(interval_or_semitones))
    if semitones < 0:
        raise ValueError(f'an interval is not negative; got {semitones}')
    octaves, within = divmod(semitones, 12)
    if within == 0 and octaves:
        within = 12
    return CONSONANCE[within]


def interval_between(lower: float, upper: float) -> int:
    """The interval in semitones between two frequencies, rounded.

    Parameters
    ----------
    lower, upper : scalar
        Frequencies in Hertz. `upper` is expected above `lower`.

    Returns
    -------
    integer
        Semitones, to the nearest one -- the frequencies of real notes
        are not exactly tempered, and an interval is a name for a
        neighbourhood rather than for one ratio.

    Raises
    ------
    ValueError
        If either frequency is not positive, or `upper` is below `lower`.

    See Also
    --------
    interval_names : names the result.
    hz_to_midi : the same conversion, against a fixed reference.

    Examples
    --------
    >>> interval_between(220.0, 330.0)   # a fifth, near enough
    7
    >>> interval_names(interval_between(220.0, 440.0))
    ('P8',)
    """
    if lower <= 0 or upper <= 0:
        raise ValueError(
            f'frequencies must be positive; got {lower} and {upper}')
    if upper < lower:
        raise ValueError(
            f'{upper} is below {lower}; an interval is measured upward, so '
            f'give the lower frequency first')
    return round(12 * math.log2(upper / lower))
