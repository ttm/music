# -*- coding: utf-8 -*-
"""Utilities to synthesize singing from text using eCantorix."""

import re
import logging
import shutil
import subprocess
from fractions import Fraction
from numbers import Real
import soundfile as sf
from music.core import normalize_mono
from .paths import (ENGINE_MARKER, cache_dir, engine_dir, is_engine,
                    require_system_dependencies)


# def sing(text="ba-na-nin-ha pra vo-cê",
#: The voices from the engine's extras that ``effect`` may name, and the
#: file each loads. ``flint`` is an earlier spelling of ``flite``.
EFFECTS = {'flite': 'flite', 'flint': 'flite', 'tremolo': 'tremolo',
           'melt': 'melt'}


def sing(text="Mar-ry had a litt-le lamb",
         notes=(4, 2, 0, 2, 4, 4, 4), durs=(1, 1, 1, 1, 1, 1, 2),
         M='4/4', L='1/4', Q=120, K='C', reference=60,
         lang='en', transpose=-24, effect=None):
    """Sing a line of text to a melody, with the eCantorix engine.

    The melody is written as ABC notation, the engine renders it through
    espeak, and the result is read back as samples.

    Parameters
    ----------
    text : str
        The lyric, one syllable per note: syllables within a word joined
        by hyphens, words separated by spaces.
    notes : sequence of int
        Each note's pitch, in semitones above ``reference``.
    durs : sequence
        Each note's duration, in units of ``L``: a number such as 2 or
        0.5, a string of ABC such as ``"3/2"``, or a negative number
        ``-n`` for ``1/n``, which this has always accepted.
    M, L, Q, K : str or int
        The ABC meter, unit note length, tempo in beats per minute and key.
    reference : int
        The MIDI note that pitch zero refers to, which is the note the
        score is written at.
    lang : str
        The espeak voice, which sets the language the text is sung in.
    transpose : int
        Semitones added to every note as it is sung: the engine sings
        MIDI ``reference + note + transpose``. The default, -24, is
        eCantorix's own, and sings two octaves below the score.
    effect : str or None
        A voice from the engine's extras: ``"flite"`` (also accepted as
        ``"flint"``, its earlier spelling here), ``"tremolo"`` or
        ``"melt"``. None sings with the plain voice.

    Returns
    -------
    ndarray
        The sung line, normalized, at 44,100 Hz.

    Raises
    ------
    RuntimeError
        If the engine is not installed (run
        :func:`music.singing.setup_engine`), cannot be built in the cache,
        or renders at a rate other than 44,100 Hz.
    ValueError
        If ``effect`` is not one of those above, there is not exactly one
        duration per note, a duration is zero, or a note falls outside
        MIDI 12 to 96. The effect is checked first, so a wrong one is
        reported whether or not the engine is installed.

    Notes
    -----
    The score used to be written an octave above ``reference`` -- MIDI 60
    as ABC's ``c``, which is 72 -- and the default transposition was -36
    to sing where eCantorix's -24 would have. The default renders as it
    did; a call that passed ``transpose`` sings an octave lower than it
    used to, and adding 12 to it restores that.
    """
    if effect and effect not in EFFECTS:
        raise ValueError(
            f"effect not understood: {effect!r}; expected one of "
            f"{sorted(EFFECTS)}, or None for the plain voice")
    engine = engine_dir()
    cache = cache_dir()
    if not is_engine(engine):
        detail = (f"the directory exists but has no {ENGINE_MARKER}"
                  if engine.is_dir() else "nothing is there")
        raise RuntimeError(
            f"no usable eCantorix engine at {engine}: {detail}. "
            "Run music.singing.setup_engine() to install it."
        )
    require_system_dependencies()
    # Inside the engine, which is there: is_engine said so.
    cache.mkdir(exist_ok=True)

    write_abc(text, notes, durs, M=M, L=L, Q=Q, K=K, reference=reference)
    conf_text = '$ESPEAK_VOICE = "{}";\n'.format(lang)
    conf_text += '$ESPEAK_TRANSPOSE = {};'.format(transpose)
    if effect:
        conf_text += f"\ndo 'extravoices/{EFFECTS[effect]}.inc';"
    with open(cache / 'achant.conf', 'w') as f:
        f.write(conf_text)
    try:
        shutil.copy(engine / 'Makefile', cache / 'Makefile')
    except OSError as exc:
        raise RuntimeError(f'Failed to prepare singing cache: {exc}') from exc
    try:
        subprocess.run(['make', '-C', str(cache)], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'Failed to build singing cache: {exc}') from exc

    samples, sample_rate = sf.read(str(cache / 'achant.wav'),
                                   dtype='float64')
    if sample_rate != 44100:
        raise RuntimeError(
            f'expected the engine to render at 44100 Hz, got {sample_rate}'
        )
    return normalize_mono(samples)


def write_abc(text, notes, durs, M='4/4', L='1/4', Q=120, K='C', reference=60):
    """Write the melody and its lyric as ABC notation in the cache.

    The engine reads ``achant.abc`` from the singing cache; the
    parameters are those of :func:`sing`.
    """
    text_ = 'X:1\n'
    text_ += 'T:Some chanting for music python package\n'
    text_ += 'M:{}\n'.format(M)
    text_ += 'L:{}\n'.format(L)
    text_ += 'Q:{}\n'.format(Q)
    text_ += 'V:1\n'
    text_ += 'K:{}\n'.format(K)
    notes = translate_to_abc(notes, durs, reference)
    text_ += notes + "\nw: " + text
    fname = cache_dir() / "achant.abc"
    with open(fname, 'w') as f:
        f.write(text_)


def translate_to_abc(notes, durs, reference):
    """Render pitches and durations as an ABC notation fragment.

    Parameters
    ----------
    notes : sequence of int
        Semitone offsets from ``reference``.
    durs : sequence
        One duration per note, as :func:`sing` takes them.
    reference : int
        The MIDI note that offset zero refers to.

    Returns
    -------
    str
        The notes with their durations, ready to append to an ABC header.

    Raises
    ------
    ValueError
        If there is not exactly one duration per note, or a numeric one
        is zero. Zipping them silently discarded the tail of whichever
        was longer, so five notes with three durations produced a
        three-note score -- and
        ``write_abc`` appends the lyric line separately, which then
        pointed at notes that were no longer there.

    Examples
    --------
    >>> translate_to_abc([0, 2, 4], [1, 0.5, 1.5], reference=60)
    '=C=D/2E3/2'

    """
    if len(notes) != len(durs):
        raise ValueError(
            f"got {len(notes)} notes and {len(durs)} durations; "
            f"there must be exactly one duration per note")
    durs = [_abc_length(i) for i in durs]
    notes = converter.convert(notes, reference)
    return ''.join([i + j for i, j in zip(notes, durs)])


def _abc_length(duration):
    """A duration in units of ``L``, as ABC writes a note's length.

    A number is written as its fraction, which is what ABC takes: 2 as
    ``2``, 0.5 as ``/2`` and 1.5 as ``3/2``, and 1, the unit, as nothing.
    It used to be written as Python prints it, so 0.5 went into the score
    as ``0.5``, which is not ABC. A negative number ``-n`` is ``1/n``, the
    convention the ``-`` to ``/`` replacement gave it. A string is ABC
    already, with ``-`` read as ``/``.
    """
    if not isinstance(duration, Real):
        text = str(duration).replace('-', '/')
        return '' if text == '1' else text
    if duration == 0:
        raise ValueError('a note cannot last no time; got a duration of 0')
    length = Fraction(float(duration)).limit_denominator(1000)
    if length < 0:
        length = 1 / -length
    if length.denominator == 1:
        return '' if length == 1 else str(length.numerator)
    numerator = '' if length.numerator == 1 else str(length.numerator)
    return f'{numerator}/{length.denominator}'


class Notes:
    """The ABC name of each MIDI note from 12 to 96.

    ABC writes middle C, MIDI 60, as ``C``, the octave above in lower
    case and further octaves with apostrophes above and commas below.
    This used to name MIDI 60 ``c``, which is 72, so every score was
    written an octave above the ``reference`` it was given.
    """

    #: Filled by make_dict, which __init__ calls.
    notes_dict: dict[int, str] | None

    def __init__(self):
        self.make_dict()

    def make_dict(self):
        """Build the table from MIDI note number to ABC note name."""
        notes = re.findall(r'[\^=]?[a-g]', '=c^c=d^de=f^f=g^g=a^ab')
        # notes=re.findall(r'[\^]{0,1}[a-g]{1}','a^abc^cd^def^fg^g')
        notes_ = [note.upper() for note in notes]
        notes__ = [note + "," for note in notes_]
        notes___ = [note + "," for note in notes__]
        notes____ = [note + "," for note in notes___]
        notes_u = [note + "'" for note in notes]
        notes__u = [note + "'" for note in notes_u]
        notes___u = [note + "'" for note in notes__u]
        notes_____ = [note + "," for note in notes____]
        # Four commas at MIDI 12, up to three apostrophes: MIDI 60, middle
        # C, is the fifth octave, the upper-case one with no marks.
        notes_all = notes_____ + notes____ + notes___ + notes__ + \
            notes_ + notes + notes_u + notes__u + notes___u
        # notes_all spans nine octaves, 108 names. The dictionary covers
        # MIDI 12 to 96, which is 85 of them; the remaining 23 are
        # deliberately unused. Sliced explicitly so that is a decision
        # rather than something zip does quietly.
        self.notes_dict = dict(zip(range(12, 97), notes_all[:85],
                                   strict=True))

    def convert(self, notes, reference):
        """Name each note, given in semitones above ``reference``.

        Raises
        ------
        ValueError
            If a note falls outside MIDI 12 to 96, which the table
            covers. It raised a bare KeyError naming only the number.
        """
        if self.notes_dict is None:
            self.make_dict()
        assert self.notes_dict is not None  # make_dict always assigns it
        notes_ = [reference + note for note in notes]
        outside = [note for note in notes_ if note not in self.notes_dict]
        if outside:
            raise ValueError(
                f"MIDI notes {outside} are outside the 12 to 96 that ABC "
                f"names here; move them, or change reference={reference}")
        return [self.notes_dict[note] for note in notes_]


converter = Notes()

if __name__ == '__main__':  # pragma: no cover - a manual smoke run
    narray = sing()
    logging.info("finished")
