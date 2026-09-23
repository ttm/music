# -*- coding: utf-8 -*-
"""Utilities to synthesize singing from text using eCantorix."""

import re
import logging
import shutil
import subprocess
import soundfile as sf
from music.core import normalize_mono
from .paths import (ENGINE_MARKER, cache_dir, engine_dir, is_engine,
                    require_system_dependencies)


# def sing(text="ba-na-nin-ha pra vo-cê",
def sing(text="Mar-ry had a litt-le lamb",
         notes=(4, 2, 0, 2, 4, 4, 4), durs=(1, 1, 1, 1, 1, 1, 2),
         M='4/4', L='1/4', Q=120, K='C', reference=60,
         lang='en', transpose=-36, effect=None):
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
        Each note's duration, in units of ``L``.
    M, L, Q, K : str or int
        The ABC meter, unit note length, tempo in beats per minute and key.
    reference : int
        The MIDI note that pitch zero refers to.
    lang : str
        The espeak voice, which sets the language the text is sung in.
    transpose : int
        The espeak transposition, in semitones.
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
        duration per note, or a note falls outside MIDI 12 to 96.
    """
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
    cache.mkdir(parents=True, exist_ok=True)

    write_abc(text, notes, durs, M=M, L=L, Q=Q, K=K, reference=reference)
    conf_text = '$ESPEAK_VOICE = "{}";\n'.format(lang)
    conf_text += '$ESPEAK_TRANSPOSE = {};'.format(transpose)
    if effect in ('flite', 'flint'):
        conf_text += "\ndo 'extravoices/flite.inc';"
    elif effect == 'tremolo':
        conf_text += "\ndo 'extravoices/tremolo.inc';"
    elif effect == 'melt':
        conf_text += "\ndo 'extravoices/melt.inc';"
    elif effect:
        raise ValueError('effect not understood')
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
        One duration per note.
    reference : int
        The MIDI note that offset zero refers to.

    Returns
    -------
    str
        The notes with their durations, ready to append to an ABC header.

    Raises
    ------
    ValueError
        If there is not exactly one duration per note. Zipping them
        silently discarded the tail of whichever was longer, so five
        notes with three durations produced a three-note score -- and
        ``write_abc`` appends the lyric line separately, which then
        pointed at notes that were no longer there.

    Examples
    --------
    >>> translate_to_abc([0, 2, 4], [1, 1, 1], reference=60)
    '=c=de'

    """
    if len(notes) != len(durs):
        raise ValueError(
            f"got {len(notes)} notes and {len(durs)} durations; "
            f"there must be exactly one duration per note")
    durs = [str(i).replace('-', '/') for i in durs]
    durs = [i if i != '1' else '' for i in durs]
    notes = converter.convert(notes, reference)
    return ''.join([i + j for i, j in zip(notes, durs, strict=True)])


class Notes:
    """The ABC name of each MIDI note from 12 to 96."""

    def __init__(self):
        self.notes_dict = None
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
        notes_all = notes____ + notes___ + notes__ + notes_ + notes + \
            notes_u + notes__u + notes___u
        # notes_all spans eight octaves, 96 names. The dictionary covers
        # MIDI 12 to 96, which is 85 of them; the remaining eleven are
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
