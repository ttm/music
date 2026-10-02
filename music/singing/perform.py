# -*- coding: utf-8 -*-
"""Singing a lyric to a melody: :func:`sing`, and the score and
configuration it writes for the eCantorix backend."""

import re
import logging
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from numbers import Real
import soundfile as sf
from music.core import normalize_mono, normalize_stereo
from .psola import syllables
from .paths import (ENGINE_MARKER, cache_dir, engine_dir, is_engine,
                    require_system_dependencies)


# def sing(text="ba-na-nin-ha pra vo-cê",
#: What an espeak voice name is made of: a language such as ``en`` or
#: ``pt-br``, and a variant after ``+``. ``lang`` is written into the
#: configuration the engine runs as Perl, so nothing else may reach it.
_VOICE = re.compile(r'[A-Za-z0-9_-]+(\+[A-Za-z0-9_-]+)?')

#: The singers :func:`sing` can use, the default first.
BACKENDS = ('psola', 'ecantorix')

#: The voices from the engine's extras that ``effect`` may name, and the
#: file each loads. ``flint`` is an earlier spelling of ``flite``.
EFFECTS = {'flite': 'flite', 'flint': 'flite', 'tremolo': 'tremolo',
           'melt': 'melt'}


def sing(text="Mar-ry had a litt-le lamb",
         notes=(4, 2, 0, 2, 4, 4, 4), durs=(1, 1, 1, 1, 1, 1, 2),
         M='4/4', L='1/4', Q=120, K='C', reference=60,
         lang='en', transpose=-12, effect=None, backend='psola'):
    """Sing a line of text to a melody.

    By default espeak-ng says each syllable, and Praat's PSOLA holds it at
    its note's pitch and length: see :mod:`music.singing.psola`. It needs
    espeak-ng and ``pip install 'music[singing]'``. With
    ``backend="ecantorix"``, the melody is written as ABC notation, the
    eCantorix engine renders it through espeak, and the result is read
    back as samples; that needs the engine, which
    :func:`~music.singing.setup_engine` clones, and Perl, abc2midi and sox.
    Both sing MIDI ``reference + note + transpose`` for the same lengths,
    so the two can be compared.

    Parameters
    ----------
    text : str
        The lyric, one syllable per note: syllables within a word joined
        by hyphens, words separated by spaces. A syllable espeak says
        without a vowel is sung with a schwa after it, as a singer sings
        the e that spoken French leaves silent: alone, the "ques" of
        "Jac-ques" is a bare /k/, and its note had nothing to sing.
    notes : sequence of int
        Each note's pitch, in semitones above ``reference``.
    durs : sequence
        Each note's duration, in units of ``L``: a number such as 2 or
        0.5, a string of ABC such as ``"3/2"``, or a negative number
        ``-n`` for ``1/n``, which this has always accepted.
    M, L, Q, K : str or int
        The ABC meter, unit note length, tempo and key. A number for ``Q``
        is units of ``L`` a minute, as ABC reads a bare ``Q:``; see
        :func:`unit_seconds`. Every note is written with its accidental,
        so the key changes none of them; E and B were written bare, and
        a key with flats sang them a semitone low. abc2midi accents the
        beats, and eCantorix sings the accented notes louder, by up to
        2.3 dB. The psola backend reads ``L`` and ``Q``, has no use for
        the meter or the key, and sings every note at one level.
    reference : int
        The MIDI note that pitch zero refers to, which is the note the
        score is written at.
    lang : str
        The espeak voice, which sets the language the text is sung in.
    transpose : int
        Semitones added to every note as it is sung: the engine sings
        MIDI ``reference + note + transpose``. The default, -12, sings an
        octave below the score, where every note used to be sung.
    effect : str or None
        One of the eCantorix engine's extra voices, which both backends
        sing: ``"tremolo"``, a tremolo of 9 Hz and a reverberation, in
        stereo; ``"melt"``, the same with espeak's female voice, its
        formants raised a quarter, and sinking with the pitch below
        216 Hz, as tape played slower; and ``"flite"`` (also accepted as
        ``"flint"``, its earlier
        spelling here), which says each syllable with the flite
        synthesizer instead of espeak, so it needs ``flite`` installed,
        and ``bc`` too for eCantorix, and ``lang`` names one of flite's
        voices, such as ``"rms"`` or ``"slt"``. None sings with the plain
        voice. The psola backend makes them with the package's own
        tremolo, reverberation and resampling, set to what eCantorix's
        sox gives.
    backend : {'psola', 'ecantorix'}
        Which singer. ``"psola"``, the default, needs espeak-ng and
        ``pip install 'music[singing]'``. ``"ecantorix"`` is the engine
        this package sang with by default until PSOLA, kept as the
        reference it is compared with.

    Returns
    -------
    ndarray
        The sung line, normalized, at 44,100 Hz: mono, or ``(2, nsamples)``
        for the ``tremolo`` and ``melt`` effects, which render in stereo
        and ring a second past the line. eCantorix's used to come back as
        ``(nsamples, 2)``, as the file holds them, and be normalized as
        one channel.

    Raises
    ------
    RuntimeError
        If espeak-ng or praat-parselmouth is missing, or flite for its
        effect; or, with ``backend="ecantorix"``, if the engine is not
        installed (run :func:`music.singing.setup_engine`), cannot be
        built in the cache, or renders at a rate other than 44,100 Hz, or
        the flite effect is asked for without ``bc``.
    ValueError
        If ``backend`` or ``effect`` is not one of those above, ``lang`` is
        not a voice name, or not one of flite's voices for its effect,
        ``transpose`` is not a finite number, ``L`` or ``Q`` is not
        a length or tempo ABC can read (see :func:`unit_seconds`), there is
        not exactly
        one duration per note, a duration is zero, or a note falls outside
        MIDI 12 to 96. These are checked first, so a wrong one is reported
        whether or not the engine is installed. The voice and the
        transposition are written into a configuration the engine runs as
        Perl, so anything else there would be run as code.

    Notes
    -----
    Measured, a note of 0 at the defaults sings at 130.8 Hz, MIDI 48, and
    at ``transpose=0`` at 261.6 Hz, middle C; eCantorix sings them at
    130.3 Hz and 262.5 Hz.

    Three things kept these parameters from reaching the engine. It loads
    the configuration with Perl's ``do "achant.conf"``, which since Perl
    5.26 no longer looks in the current directory, so ``lang``,
    ``transpose`` and ``effect`` were never read and it sang at its own
    -24. The score was written an octave above ``reference``, MIDI 60 as
    ABC's ``c``. And the effects load files from the engine's examples,
    which were not in the cache. Together, every note was sung at
    ``reference + note - 12`` in the default voice, whatever was asked;
    the default transposition, -12, keeps that.
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"backend must be one of {BACKENDS}; got {backend!r}")
    if effect and effect not in EFFECTS:
        raise ValueError(
            f"effect not understood: {effect!r}; expected one of "
            f"{sorted(EFFECTS)}, or None for the plain voice")
    if not (isinstance(lang, str) and _VOICE.fullmatch(lang)):
        raise ValueError(
            f"lang must be an espeak voice name, such as 'en' or 'pt-br'; "
            f"got {lang!r}")
    if (isinstance(transpose, bool) or not isinstance(transpose, Real)
            or not math.isfinite(transpose)):
        raise ValueError(
            f"transpose must be a finite number of semitones; got "
            f"{transpose!r}")
    # The tempo is read here for both backends: eCantorix handed a Q of
    # 0 to abc2midi, which called it malformed and stopped the build.
    unit_seconds(L, Q)
    if backend == 'psola':
        from . import psola
        # The notes the score could be written with, as eCantorix's are,
        # so that the two backends refuse the same melodies.
        translate_to_abc(notes, durs, reference)
        return psola.sing(text, notes, durs, L=L, Q=Q, reference=reference,
                          lang=lang, transpose=transpose,
                          effect=EFFECTS[effect] if effect else None)
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
    if effect and EFFECTS[effect] == 'flite':
        _require_flite(lang)
    # Inside the engine, which is there: is_engine said so.
    cache.mkdir(exist_ok=True)

    write_abc(text, notes, durs, M=M, L=L, Q=Q, K=K, reference=reference)
    conf_text = '$ESPEAK_VOICE = "{}";\n'.format(lang)
    conf_text += '$ESPEAK_TRANSPOSE = {};'.format(transpose)
    if effect:
        conf_text += f"\ndo 'extravoices/{EFFECTS[effect]}.inc';"
    if not (effect and EFFECTS[effect] == 'flite'):
        # flite reads no espeak phonemes, and lang names a flite voice.
        sung = sung_phonemes('espeak', lang, syllables(text))
        if sung:
            conf_text += '\n' + _edit_syllables(sung)
    with open(cache / 'achant.conf', 'w') as f:
        f.write(conf_text)
    try:
        _prepare_cache(engine, cache)
    except OSError as exc:
        raise RuntimeError(f'Failed to prepare singing cache: {exc}') from exc
    # A render left from an earlier call would otherwise be read back as
    # this one's when the engine writes nothing.
    rendered = cache / 'achant.wav'
    rendered.unlink(missing_ok=True)
    # The script is run with the perl the requirements were checked
    # against, rather than by its #!/usr/bin/perl, which is another Perl
    # on macOS and one without the modules; and with the cache on @INC,
    # where its `do "achant.conf"` looks since Perl 5.26 took the current
    # directory off it.
    try:
        result = subprocess.run(
            ['make', '-C', str(cache), 'ECANTORIX=perl -I. ../ecantorix.pl'],
            check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f'Failed to build singing cache: {exc}\n'
            f'{_tail(exc.stdout, exc.stderr)}') from exc
    if not rendered.is_file():
        # The Makefile pipes the script through tee, so make succeeds
        # when the script fails.
        raise RuntimeError(
            'the singing engine ran but wrote no achant.wav; what it '
            f'said:\n{_tail(result.stdout, result.stderr)}')

    samples, sample_rate = sf.read(str(rendered), dtype='float64')
    if sample_rate != 44100:
        raise RuntimeError(
            f'expected the engine to render at 44100 Hz, got {sample_rate}'
        )
    if samples.ndim == 2:
        return normalize_stereo(samples.T)
    return normalize_mono(samples)


def _require_flite(lang):
    """flite and bc, which the flite effect runs, and a voice flite has.

    The effect has the engine say each syllable with flite rather than
    espeak, in the voice ``lang`` names, so ``lang`` is one of flite's
    voices -- ``rms``, ``slt`` -- not a language.
    """
    missing = [name for name in ('flite', 'bc') if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            f"the flite effect runs {' and '.join(missing)}, which "
            f"{'is' if len(missing) == 1 else 'are'} not installed. On "
            f"Debian or Ubuntu: sudo apt install {' '.join(missing)}. On "
            "macOS with Homebrew: brew install flite.")
    from .psola import _require_flite_voice

    _require_flite_voice(lang)


def _prepare_cache(engine, cache):
    """Put in the cache what the engine reads from it when it sings.

    Its Makefile; the effects' files, which the configuration loads by a
    path relative to the cache; and, for the effects that give espeak a
    voice of their own, a copy of espeak's data with those voices in it.
    The engine's Makefile makes that copy from a Linux path,
    ``/usr/lib/x86_64-linux-gnu/espeak-data``, and does not try again
    once the directory exists, so this makes it from wherever espeak says
    its data is.
    """
    shutil.copy(engine / 'Makefile', cache / 'Makefile')
    voices = engine / 'examples' / 'extravoices'
    if voices.is_dir():
        shutil.copytree(voices, cache / 'extravoices', dirs_exist_ok=True)
    data = cache / 'espeak-data'
    source = _espeak_data()
    if source is not None and not (data / 'phontab').is_file():
        shutil.copytree(source, data, dirs_exist_ok=True)
    if voices.is_dir() and data.is_dir():
        shutil.copytree(voices, data / 'voices' / '!v', dirs_exist_ok=True)


def _espeak_data():
    """Where espeak keeps its data, as its version line says, or None."""
    result = subprocess.run(['espeak', '--version'], capture_output=True,
                            text=True)
    found = re.search(r'Data at: (.+)', result.stdout)
    return Path(found.group(1).strip()) if found else None


def _tail(*outputs, lines=20):
    """The last lines of what a command printed, to put in an error."""
    text = '\n'.join(output for output in outputs if output)
    return '\n'.join(text.strip().splitlines()[-lines:])


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
    '=C=D/2=E3/2'

    """
    if len(notes) != len(durs):
        raise ValueError(
            f"got {len(notes)} notes and {len(durs)} durations; "
            f"there must be exactly one duration per note")
    durs = [_abc_length(i) for i in durs]
    notes = converter.convert(notes, reference)
    return ''.join([i + j for i, j in zip(notes, durs)])


#: What a syllable is sung on: a vowel of the IPA, as espeak writes them,
#: or the mark of a consonant that makes a syllable of itself.
_NUCLEUS = re.compile('[aeiouyæøœɐɑɒɔəɘɛɜɞɤɪɨɯɵɶʉʊʌʏᵻɚɝ̩]')


def sung_phonemes(program, voice, syllables):
    """The phonemes to sing each syllable espeak says without a vowel.

    espeak says a syllable as it is spoken, and spoken French leaves the e
    of "Jacques" silent: "ques" alone is a bare /k/, 41 ms long, with
    nothing to hold a note on. A singer gives that e its note, as a
    schwa, and so do both backends of :func:`sing`: the phonemes espeak
    says, then a schwa.

    Parameters
    ----------
    program : str
        The espeak that will sing them.
    voice : str
        The espeak voice, as :func:`sing`'s ``lang``.
    syllables : iterable of str
        The syllables, each as the singer will be handed it.

    Returns
    -------
    dict
        Each syllable espeak says without a vowel, mapped to espeak's
        phoneme input for it with a schwa after, such as ``"[[k@]]"``. A
        syllable said with a vowel, or said as nothing, is left out; so is
        every one if `program` cannot be run or speak in `voice`.
    """
    sung = {}
    for syllable in dict.fromkeys(syllables):
        ipa = _transcribe(program, voice, syllable, '--ipa')
        if not ipa or _NUCLEUS.search(ipa):
            continue
        phonemes = _transcribe(program, voice, syllable, '-x')
        if phonemes:
            sung[syllable] = f'[[{phonemes}@]]'
    return sung


def _transcribe(program, voice, text, notation):
    """`text` as `program` says it in `voice`, written in `notation`.

    ``"--ipa"`` for the IPA, ``"-x"`` for espeak's phoneme mnemonics;
    None where `program` fails, as for a voice it does not have, or
    cannot be run. The lyric is then sung as written, and the singer
    says what is missing.
    """
    try:
        result = subprocess.run([program, '-q', notation, '-v', voice,
                                 text], capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode:
        return None
    return ''.join(result.stdout.split())


def _edit_syllables(sung):
    """Perl that has eCantorix sing each syllable of `sung` as mapped.

    The engine hands ``$EDIT_SYLLABLES`` each syllable as abc2midi wrote
    it, after a space or a line break, with its punctuation; and it sings
    the part of a syllable in ``[[ ]]``, showing the rest. Each syllable
    is written as a Perl string and only ever read as one.
    """
    pairs = ', '.join(f'{_perl_string(said)} => {_perl_string(phonemes)}'
                      for said, phonemes in sung.items())
    return ('my %sung = (' + pairs + ');\n'
            '$EDIT_SYLLABLES = sub { (my $said = $_) =~ s/^\\s+|\\s+$//g; '
            '$_ .= $sung{$said} if exists $sung{$said}; };')


def _perl_string(text):
    """`text` as a single-quoted Perl string, in which only ``\\\\`` and
    ``\\'`` are read as anything but themselves."""
    return "'" + text.replace('\\', '\\\\').replace("'", "\\'") + "'"


def _note_length(duration):
    """A note's duration in units of ``L``, as a fraction.

    A number is that many units, and a negative one ``-n`` is ``1/n``, the
    convention the ``-`` to ``/`` replacement gave it. A string is ABC's
    length notation, with ``-`` read as ``/``: ``"3/2"``, ``"/2"`` for a
    half, ``"3/"`` for three halves and ``"//"`` for a quarter.

    Raises
    ------
    ValueError
        If the duration is zero, or a string ABC cannot read as a length.
    """
    if isinstance(duration, Real):
        if duration == 0:
            raise ValueError(
                'a note cannot last no time; got a duration of 0')
        length = Fraction(float(duration)).limit_denominator(1000)
        return 1 / -length if length < 0 else length
    text = str(duration).replace('-', '/')
    written = _ABC_LENGTH.fullmatch(text)
    if not written:
        raise ValueError(
            f'{duration!r} is not a length ABC can read, such as "3/2" or '
            f'"/2"')
    numerator, slashes, denominator = written.groups()
    if not slashes:
        length = Fraction(int(numerator))
    else:
        halves = 2 ** len(slashes) if not denominator else int(denominator)
        length = Fraction(int(numerator or 1), halves)
    if length == 0:
        raise ValueError(
            f'a note cannot last no time; got a duration of {duration!r}')
    return length


#: ABC's lengths: a whole number, a fraction, or slashes that halve.
_ABC_LENGTH = re.compile(r'(\d*)(/*)(\d*)')


def unit_seconds(L='1/4', Q=120):
    """How long one unit of ``L`` lasts, in seconds, at the tempo ``Q``.

    Parameters
    ----------
    L : str
        The unit note length, as ABC writes it: ``"1/4"`` for a quarter.
    Q : int or str
        The tempo. A number is that many units of ``L`` a minute, which is
        how ABC reads a bare ``Q:``; ``"1/4=120"`` is 120 quarters a minute
        whatever ``L`` is.

    Returns
    -------
    Fraction
        Seconds per unit.

    Raises
    ------
    ValueError
        If either is not what ABC writes there, or the tempo is not
        positive.

    Examples
    --------
    >>> float(unit_seconds('1/4', 120)), float(unit_seconds('1/8', '1/4=120'))
    (0.5, 0.25)
    """
    unit = _abc_fraction(L, 'L')
    if isinstance(Q, Real) and not isinstance(Q, bool):
        beat, per_minute = unit, Fraction(float(Q)).limit_denominator(1000)
    else:
        beat_text, equals, rate = str(Q).partition('=')
        if not equals:
            raise ValueError(
                f'Q must be a number or "beat=count", such as "1/4=120"; '
                f'got {Q!r}')
        beat = _abc_fraction(beat_text, 'the beat in Q')
        per_minute = _abc_fraction(rate, 'the count in Q')
    if per_minute <= 0:
        raise ValueError(f'the tempo must be positive; got Q={Q!r}')
    return Fraction(60) / per_minute * unit / beat


def _abc_fraction(text, name):
    """A positive fraction such as ``1/4``, for `name`, or an error."""
    try:
        value = Fraction(str(text).strip())
    except (ValueError, ZeroDivisionError):
        raise ValueError(
            f'{name} must be a fraction such as "1/4"; got {text!r}') from None
    if value <= 0:
        raise ValueError(f'{name} must be positive; got {text!r}')
    return value


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
    length = _note_length(duration)
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
        # Every name has its accidental, the natural sign included: an
        # unmarked E or B took the key's flat, so K="F" sang B as B flat.
        notes = re.findall(r'[\^=]?[a-g]', '=c^c=d^d=e=f^f=g^g=a^a=b')
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
