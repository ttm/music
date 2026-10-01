"""Singing by speech synthesis and PSOLA: the default backend of ``sing``.

espeak-ng says each syllable of the lyric, and Praat's pitch-synchronous
overlap-add, through praat-parselmouth, holds it at its note's pitch and
stretches its voiced part to the note's length. The notes are then joined
end to end. That is eCantorix's idea -- a speech synthesizer made to sing
one syllable at a time -- without its toolchain: no Perl and its modules,
no round trip through ABC and MIDI, no sox.

It sings the score :func:`~music.singing.perform.sing` would hand
eCantorix, at the same pitches, MIDI ``reference + note + transpose``, and
for the same lengths, so the two can be compared; ``tools/compare_singing.py``
does. It needs espeak-ng, or the older espeak, and
``pip install 'music[singing]'`` for praat-parselmouth.

References
----------
.. [1] Moulines, E., and Charpentier, F. "Pitch-synchronous waveform
       processing techniques for text-to-speech synthesis using diphones."
       Speech Communication 9 (1990): 453-467.
.. [2] Jadoul, Y., Thompson, B., and de Boer, B. "Introducing Parselmouth:
       A Python interface to Praat." Journal of Phonetics 71 (2018): 1-15.
"""
from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

#: The programs that can say a syllable, in the order they are preferred.
SPEAKERS = ("espeak-ng", "espeak")

#: The rate a sung line is rendered at, as eCantorix renders it.
RATE = 44100

#: The range, in Hertz, a spoken syllable's pitch is looked for in.
PITCH_FLOOR, PITCH_CEILING = 60, 600

#: The longest a syllable can be said in, in seconds, and still be too
#: short to have a pitch: Praat looks for one over three periods of
#: :data:`PITCH_FLOOR`. espeak says French "ques" as a /k/ 41 ms long.
#: A syllable this short has no vowel to hold, and is sung as it was said.
SHORTEST = 3 / PITCH_FLOOR

#: How long each note fades in and out, in seconds, so notes meet
#: without a click. A note shorter than four of these fades over a
#: quarter of itself.
FADE = 0.005


def speaker() -> str | None:
    """The first of :data:`SPEAKERS` on PATH, or None."""
    for name in SPEAKERS:
        found = shutil.which(name)
        if found:
            return found
    return None


def missing_requirements() -> list[str]:
    """What this backend needs and cannot find: a speaker, and Parselmouth.

    Returns
    -------
    list of str
        ``"espeak-ng"`` when neither of :data:`SPEAKERS` is on PATH, and
        ``"praat-parselmouth"`` when it cannot be imported.
    """
    missing = []
    if speaker() is None:
        missing.append("espeak-ng")
    if importlib.util.find_spec("parselmouth") is None:
        missing.append("praat-parselmouth")
    return missing


def require() -> None:
    """Raise RuntimeError naming what this backend needs and lacks."""
    missing = missing_requirements()
    if missing:
        hints = []
        if "espeak-ng" in missing:
            hints.append("espeak-ng: sudo apt install espeak-ng, or brew "
                         "install espeak-ng")
        if "praat-parselmouth" in missing:
            hints.append("praat-parselmouth: pip install 'music[singing]'")
        raise RuntimeError(
            "the psola singing backend needs "
            f"{' and '.join(missing)}, which are not installed. "
            + "; ".join(hints) + ".")


def syllables(text: str) -> list[str]:
    """The syllables of a lyric, as ABC's ``w:`` line reads it.

    Words are split at spaces, and at hyphens within a word.

    Examples
    --------
    >>> syllables("Mar-ry had a litt-le lamb")
    ['Mar', 'ry', 'had', 'a', 'litt', 'le', 'lamb']
    """
    return [part for word in text.split() for part in word.split("-")
            if part]


def sing(text, notes, durs, L="1/4", Q=120, reference=60, lang="en",
         transpose=-12) -> NDArray[np.float64]:
    """Sing a line of text to a melody, one syllable a note.

    The parameters are those of :func:`~music.singing.perform.sing`, which
    checks them and calls this for ``backend="psola"``.

    Returns
    -------
    ndarray
        The sung line, mono, at :data:`RATE`, scaled to a peak of 1: every
        note exactly as long as its duration says. It is scaled rather
        than normalized, so a rest stays silent: taking out the mean would
        lift it off zero.

    Raises
    ------
    RuntimeError
        If espeak-ng or praat-parselmouth is missing, or the speaker
        cannot say a syllable in ``lang``.
    ValueError
        If there is not one syllable of ``text`` and one duration for
        each note.
    """
    from .perform import _note_length, sung_phonemes, unit_seconds

    require()
    words = syllables(text)
    if not len(words) == len(notes) == len(durs):
        raise ValueError(
            f"got {len(words)} syllables, {len(notes)} notes and "
            f"{len(durs)} durations; there must be one syllable and one "
            f"duration for each note")
    unit = unit_seconds(L, Q)
    lengths = [float(_note_length(duration) * unit) for duration in durs]
    edges = np.round(np.cumsum([0.0] + lengths) * RATE).astype(np.int64)
    line = np.zeros(int(edges[-1]))
    program = speaker()
    # A syllable said without a vowel is sung with a schwa.
    sung = sung_phonemes(program, lang,
                         [said for said in map(_clean, words) if said])
    with tempfile.TemporaryDirectory() as scratch:
        for index, (word, note) in enumerate(zip(words, notes)):
            start, end = int(edges[index]), int(edges[index + 1])
            said = _clean(word)
            if not said or end == start:
                continue  # a syllable of punctuation alone is a rest
            spoken = _speak(program, sung.get(said, said), lang,
                            Path(scratch) / f"{index}.wav")
            frequency = 440 * 2 ** ((reference + note + transpose - 69) / 12)
            line[start:end] = _fit(_sung(spoken, frequency, (end - start)
                                         / RATE), end - start)
    peak = np.abs(line).max() if len(line) else 0.0
    return line / peak if peak else line


def _clean(word: str) -> str:
    """A syllable with its punctuation taken off, as it is said."""
    return re.sub(r"[^\w']", "", word)


def _speak(program, syllable, voice, path):
    """`syllable` said by `program` in `voice`, trimmed of its silence."""
    import parselmouth

    result = subprocess.run(
        [program, "-v", voice, "-w", str(path), syllable],
        capture_output=True, text=True)
    if result.returncode or not path.is_file():
        raise RuntimeError(
            f"{Path(program).name} could not say {syllable!r} in voice "
            f"{voice!r}: {result.stderr.strip() or 'it wrote nothing'}")
    sound = parselmouth.Sound(str(path))
    samples = sound.values[0]
    heard = np.nonzero(np.abs(samples) > 0.01 * np.abs(samples).max())[0]
    if not len(heard):
        raise RuntimeError(
            f"{Path(program).name} said {syllable!r} in voice {voice!r} as "
            f"silence")
    rate = sound.sampling_frequency
    return sound.extract_part(heard[0] / rate, (heard[-1] + 1) / rate,
                              preserve_times=False)


def _sung(sound, frequency, seconds):
    """`sound` held at `frequency` and made `seconds` long, at RATE.

    Only the voiced stretch is lengthened, so a consonant keeps the length
    it was said with; a syllable too short for that, or with no voiced
    stretch, is scaled as a whole, to at most three times its length. One
    said in no more than :data:`SHORTEST` is only resampled: Praat cannot
    analyse it.
    """
    from parselmouth.praat import call

    total = sound.get_total_duration()
    if total <= SHORTEST:
        return call(sound, "Resample", RATE, 50).values[0]
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=PITCH_FLOOR,
                           pitch_ceiling=PITCH_CEILING)
    voiced = pitch.xs()[pitch.selected_array["frequency"] > 0]
    start, end = (voiced[0], voiced[-1]) if len(voiced) else (0.0, 0.0)
    unvoiced = total - (end - start)
    held = end - start > 0.01 and seconds > unvoiced
    if held and seconds > 3 * total:
        # Praat's overlap-add writes into a sound three times as long as
        # the one it is given, and stops at its end: a 0.3 s "laa" held
        # for four seconds was sung for one, and _fit made the rest
        # silence. Silence after the syllable gives it the room, and
        # _fit cuts the silence off again.
        sound = _padded(sound, seconds)
    manipulation = call(sound, "To Manipulation", 0.01, PITCH_FLOOR,
                        PITCH_CEILING)

    tier = call(manipulation, "Extract pitch tier")
    call(tier, "Remove points between", 0, total)
    call(tier, "Add point", 0, frequency)
    call(tier, "Add point", total, frequency)
    call([tier, manipulation], "Replace pitch tier")

    durations = call(manipulation, "Extract duration tier")
    if held:
        factor = (seconds - unvoiced) / (end - start)
        for time, value in ((max(start - 0.001, 0), 1), (start, factor),
                            (end, factor), (min(end + 0.001, total), 1)):
            call(durations, "Add point", time, value)
    else:
        call(durations, "Add point", 0, seconds / total)
    call([durations, manipulation], "Replace duration tier")

    sung = call(manipulation, "Get resynthesis (overlap-add)")
    return call(sung, "Resample", RATE, 50).values[0]


def _padded(sound, seconds):
    """`sound` followed by `seconds` of silence."""
    import parselmouth

    rate = sound.sampling_frequency
    silence = np.zeros(int(np.ceil(seconds * rate)))
    return parselmouth.Sound(np.concatenate([sound.values[0], silence]),
                             rate)


def _fit(samples, count):
    """`samples` made exactly `count` long, faded in and out."""
    fitted = np.zeros(count)
    fitted[:min(count, len(samples))] = samples[:count]
    fade = min(int(FADE * RATE), count // 4)
    if fade:
        ramp = np.linspace(0, 1, fade, endpoint=False)
        fitted[:fade] *= ramp
        fitted[count - fade:] *= ramp[::-1]
    return fitted
