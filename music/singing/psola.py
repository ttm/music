"""Singing by speech synthesis and PSOLA: the default backend of ``sing``.

espeak-ng says each syllable of the lyric, and Praat's pitch-synchronous
overlap-add, through praat-parselmouth, holds it at its note's pitch and
stretches its voiced part to the note's length. The notes are then joined
end to end. That is eCantorix's idea -- a speech synthesizer made to sing
one syllable at a time -- without its toolchain: no Perl and its modules,
no round trip through ABC and MIDI, no sox.

It sings with the effects eCantorix's extra voices give, made with the
package's own tremolo, reverberation and resampling, and flite: see
:data:`EFFECTS`.

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

#: The effects this backend sings with, after eCantorix's extra voices of
#: the same names: a tremolo and reverberation, in stereo; the same with
#: the voice melted, its formants moving with the pitch; and flite's voice
#: in place of espeak's.
EFFECTS = ("tremolo", "melt", "flite")

#: The tremolo, as eCantorix's sox ``tremolo 9 50`` makes it: nine times a
#: second, the level falling to half and back. The package's
#: :func:`~music.core.synths.envelopes.tremolo` makes that ratio of two
#: as 3.01 dB either way of the middle.
TREMOLO_FREQ, TREMOLO_DB = 9, 10 * np.log10(2)

#: The reverberation, as sox's reverb at its defaults, which eCantorix
#: runs after the tremolo, was measured: its tail 10 dB below the direct
#: sound, with an RT60 of 0.87 s, and a second of silence after the line
#: to ring into. The package's :func:`~music.core.filters.reverb.reverb`,
#: decaying 40 dB over that second, rings for 0.86 s.
TAIL, WET_DB, DECAY = 1.0, -10.0, -40.0

#: The random seed the room is drawn from, so that a line has the same
#: reverberation every time it is sung.
ROOM = 0

#: The melted voice. eCantorix's melt voice is espeak's female ``f1``
#: with its formants raised by a quarter, at a pitch espeak can move down
#: to 216 Hz and no further: below that, eCantorix resamples it to each
#: note, and its formants sink with the pitch, as tape played slower. This
#: sings with ``f1``, unless ``lang`` names a variant, and moves its
#: formants the same way.
MELT_VARIANT, MELT_FORMANTS, MELT_FLOOR = "f1", 1.25, 216.0

#: How to install what this backend needs.
_INSTALL = {
    "espeak-ng": "espeak-ng: sudo apt install espeak-ng, or brew install "
                 "espeak-ng",
    "flite": "flite: sudo apt install flite, or brew install flite",
    "praat-parselmouth": "praat-parselmouth: pip install 'music[singing]'",
}


def speaker() -> str | None:
    """The first of :data:`SPEAKERS` on PATH, or None."""
    for name in SPEAKERS:
        found = shutil.which(name)
        if found:
            return found
    return None


def missing_requirements(effect=None) -> list[str]:
    """What this backend needs and cannot find: a speaker, and Parselmouth.

    Parameters
    ----------
    effect : str or None
        The effect to be sung with: ``"flite"`` speaks with flite rather
        than espeak-ng.

    Returns
    -------
    list of str
        ``"espeak-ng"`` when neither of :data:`SPEAKERS` is on PATH, or
        ``"flite"`` when the flite effect is asked for and flite is not;
        and ``"praat-parselmouth"`` when it cannot be imported.
    """
    missing = []
    if effect == "flite":
        if shutil.which("flite") is None:
            missing.append("flite")
    elif speaker() is None:
        missing.append("espeak-ng")
    if importlib.util.find_spec("parselmouth") is None:
        missing.append("praat-parselmouth")
    return missing


def require(effect=None) -> None:
    """Raise RuntimeError naming what this backend needs for `effect` and
    lacks."""
    missing = missing_requirements(effect)
    if missing:
        raise RuntimeError(
            "the psola singing backend needs "
            f"{' and '.join(missing)}, which are not installed. "
            + "; ".join(_INSTALL[name] for name in missing) + ".")


def _flite_voices() -> list[str]:
    """The voices flite has, as ``flite -lv`` lists them."""
    listed = subprocess.run(["flite", "-lv"], capture_output=True,
                            text=True).stdout
    return listed.partition(":")[2].split()


def _require_flite_voice(lang) -> None:
    """Raise ValueError unless `lang` names one of flite's voices.

    flite says a syllable in a voice it lacks in its default one, and says
    nothing of it, so the flite effect asks first.
    """
    voices = _flite_voices()
    if lang not in voices:
        raise ValueError(
            f"with the flite effect, lang names one of flite's voices, "
            f"{voices}; got {lang!r}")


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
         transpose=-12, effect=None) -> NDArray[np.float64]:
    """Sing a line of text to a melody, one syllable a note.

    The parameters are those of :func:`~music.singing.perform.sing`, which
    checks them and calls this for ``backend="psola"``, with ``effect`` one
    of :data:`EFFECTS` or None.

    Returns
    -------
    ndarray
        The sung line at :data:`RATE`, scaled to a peak of 1: every note
        exactly as long as its duration says. It is mono, or
        ``(2, nsamples)`` with the tremolo and melt effects, which ring
        :data:`TAIL` seconds past the line in stereo, as eCantorix's do.
        It is scaled rather than normalized, so a rest stays silent:
        taking out the mean would lift it off zero.

    Raises
    ------
    RuntimeError
        If espeak-ng, or flite for its effect, or praat-parselmouth is
        missing, or the speaker cannot say a syllable in ``lang``.
    ValueError
        If ``effect`` is not one of :data:`EFFECTS`, ``lang`` is not one
        of flite's voices for its effect, or there is not one syllable of
        ``text`` and one duration for each note.
    """
    from .perform import _note_length, sung_phonemes, unit_seconds

    if effect is not None and effect not in EFFECTS:
        raise ValueError(
            f"effect must be one of {EFFECTS}, or None; got {effect!r}")
    require(effect)
    if effect == "flite":
        _require_flite_voice(lang)
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
    reverberant = effect in ("tremolo", "melt")
    if effect == "flite":
        # flite reads no espeak phonemes, and lang names its voice.
        program, voice, sung = shutil.which("flite"), lang, {}
    else:
        program = speaker()
        voice = _melted(lang) if effect == "melt" else lang
        # A syllable said without a vowel is sung with a schwa.
        sung = sung_phonemes(program, voice,
                             [said for said in map(_clean, words) if said])
    with tempfile.TemporaryDirectory() as scratch:
        for index, (word, note) in enumerate(zip(words, notes)):
            start, end = int(edges[index]), int(edges[index + 1])
            said = _clean(word)
            if not said or end == start:
                continue  # a syllable of punctuation alone is a rest
            spoken = _speak(program, sung.get(said, said), voice,
                            Path(scratch) / f"{index}.wav")
            frequency = 440 * 2 ** ((reference + note + transpose - 69) / 12)
            seconds = (end - start) / RATE
            if effect == "melt":
                # Held at the pitch that resampling by `shift` takes to
                # the note, and for the length it takes to the note's.
                shift = MELT_FORMANTS * min(1.0, frequency / MELT_FLOOR)
                samples = _resampled(
                    _sung(spoken, frequency / shift, seconds * shift), shift)
            else:
                samples = _sung(spoken, frequency, seconds)
            part = _fit(samples, end - start)
            line[start:end] = _trembling(part) if reverberant else part
    if reverberant:
        line = _reverberated(line)
    peak = np.abs(line).max() if line.size else 0.0
    return line / peak if peak else line


def _melted(lang):
    """The voice the melt effect sings with: `lang`, in the variant
    :data:`MELT_VARIANT` unless it names one."""
    return lang if "+" in lang else f"{lang}+{MELT_VARIANT}"


def _clean(word: str) -> str:
    """A syllable with its punctuation taken off, as it is said."""
    return re.sub(r"[^\w']", "", word)


def _speak(program, syllable, voice, path):
    """`syllable` said by `program` in `voice`, trimmed of its silence."""
    import parselmouth

    result = subprocess.run(_command(program, syllable, voice, path),
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


def _command(program, syllable, voice, path):
    """What has `program` say `syllable` in `voice`, into `path`: flite
    names its voice, its text and its output otherwise than espeak."""
    if Path(program).stem == "flite":
        return [program, "-voice", voice, "-t", syllable, "-o", str(path)]
    return [program, "-v", voice, "-w", str(path), syllable]


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



def _trembling(samples):
    """`samples` with eCantorix's tremolo, from their own start, as sox
    gave each syllable its own."""
    from ..core.synths.envelopes import tremolo

    return tremolo(sonic_vector=samples, tremolo_freq=TREMOLO_FREQ,
                   max_db_dev=TREMOLO_DB, sample_rate=RATE)


def _resampled(samples, ratio):
    """`samples` played `ratio` times as fast, as tape would be, back at
    :data:`RATE`: pitch and formants up by `ratio`, the length down."""
    import parselmouth
    from parselmouth.praat import call

    return call(parselmouth.Sound(samples, RATE * ratio), "Resample", RATE,
                50).values[0]


def _reverberated(line):
    """`line` in a room, in stereo, ringing :data:`TAIL` seconds past it.

    Both channels have the direct sound and a reverberation of their own,
    as sox's reverb gives a line made stereo. The room is drawn from
    :data:`ROOM`, so it is the same every time, and the caller's random
    state is left as it was.
    """
    from ..core.filters.reverb import reverb

    state = np.random.get_state()
    try:
        np.random.seed(ROOM)
        responses = [reverb(duration=TAIL, decay=DECAY, sample_rate=RATE)
                     for _ in range(2)]
    finally:
        np.random.set_state(state)
    length = len(line) + int(TAIL * RATE)
    channels = []
    for response in responses:
        wet = response[1:]
        response[1:] = wet * 10 ** (WET_DB / 20) / np.sqrt(np.sum(wet ** 2))
        channels.append(_convolved(line, response, length))
    return np.array(channels)


def _convolved(signal, response, length):
    """The first `length` samples of `signal` convolved with `response`,
    by FFT: numpy's convolve takes seconds over a sung line."""
    needed = max(len(signal) + len(response) - 1, length)
    size = 1 << (needed - 1).bit_length()
    spectrum = np.fft.rfft(signal, size) * np.fft.rfft(response, size)
    return np.fft.irfft(spectrum, size)[:length]
