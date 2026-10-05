"""Singing by speech synthesis and PSOLA: the default backend of ``sing``.

espeak-ng says each syllable of the lyric, and Praat's pitch-synchronous
overlap-add, through praat-parselmouth, holds it at its note's pitch and
makes it the note's length. A syllable said in less time than its note is
said again slower, as eCantorix has espeak say it, and then only the
strong middle of its vowel is lengthened, so that its consonants keep the
length they were said with. The notes are then joined end to end. That is
eCantorix's idea -- a speech synthesizer made to sing one syllable at a
time -- without its toolchain: no Perl and its modules, no round trip
through ABC and MIDI, no sox.

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
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import parselmouth

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

#: espeak's speaking speed, in words a minute, and the slowest it is asked
#: for. A syllable said in less time than its note is said again slower,
#: as eCantorix has espeak say each syllable at the speed that fits its
#: note, down to the same 80. Never faster: on a note shorter than its
#: syllable, espeak hurried was heard less clearly than PSOLA compressing
#: its usual speech, by Whisper and by ear (2026-10-05 and 06; see
#: ``tools/score_singing_asr.py``).
SPEED, SLOWEST = 175, 80

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
    assert program is not None  # require() found it
    with tempfile.TemporaryDirectory() as scratch:
        for index, (word, note) in enumerate(zip(words, notes)):
            start, end = int(edges[index]), int(edges[index + 1])
            said = _clean(word)
            if not said or end == start:
                continue  # a syllable of punctuation alone is a rest
            frequency = 440 * 2 ** ((reference + note + transpose - 69) / 12)
            # The melted voice is held at the pitch that resampling by
            # `shift` takes to the note, and for the length it takes to
            # the note's.
            shift = (MELT_FORMANTS * min(1.0, frequency / MELT_FLOOR)
                     if effect == "melt" else 1.0)
            seconds = (end - start) / RATE * shift
            spoken, held = _said_for(program, sung.get(said, said), voice,
                                     Path(scratch) / f"{index}.wav", seconds)
            samples = _sung(spoken, frequency / shift, seconds, region=held)
            if effect == "melt":
                samples = _resampled(samples, shift)
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


def _speak(program, syllable, voice, path, speed=None):
    """`syllable` said by `program` in `voice`, trimmed of its silence:
    at `speed` words a minute if one is given, and the program is espeak.
    """
    import parselmouth

    result = subprocess.run(_command(program, syllable, voice, path, speed),
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


def _command(program, syllable, voice, path, speed=None):
    """What has `program` say `syllable` in `voice`, into `path`: flite
    names its voice, its text and its output otherwise than espeak, and
    has no `speed` to be asked for."""
    if _is_flite(program):
        return [program, "-voice", voice, "-t", syllable, "-o", str(path)]
    command = [program, "-v", voice, "-w", str(path)]
    if speed is not None:
        command += ["-s", str(speed)]
    return command + [syllable]


def _is_flite(program: str) -> bool:
    """Whether `program` is flite, which is asked otherwise than espeak."""
    return Path(program).stem == "flite"


def _said_for(program: str, syllable: str, voice: str, path: Path,
              seconds: float
              ) -> tuple[parselmouth.Sound, tuple[float, float] | None]:
    """`syllable` said for a note `seconds` long, and the stretch to hold.

    A syllable said in less time than its note is said again slower, down
    to :data:`SLOWEST` words a minute, unless flite says it, which cannot
    be asked to; if it is still shorter than the note, the strong middle
    of its vowel, as :func:`_nucleus` finds it, is held for the rest, so
    that its consonants keep the length they were said with. A syllable
    said in the note's time or more is left as it was said, for
    :func:`_sung` to fit.

    Returns
    -------
    sound, tuple of float or None
        The syllable as said, and the stretch of it to hold, in seconds,
        or None to let :func:`_sung` choose.
    """
    spoken = _speak(program, syllable, voice, path)
    if spoken.get_total_duration() < seconds and not _is_flite(program):
        spoken = _slowed(program, syllable, voice, path, seconds, spoken)
    if spoken.get_total_duration() < seconds:
        return spoken, _nucleus(spoken)
    return spoken, None


def _slowed(program: str, syllable: str, voice: str, path: Path,
            seconds: float, spoken: parselmouth.Sound) -> parselmouth.Sound:
    """`syllable` said slower, as near `seconds` long as espeak will.

    `spoken` is the syllable said at :data:`SPEED`. As eCantorix does,
    the speed is scaled by the length said over the length wanted, down
    to :data:`SLOWEST`, for up to three more tries. espeak's speed is
    approximate, and a syllable's length not inverse in it, so the try
    nearest `seconds` is kept; one within 5 percent of it ends the
    search, and so does a speed tried before.
    """
    best, speed, tried = spoken, SPEED, {SPEED}
    said = spoken.get_total_duration()
    for _ in range(3):
        if abs(said - seconds) <= 0.05 * seconds:
            break
        speed = min(SPEED, max(SLOWEST, round(speed * said / seconds)))
        if speed in tried:
            break
        tried.add(speed)
        sound = _speak(program, syllable, voice, path, speed)
        said = sound.get_total_duration()
        if abs(said - seconds) < abs(best.get_total_duration() - seconds):
            best = sound
    return best


def _nucleus(sound: parselmouth.Sound) -> tuple[float, float] | None:
    """The strong middle of the vowel of `sound`, as (start, end) seconds.

    Its pitch every 10 ms, as :func:`_sung` finds it, and its level over
    40 ms around each; see :func:`_nucleus_from_frames`. None where the
    syllable is too short for Praat to find a pitch in.
    """
    if sound.get_total_duration() <= SHORTEST:
        return None
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=PITCH_FLOOR,
                           pitch_ceiling=PITCH_CEILING)
    times = pitch.xs()
    samples, rate = sound.values[0], sound.sampling_frequency
    centres = np.round(times * rate).astype(int)
    half = max(1, round(0.02 * rate))
    left = np.maximum(0, centres - half)
    right = np.minimum(len(samples), centres + half)
    energy = np.concatenate(([0.0], np.cumsum(samples ** 2)))
    levels = np.sqrt((energy[right] - energy[left]) / (right - left))
    return _nucleus_from_frames(times, pitch.selected_array["frequency"],
                                levels)


def _nucleus_from_frames(times: ArrayLike, frequencies: ArrayLike,
                         levels: ArrayLike) -> tuple[float, float] | None:
    """The loudest voiced stretch of frames, less its edges, or None.

    The voiced frame with the most energy, and the voiced frames either
    side of it at least 70 percent as strong, never across an unvoiced
    one; less 10 ms at each end, and None where that leaves less than
    30 ms. It reads energy, not phonemes: it keeps the onset and coda
    out of the hold, but a loud voiced consonant, or a diphthong's glide,
    can be held too.
    """
    at, pitch, level = (np.asarray(values, dtype=float)
                        for values in (times, frequencies, levels))
    voiced = pitch > 0
    if not voiced.any():
        return None
    peak = int(np.argmax(np.where(voiced, level, -1)))
    if level[peak] <= 0:
        return None
    usable = voiced & (level >= 0.7 * level[peak])
    left = right = peak
    while left > 0 and usable[left - 1]:
        left -= 1
    while right + 1 < len(at) and usable[right + 1]:
        right += 1
    start, end = float(at[left] + 0.01), float(at[right] - 0.01)
    return (start, end) if end - start >= 0.03 - 1e-12 else None


def _sung(sound, frequency, seconds, *, region=None):
    """`sound` held at `frequency` and made `seconds` long, at RATE.

    Only the voiced stretch is lengthened, so a consonant keeps the length
    it was said with; a syllable too short for that, or with no voiced
    stretch, is scaled as a whole, to at most three times its length. One
    said in no more than :data:`SHORTEST` is only resampled: Praat cannot
    analyse it. ``region=(start, end)``, such as the nucleus
    :func:`_nucleus` finds, is the stretch lengthened instead, its short
    ramps counted in its length; one that cannot be made to fit scales
    the syllable whole.
    """
    from parselmouth.praat import call

    total = sound.get_total_duration()
    if total <= SHORTEST:
        return call(sound, "Resample", RATE, 50).values[0]
    if region is None:
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=PITCH_FLOOR,
                               pitch_ceiling=PITCH_CEILING)
        voiced = pitch.xs()[pitch.selected_array["frequency"] > 0]
        start, end = (voiced[0], voiced[-1]) if len(voiced) else (0.0, 0.0)
        unvoiced = total - (end - start)
        held = end - start > 0.01 and seconds > unvoiced
    else:
        start, end = region
        held = 0 <= start < end <= total and end - start > 0.01
        if held:
            left, right = min(0.001, start), min(0.001, total - end)
            factor = 1 + (seconds - total) / (
                end - start + (left + right) / 2)
            held = factor > 0
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
        if region is None:
            factor = (seconds - unvoiced) / (end - start)
            points = [(max(start - 0.001, 0), 1), (start, factor),
                      (end, factor), (min(end + 0.001, total), 1)]
        else:
            points = []
            if left:
                points.append((start - left, 1))
            points.extend(((start, factor), (end, factor)))
            if right:
                points.append((end + right, 1))
            elif seconds > 3 * total:
                # A region reaching the end has no outgoing ramp. Return
                # to unit rate in the padding, without duplicate times.
                points.append((end + 0.001, 1))
        for time, value in points:
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
