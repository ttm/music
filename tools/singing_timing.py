"""The syllable timings of the listening experiment, beside ``sing``'s.

Run ``tools/compare_singing_timing.py`` to compare them: ``baseline``,
the timing the psola backend had until 2026-10-06, which said each
syllable at espeak's speed and stretched its whole voiced part; speech
rate fitting alone, up to 450 WPM; vowel holding alone; both together;
and ``slowed``: both, but only ever slowing espeak below its default 175
WPM, and holding the nucleus only on a note longer than the syllable, so
a short note is sung as by ``baseline``. ``slowed`` was heard clearest,
by Whisper and by ear, and is what ``music.sing`` now does; it is
rendered here by a search of its own, which a test holds to the
package's, sample for sample. The renderer uses the same private
synthesis and effects as the package, without patching module globals or
substituting a different speech engine.

The nucleus, :func:`music.singing.psola._nucleus`, is an energy
heuristic, not a phoneme aligner: it holds an interior part of the
strongest contiguous voiced region, and a diphthong or a voiced
consonant can still be selected.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from music.singing import perform, psola

VARIANTS = ("baseline", "speed", "vowel", "combined", "slowed")
SPEED_MIN, SPEED_START, SPEED_MAX = psola.SLOWEST, psola.SPEED, 450
ATTEMPTS = 4


def fit_speech(program, syllable, voice, path, seconds, fastest=SPEED_MAX):
    """Say a syllable at up to four rates, keeping the closest duration.

    Like eCantorix, update WPM by measured length / wanted length, within
    80..`fastest`. espeak's rate is approximate and its duration is not
    exactly inverse in WPM, so a later attempt need not be better.
    Repeated rates and a fit within 5 percent stop the search. PSOLA
    supplies the rest. With ``fastest=SPEED_START`` a syllable longer than
    its note is said once, at espeak's default speed.
    """
    speed, best, best_error = SPEED_START, None, float("inf")
    attempts = []
    seen = set()
    for _ in range(ATTEMPTS):
        if speed in seen:
            break
        seen.add(speed)
        sound = psola._speak(program, syllable, voice, path, speed=speed)
        duration = sound.get_total_duration()
        attempts.append({"wpm": speed, "seconds": duration})
        error = abs(duration - seconds)
        if error < best_error:
            best, best_error = sound, error
            chosen_speed = speed
        if error <= 0.05 * seconds:
            break
        speed = int(np.clip(round(speed * duration / seconds),
                            SPEED_MIN, fastest))
    return best, {"wpm": chosen_speed, "attempts": attempts}


def render(score, variant, speaker=None):
    """Render a comparison score and return samples plus note diagnostics.

    ``score`` uses the singing arguments in ``compare_singing.SCORES``.
    The small render loop deliberately mirrors the package's default so
    all variants use the same phonemes, effects, fades and rounding. A
    regression compares ``slowed`` samples with ``psola.sing`` exactly.
    ``speaker`` selects the same espeak binary for all the variants;
    eCantorix still uses the espeak on its PATH.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown timing variant: {variant!r}")
    effect = score.get("effect")
    if effect == "flint":
        effect = "flite"
    if effect not in (None, *psola.EFFECTS):
        raise ValueError(f"unknown effect: {effect!r}")
    if speaker is not None and effect != "flite":
        if not shutil.which(speaker):
            raise RuntimeError(f"speaker executable not found: {speaker!r}")
        if psola.importlib.util.find_spec("parselmouth") is None:
            raise RuntimeError(psola._INSTALL["praat-parselmouth"])
    else:
        psola.require(effect)
    voice = score.get("lang", "en")
    words = psola.syllables(score["text"])
    notes, durs = score["notes"], score["durs"]
    if not len(words) == len(notes) == len(durs):
        raise ValueError("one syllable and duration are needed per note")
    reference, transpose = score.get("reference", 60), score.get(
        "transpose", -12)
    perform.translate_to_abc(notes, durs, reference)
    unit = perform.unit_seconds(score.get("L", "1/4"), score.get("Q", 120))
    levels = perform.accents(durs, score.get("M", "4/4"),
                             score.get("L", "1/4"))
    lengths = [float(perform._note_length(d) * unit) for d in durs]
    edges = np.round(np.cumsum([0.0] + lengths) * psola.RATE).astype(int)
    line = np.zeros(edges[-1])
    if effect == "flite":
        psola._require_flite_voice(voice)
        program, phonemes = shutil.which("flite"), {}
    else:
        program = speaker or psola.speaker()
        if effect == "melt":
            voice = psola._melted(voice)
        phonemes = perform.sung_phonemes(
            program, voice, [w for w in map(psola._clean, words) if w])
    diagnostics = []
    with tempfile.TemporaryDirectory() as scratch:
        for index, (word, note) in enumerate(zip(words, notes)):
            start, end = int(edges[index]), int(edges[index + 1])
            said = psola._clean(word)
            if not said or end == start:
                continue
            frequency = 440 * 2 ** ((reference + note + transpose - 69) / 12)
            seconds = (end - start) / psola.RATE
            shift = (psola.MELT_FORMANTS * min(
                1.0, frequency / psola.MELT_FLOOR)) if effect == "melt" else 1
            target = seconds * shift
            path = Path(scratch) / f"{index}.wav"
            syllable = phonemes.get(said, said)
            fitting = {"wpm": None, "attempts": []}
            if variant in ("speed", "combined", "slowed") \
                    and effect != "flite":
                fastest = SPEED_START if variant == "slowed" else SPEED_MAX
                spoken, fitting = fit_speech(program, syllable, voice,
                                             path, target, fastest)
            else:
                spoken = psola._speak(program, syllable, voice, path)
            hold = variant in ("vowel", "combined") or (
                variant == "slowed"
                and target > spoken.get_total_duration())
            region = psola._nucleus(spoken) if hold else None
            samples = psola._sung(spoken, frequency / shift, target,
                                  region=region)
            if effect == "melt":
                samples = psola._resampled(samples, shift)
            part = psola._fit(samples, end - start) * levels[index]
            line[start:end] = psola._trembling(part) if effect in (
                "tremolo", "melt") else part
            duration = spoken.get_total_duration()
            # A region too narrow for compression falls back to scaling
            # the syllable whole. Report that, rather than claiming the
            # consonants were preserved when the score left no room.
            held = None
            if region is not None:
                a, b = region
                area = b - a + (min(0.001, a)
                                + min(0.001, duration - b)) / 2
                if target > duration - area:
                    held = list(region)
            diagnostics.append({
                "index": index, "syllable": said, "seconds": seconds,
                "source_seconds": duration, "target_seconds": target,
                "nucleus": list(region) if region else None,
                "held_nucleus": held, **fitting,
            })
    if effect in ("tremolo", "melt"):
        line = psola._reverberated(line)
    peak = np.abs(line).max() if line.size else 0.0
    return (line / peak if peak else line), diagnostics
