#!/usr/bin/env python3
"""Render the PSOLA timing experiment for a level-matched listening review.

Run ``python tools/compare_singing_timing.py --out DIR``. The baseline,
speed fitting alone, vowel holding alone, both changes, and both changes
limited to slowing espeak on longer notes sing each score; eCantorix also
sings when installed. ``index.html`` plays the matched WAVs,
and JSON/Markdown reports retain measurements and synthesis diagnostics.
``slowed`` is the psola backend's own timing since 2026-10-06, and
``baseline`` the one it replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_singing import (  # noqa: E402
    RATE, SCORES as STANDARD_SCORES, available_backends, cents,
    expected_pitches, expected_seconds, markdown as measurement_table,
    note_pitches,
)

SCORES = {
    **STANDARD_SCORES,
    "french": dict(text="Jac-ques", notes=(4, 0), durs=(1, 1), lang="fr"),
    "consonants": dict(text="bright blue bells ring sweet songs",
                       notes=(0, 2, 4, 5, 7, 0),
                       durs=(0.5, 1, 0.5, 2, 1, 2)),
    "long coda": dict(text="lamb ring", notes=(0, 7), durs=(8, 8)),
}

#: One-syllable words, half of them opening or closing on a cluster, in no
#: order a sentence would put them, so that a listener, or the recognizer
#: of ``tools/score_singing_asr.py``, has the sound alone to go by.
WORD_LINES = (
    "swim stop kick king duck stone", "fig lake frog plant black cloud",
    "jump dig bat gold man fin", "flag ship pack crowd smile class",
    "map back truck late snake cup", "bread thick shell cake beach heat",
    "glass sweet brush tent cave spring", "bug beam green chip dip path",
)

#: ``--words``: each line sung a word a note, of 0.5, 1, 2 and 4 quarter
#: notes at 120 a minute, 0.25 to 2 seconds, since stretching matters more
#: the longer the note.
WORDS = {
    f"{units / 2:g} s words {number}": dict(
        text=line, notes=(0, 2, 4, 5, 7, 4), durs=(units,) * 6)
    for units in (0.5, 1, 2, 4)
    for number, line in enumerate(WORD_LINES, 1)
}

LABELS = {
    "baseline": "PSOLA's timing until 2026-10-06",
    "speed": "Fit espeak speed",
    "vowel": "Hold the vowel nucleus",
    "combined": "Fit speed and hold the nucleus",
    "slowed": "Only slow espeak, hold the nucleus on long notes (sing's)",
    "ecantorix": "eCantorix",
}

PRAAT_SEED = 42


def rms(sound):
    """The sample RMS, including silence, over all channels."""
    samples = np.asarray(sound, dtype=float)
    return float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0


def level_match(sounds, target_rms=0.1, ceiling=0.98):
    """Match non-silent sounds to one RMS without clipping any of them.

    A shared final attenuation makes room for the largest crest factor.
    No per-file peak normalization follows the RMS matching. Silent files
    stay silent and are explicitly identified in the returned metadata.
    RMS matching is a repeatable level control, not a perceptual loudness
    model; different vowel spectra can still sound differently loud.
    """
    if not np.isfinite(target_rms) or target_rms <= 0:
        raise ValueError("target_rms must be finite and positive")
    if not np.isfinite(ceiling) or not 0 < ceiling <= 1:
        raise ValueError("ceiling must be between zero and one")
    arrays = [np.asarray(sound, dtype=float) for sound in sounds]
    if any(not np.isfinite(sound).all() for sound in arrays):
        raise ValueError("cannot match a sound with non-finite samples")
    levels = [rms(sound) for sound in arrays]
    peaks = [float(np.max(np.abs(sound))) if sound.size else 0.0
             for sound in arrays]
    nominal = [target_rms / level if level else 1.0 for level in levels]
    largest = max((peak * gain for peak, gain in zip(peaks, nominal)),
                  default=0.0)
    attenuation = min(1.0, ceiling / largest) if largest else 1.0
    matched, gains = [], []
    for sound, level, peak, gain in zip(arrays, levels, peaks, nominal):
        gain *= attenuation
        output = sound * gain
        matched.append(output)
        gains.append({
            "raw_rms": level,
            "raw_peak": peak,
            "gain": gain,
            "gain_db": float(20 * np.log10(gain)),
            "matched_rms": rms(output),
            "matched_peak": peak * gain,
            "silent": level == 0,
            "requested_rms": target_rms,
            "shared_rms": target_rms * attenuation,
            "shared_attenuation": attenuation,
        })
    return matched, gains


def metrics(sound, score, name, variant, elapsed):
    """Measure a rendering once against the existing comparison rules."""
    mono = np.asarray(sound)
    if mono.ndim == 2:
        mono = mono.mean(axis=0)
    seconds = expected_seconds(score)
    expected = expected_pitches(score)
    pitches = note_pitches(mono, seconds)
    errors = [None if pitch is None else round(float(cents(pitch, hz)), 1)
              for pitch, hz in zip(pitches, expected)]
    measured = [abs(error) for error in errors if error is not None]
    tail = 1.0 if score.get("effect") in ("tremolo", "melt") else 0.0
    return {
        "score": name,
        "backend": variant,
        "render_seconds": round(elapsed, 3),
        "length_error_ms": round(1000 * (len(mono) / RATE
                                         - sum(seconds) - tail), 3),
        "expected_tail_seconds": tail,
        "cents_by_note": errors,
        "expected_hz_by_note": expected,
        "measured_hz_by_note": pitches,
        "median_abs_cents": (round(float(np.median(measured)), 1)
                             if measured else None),
        "max_abs_cents": round(max(measured), 1) if measured else None,
        "unmeasured_notes": errors.count(None),
    }


def _version(command):
    try:
        result = subprocess.run([command, "--version"], capture_output=True,
                                text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"
    return (result.stdout or result.stderr).strip()


def _revision():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _seed_praat():
    """Give each PSOLA variant the same reproducible Praat random state."""
    from parselmouth.praat import run

    run(f"random_initializeWithSeedUnsafelyButPredictably ({PRAAT_SEED})")


def _provenance():
    """Identify the actual working sources, even before they are committed."""
    sources = ("music/singing/psola.py", "music/singing/perform.py",
               "tools/singing_timing.py", "tools/compare_singing.py",
               "tools/compare_singing_timing.py")
    versions = {}
    for name in ("numpy", "praat-parselmouth", "soundfile"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "package_versions": versions,
            "source_sha256": {name: hashlib.sha256(
                (ROOT / name).read_bytes()).hexdigest() for name in sources}}


def _table_rows(report):
    """Escape arbitrary score/variant names for the Markdown table."""
    return [dict(row, **{key: str(row[key]).replace("|", "\\|")
                        .replace("\n", " ")
                        for key in ("score", "backend")})
            for row in report["results"]]


def markdown(report):
    return (
        "# PSOLA timing experiment\n\n"
        "Open `index.html` to compare the level-matched recordings. "
        "`slowed` is the psola backend's timing since 2026-10-06, heard "
        "clearest in general by Whisper and by ear; `baseline` is the one "
        "it replaced. Pitch and duration measurements do not establish "
        "syllable clarity.\n\n"
        "Each score uses a shared sample RMS, lowered equally across its "
        "variants if needed to prevent clipping. RMS matching is not "
        "perceptual loudness normalization. `raw/` contains the original "
        "renderings; `matched/` contains the review files. Exact gains, "
        "speaker versions and per-syllable diagnostics are in "
        "`report.json`. Praat starts each PSOLA rendering with the same "
        f"random seed ({PRAAT_SEED}).\n\n"
        "All PSOLA variants use the selected speech engine. When eCantorix "
        "is included, the CLI defaults to its `espeak` for this comparison; "
        "the package normally prefers `espeak-ng` when installed.\n\n"
        + measurement_table(_table_rows(report))
    )


def review_html(report):
    """A standalone local review page; all audio URLs are relative."""
    esc = html.escape
    sections = []
    for name, score in report["scores"].items():
        rows = []
        for row in report["results"]:
            if row["score"] != name:
                continue
            variant = row["backend"]
            error = row["max_abs_cents"]
            largest = "unmeasured" if error is None else f"{error} cents"
            rows.append(
                "<tr><th scope='row'>"
                + esc(LABELS.get(variant, variant))
                + "</th><td><audio controls preload='none' src='"
                + esc(row["matched_file"], quote=True)
                + "'></audio></td><td>"
                + esc(largest)
                + f"</td><td>{row['unmeasured_notes']}</td><td>"
                + f"{row['length_error_ms']} ms</td></tr>")
        sections.append(
            "<section><h2>" + esc(name) + "</h2><p>"
            + esc(score["text"])
            + "</p><div class='scroll'><table><thead><tr>"
            "<th>Variant</th><th>Recording</th><th>Largest pitch error</th>"
            "<th>Unmeasured notes</th><th>Length error</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table></div></section>")
    speaker = report["environment"]["speaker"]
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PSOLA timing listening review</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;
padding:0 1rem;color:#18232e;background:#fafbfc}h1{line-height:1.2}
section{margin:2rem 0}table{border-collapse:collapse;width:100%;
background:white}
th,td{text-align:left;padding:.65rem;border-bottom:1px solid #ccd5df}
thead{background:#edf2f7}.scroll{overflow-x:auto}audio{max-width:100%;width:280px}
a{color:#145fa5}code{overflow-wrap:anywhere}
.note{border-left:4px solid #317c9d;
padding:.6rem 1rem;background:#edf6f8}footer{font-size:.9rem}
</style></head><body>
<h1>PSOLA timing listening review</h1>
<p class="note">Since 2026-10-06, slowed is the psola backend's own timing:
heard clearest in general, by Whisper and by ear, if not always clearly
better than the others. Baseline is the timing it replaced.</p>
<p>Compare the consonant attack, vowel transition, sustained vowel and final
consonants. Check that words stay recognizable on quick notes and that long
notes sound through their endings. Pitch and duration measurements alone do
not establish clarity.</p>
<p>All non-silent variants of a score share a sample RMS, with common
attenuation if needed to prevent clipping. This controls signal level;
perceived loudness can still differ. Pause a recording before playing the
next. Baseline says each syllable at espeak's speed and stretches its whole
voiced part; speed fits espeak's speaking rate to each note; nucleus holds
an energy-based estimate of its strong voiced middle; slowed does both, but
never speeds espeak up and holds the nucleus only on a note longer than the
syllable, so short notes are sung as by the baseline.</p>
<p>All PSOLA variants use the selected speaker below. The comparison defaults
to eCantorix's espeak when that backend is available; the package normally
prefers espeak-ng. Thus each PSOLA row is its timing with the selected
speaker, which need not be the one <code>sing</code> would use. eCantorix
always uses espeak.</p>
<p><a href="report.json">Full measurements and synthesis diagnostics</a>
 · <a href="report.md">Markdown report</a></p>
""" + "<p>PSOLA speaker: <code>" + esc(speaker) + "</code></p>" \
        + "".join(sections) + """
<footer>Original renderings are in <code>raw/</code>; review recordings are
in <code>matched/</code>. Gain values and engine versions are in the JSON
report. Browser playback requires no server or network connection.</footer>
</body></html>
"""


def compare(out, scores, speaker, include_ecantorix=False):
    """Render each score/variant serially once and write review artifacts."""
    import soundfile
    from music.singing.perform import sing
    from tools.singing_timing import VARIANTS, render

    out = Path(out)
    for directory in (out / "raw", out / "matched"):
        directory.mkdir(parents=True, exist_ok=True)
    other_speaker = shutil.which("espeak") if include_ecantorix else None
    report = {
        "experiment": "PSOLA timing: rate fitting and vowel-nucleus holding",
        "package_timing": "slowed",
        "sample_rate": RATE,
        "scores": scores,
        "environment": {
            **_provenance(),
            "speaker": speaker,
            "speaker_version": _version(speaker),
            "ecantorix_speaker": other_speaker,
            "ecantorix_speaker_version": (_version(other_speaker)
                                           if other_speaker else None),
            "git_head": _revision(),
            "ecantorix_included": include_ecantorix,
            "praat_seed_per_render": PRAAT_SEED,
        },
        "results": [],
    }
    variants = list(VARIANTS) + (["ecantorix"] if include_ecantorix else [])
    for index, (name, score) in enumerate(scores.items()):
        sounds, rows = [], []
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "score"
        slug = f"{index + 1:02d}-{slug}"
        for variant in variants:
            print(f"Rendering {name}: {variant}", flush=True)
            started = time.perf_counter()
            if variant == "ecantorix":
                sound, diagnostics = sing(backend="ecantorix", **score), []
            else:
                _seed_praat()
                sound, diagnostics = render(score, variant, speaker=speaker)
            elapsed = time.perf_counter() - started
            row = metrics(sound, score, name, variant, elapsed)
            row["diagnostics"] = diagnostics
            row["raw_file"] = f"raw/{slug}.{variant}.wav"
            row["matched_file"] = f"matched/{slug}.{variant}.wav"
            # Float WAV retains the renderer's original levels and peaks.
            soundfile.write(out / row["raw_file"], np.asarray(sound).T,
                            RATE, subtype="FLOAT")
            sounds.append(sound)
            rows.append(row)
        matched, gains = level_match(sounds)
        for sound, row, gain in zip(matched, rows, gains):
            soundfile.write(out / row["matched_file"], sound.T, RATE,
                            subtype="PCM_16")
            row["level_matching"] = gain
        report["results"].extend(rows)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "report.md").write_text(markdown(report))
    (out / "index.html").write_text(review_html(report))
    return report


def main(argv=None):
    from music.singing import psola

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="singing-timing-comparison",
                        help="directory for the review page, WAVs and reports")
    parser.add_argument("--speaker", help="espeak or espeak-ng executable")
    parser.add_argument("--skip-ecantorix", action="store_true",
                        help="render only the PSOLA variants")
    parser.add_argument("--words", action="store_true",
                        help="sing the word lists, for "
                             "tools/score_singing_asr.py, not the scores")
    parser.add_argument("--score", action="append",
                        choices=[*SCORES, *WORDS], metavar="NAME",
                        help="score to render, from SCORES or WORDS; "
                             "repeat for several")
    args = parser.parse_args(argv)
    if "praat-parselmouth" in psola.missing_requirements():
        parser.error("the timing experiment needs praat-parselmouth; "
                     "install music[singing]")
    include_ecantorix = (not args.skip_ecantorix
                         and "ecantorix" in available_backends())
    chosen = args.speaker or (shutil.which("espeak") if include_ecantorix
                              else psola.speaker())
    speaker = shutil.which(chosen) if chosen else None
    if speaker is None:
        parser.error(f"speaker executable not found: {chosen!r}")
    if not args.skip_ecantorix and not include_ecantorix:
        print("eCantorix is unavailable; rendering the PSOLA variants only.")
    if include_ecantorix and Path(speaker).name != "espeak":
        print("PSOLA uses the selected speaker; eCantorix uses espeak. "
              "Their versions are recorded separately.")
    sung = WORDS if args.words else SCORES
    scores = {name: {**SCORES, **WORDS}[name]
              for name in (args.score or sung)}
    report = compare(args.out, scores, speaker, include_ecantorix)
    print(measurement_table(report["results"]))
    print(f"Listening review: {Path(args.out).resolve() / 'index.html'}")


if __name__ == "__main__":
    main()
