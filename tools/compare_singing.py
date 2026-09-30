#!/usr/bin/env python3
"""Sing the same scores with every singing backend, and measure them.

``music.singing.sing`` has two backends: eCantorix, the Perl engine this
package has always used, and ``psola``, espeak-ng with Praat's PSOLA. They
sing the same scores to the same pitches, MIDI ``reference + note +
transpose``, for the same lengths, so they can be measured against the
score and against each other. This renders each score below with each
backend that is installed, and reports for each:

* the pitch of every note, as the autocorrelation peak in the voiced
  middle of the note, and its error from the score in cents;
* how far the whole line is from the length the score gives it;
* how long it took to render.

It writes each rendering as a WAV, for listening, with a JSON and a
Markdown report beside them.

Usage
-----
::

    python tools/compare_singing.py                # into ./singing-comparison
    python tools/compare_singing.py --out DIR --backend psola

The measurement here is also what ``tests/test_singing_engine.py`` asserts
with, so a number this reports is one the tests check.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The scores, each as the keyword arguments ``sing`` takes.
SCORES = {
    "mary": dict(text="Mar-ry had a litt-le lamb",
                 notes=(4, 2, 0, 2, 4, 4, 4), durs=(1, 1, 1, 1, 1, 1, 2)),
    "octave": dict(text="laa laa laa laa laa",
                   notes=(0, 4, 7, 12, 0), durs=(2, 2, 2, 2, 4),
                   transpose=0),
    "quick": dict(text="la la la la la la la la",
                  notes=(0, 2, 4, 5, 7, 9, 11, 12),
                  durs=(0.5,) * 8, transpose=0),
    "test song": dict(text="hey ma bro, why fly while dive?",
                      notes=(7, 0, 5, 7, 11, 12, 7),
                      durs=(0.5, 0.5, 0.25, 0.25, 1, 0.25, 0.5)),
    "german": dict(text="Al-le mei-ne Ent-chen", notes=(0, 2, 4, 5, 7, 7),
                   durs=(1, 1, 1, 1, 2, 2), lang="de"),
}

RATE = 44100


def fundamental(segment, rate=RATE, low=40, high=1500):
    """The pitch of `segment`, in Hertz, or None where it has none.

    The autocorrelation peak between `low` and `high` Hz, refined by a
    parabola through it. A segment whose peak is under half of its energy
    is unvoiced, or too short to tell, and gives None.
    """
    segment = np.asarray(segment, dtype=float)
    segment = segment - segment.mean()
    energy = float(np.dot(segment, segment))
    if not energy:
        return None
    correlation = np.correlate(segment, segment, "full")[len(segment) - 1:]
    lags = np.arange(len(correlation))
    usable = (lags >= rate / high) & (lags <= rate / low)
    if not usable.any():
        return None
    lag = int(lags[usable][np.argmax(correlation[usable])])
    if correlation[lag] < 0.5 * energy or lag + 1 >= len(correlation):
        return None
    left, peak, right = correlation[lag - 1:lag + 2]
    bend = left - 2 * peak + right
    offset = 0.5 * (left - right) / bend if bend else 0.0
    return rate / (lag + offset)


def note_pitches(sound, seconds, rate=RATE, windows=9):
    """The pitch of each note of `sound`, whose lengths are `seconds`.

    The median over windows spread across the middle of each note, of the
    voiced ones at least half as loud as the loudest: a sung syllable can
    fall nearly silent for a moment, and a window there measures nothing.
    A note with no voiced window gives None.
    """
    edges = np.round(np.cumsum([0.0] + list(seconds)) * rate).astype(int)
    pitches = []
    for start, end in zip(edges[:-1], edges[1:]):
        half = max(min(4096, (end - start) // 4), 64)
        centres = start + np.linspace(0.2, 0.8, windows) * (end - start)
        segments = [sound[max(int(c) - half, 0):int(c) + half]
                    for c in centres]
        loudest = max(float(np.abs(s).max()) if len(s) else 0.0
                      for s in segments)
        found = [fundamental(s, rate) for s in segments
                 if len(s) and np.abs(s).max() >= loudest / 2]
        found = [f for f in found if f is not None]
        pitches.append(float(np.median(found)) if found else None)
    return pitches


def cents(measured, expected):
    return 1200 * np.log2(measured / expected)


def expected_pitches(score):
    reference = score.get("reference", 60)
    transpose = score.get("transpose", -12)
    return [440 * 2 ** ((reference + note + transpose - 69) / 12)
            for note in score["notes"]]


def expected_seconds(score):
    from music.singing.perform import _note_length, unit_seconds

    unit = unit_seconds(score.get("L", "1/4"), score.get("Q", 120))
    return [float(_note_length(d) * unit) for d in score["durs"]]


def available_backends():
    """The backends whose engines and programs are all here."""
    from music.singing import paths, psola

    found = []
    try:
        if (paths.is_engine(paths.engine_dir())
                and not paths.missing_requirements()
                and not paths.missing_perl_modules()):
            found.append("ecantorix")
    except OSError:
        pass
    if not psola.missing_requirements():
        found.append("psola")
    return found


def measure(backend, name, score, out=None):
    """Render `score` with `backend` and measure it against the score."""
    from music.singing.perform import sing

    started = time.perf_counter()
    sound = sing(backend=backend, **score)
    elapsed = time.perf_counter() - started
    if sound.ndim == 2:
        sound = sound.mean(axis=0)
    seconds = expected_seconds(score)
    pitches = note_pitches(sound, seconds)
    errors = [None if p is None else round(float(cents(p, e)), 1)
              for p, e in zip(pitches, expected_pitches(score))]
    measured = [abs(e) for e in errors if e is not None]
    if out is not None:
        import soundfile
        slug = name.replace(" ", "-")
        soundfile.write(str(out / f"{slug}.{backend}.wav"), sound, RATE)
    return {
        "backend": backend,
        "score": name,
        "render_seconds": round(elapsed, 2),
        "length_error_ms": round(1000 * (len(sound) / RATE
                                         - sum(seconds)), 1),
        "cents_by_note": errors,
        "median_abs_cents": (round(float(np.median(measured)), 1)
                             if measured else None),
        "max_abs_cents": round(max(measured), 1) if measured else None,
        "unmeasured_notes": errors.count(None),
    }


def markdown(results):
    lines = ["| Score | Backend | Median error | Largest error | "
             "Unmeasured notes | Length error | Render time |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in results:
        median = row["median_abs_cents"]
        largest = row["max_abs_cents"]
        lines.append(
            f"| {row['score']} | {row['backend']} | "
            f"{'-' if median is None else f'{median} cents'} | "
            f"{'-' if largest is None else f'{largest} cents'} | "
            f"{row['unmeasured_notes']} | {row['length_error_ms']} ms | "
            f"{row['render_seconds']} s |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="singing-comparison",
                        help="where to write the WAVs and the report")
    parser.add_argument("--backend", action="append",
                        help="a backend to use; every one installed if none")
    args = parser.parse_args(argv)
    backends = args.backend or available_backends()
    if not backends:
        raise SystemExit("no singing backend is installed: see "
                         "music.singing.setup_engine and "
                         "music.singing.psola")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = [measure(backend, name, score, out)
               for name, score in SCORES.items() for backend in backends]
    (out / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    table = markdown(results)
    (out / "report.md").write_text(table)
    print(table, end="")
    print(f"\nWAVs and the report are in {out}/")


if __name__ == "__main__":
    main()
