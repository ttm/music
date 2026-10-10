#!/usr/bin/env python3
"""Prepare randomized, blinded comparisons of the two singing backends.

First run: python tools/compare_singing.py --out singing-comparison
Then: python tools/prepare_singing_listening.py singing-comparison listening

Give a listener ONLY listening/reviewer/, not listening/organizer/.
The CSV is a blank instrument, not a completed experiment. Listening is
still required; the existing ASR benchmark is not human perceptual evidence.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf


def _equal_level(a, b):
    """RMS-match, then apply joint headroom for any waveform overshoot.

    Reject empty and non-finite recordings rather than writing invalid
    material for the subsequent human listening comparison.
    """
    if any(not len(x) or not np.isfinite(x).all() for x in (a, b)):
        raise ValueError("listening recordings must be nonempty and finite")
    rms = [float(np.sqrt(np.mean(x ** 2))) for x in (a, b)]
    if min(rms) == 0:
        raise ValueError("cannot blind-compare a silent recording")
    target = min(rms)
    a, b = (a * target / rms[0], b * target / rms[1])
    peak = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    gain = min(1.0, 0.95 / peak)
    return a * gain, b * gain


def prepare(source, output, seed=0):
    """Create matched A/B files, a reviewer form and a separate key."""
    source = Path(source)
    output = Path(output)
    scores = sorted(path.name[:-len(".psola.wav")]
                    for path in source.glob("*.psola.wav")
                    if (source / path.name.replace(
                        ".psola.wav", ".ecantorix.wav")).exists())
    if not scores:
        raise ValueError("need paired *.psola.wav and *.ecantorix.wav")
    rng = random.Random(seed)
    reviewer = output / "reviewer"
    organizer = output / "organizer"
    reviewer.mkdir(parents=True, exist_ok=True)
    organizer.mkdir(parents=True, exist_ok=True)
    key = []
    form = []
    for number, name in enumerate(scores, 1):
        sounds = {}
        rate = None
        for backend in ("psola", "ecantorix"):
            samples, sample_rate = sf.read(
                source / f"{name}.{backend}.wav", always_2d=False)
            if samples.ndim != 1:
                raise ValueError(f"{name}: expected mono WAV")
            if rate is not None and rate != sample_rate:
                raise ValueError(f"{name}: sample rates do not agree")
            rate = sample_rate
            sounds[backend] = np.asarray(samples)
        a, b = _equal_level(sounds["psola"], sounds["ecantorix"])
        if rng.randrange(2):
            a, b = b, a
            ordered = ("ecantorix", "psola")
        else:
            ordered = ("psola", "ecantorix")
        trial = f"trial-{number:02d}"
        sf.write(reviewer / f"{trial}-A.wav", a, rate)
        sf.write(reviewer / f"{trial}-B.wav", b, rate)
        form.append({
            "trial": trial,
            "intelligibility_preference_A_B_equal": "",
            "naturalness_preference_A_B_equal": "",
            "comments": "",
        })
        key.append({"trial": trial, "score": name,
                    "A": ordered[0], "B": ordered[1]})
    fields = list(form[0])
    with (reviewer / "ratings.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(form)
    (reviewer / "instructions.txt").write_text(
        "Listen with matched playback settings, in a quiet environment.\n"
        "For each pair, choose A, B, or equal for intelligibility and\n"
        "naturalness independently. Record what you heard, not what you\n"
        "expected. The backend key is deliberately not in this folder.\n")
    (organizer / "key.json").write_text(
        json.dumps({"seed": seed, "trials": key}, indent=2) + "\n")
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", help="folder from compare_singing.py")
    parser.add_argument("output", help="new listening-trial folder")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    trials = prepare(args.source, args.output, args.seed)
    print(f"Prepared {len(trials)} randomized trials. Give reviewers ONLY "
          f"{args.output}/reviewer/.")


if __name__ == "__main__":
    main()
