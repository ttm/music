#!/usr/bin/env python3
"""Prepare blinded, level-matched waveform pairs; do not score listeners.

Run: python tools/prepare_polyblep_listening.py --out blep-listening
Only share the reviewer subfolder; organizer/key.json reveals identities.
Use a low playback level initially: near-Nyquist rich waves may sound harsh.
"""
import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import soundfile as sf

from music.core.synths.polyblep import polyblep_frequency_path

RATES = (44100, 48000, 96000)


def _match(a, b):
    """RMS-match without clipping or independent loudness cues."""
    levels = [float(np.sqrt(np.mean(x ** 2))) for x in (a, b)]
    if (min(levels) <= 0 or
            not all(np.isfinite(x).all() for x in (a, b))):
        raise ValueError("both sounds must be finite and nonsilent")
    target = min(levels)
    pair = [a * (target / levels[0]), b * (target / levels[1])]
    peak = max(float(np.max(np.abs(x))) for x in pair)
    headroom = min(1., .9 / peak)
    return tuple(x * headroom for x in pair)


def prepare(out, *, seed=0, rates=RATES):
    """Write paired PCM24 audio, reviewer form and segregated organizer key."""
    if (not isinstance(seed, int) or isinstance(seed, bool)
            or any(rate not in RATES for rate in rates)):
        raise ValueError("invalid seed or sample rate")
    directory = Path(out)
    reviewer = directory / "reviewer"
    organizer = directory / "organizer"
    reviewer.mkdir(parents=True, exist_ok=True)
    organizer.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    # Keep A/B identities balanced over the small evaluation set, while
    # randomizing which condition appears first on each individual trial.
    swaps = [False, True] * len(rates)
    rng.shuffle(swaps)
    pairs = []
    form = []
    for rate in rates:
        frequency = int(round(rate * .20835))
        phase = (np.arange(rate) * frequency / rate) % 1
        for waveform in ("sawtooth", "square"):
            naive = (2 * phase - 1 if waveform == "sawtooth"
                     else np.where(phase < .5, 1., -1.))
            blep = polyblep_frequency_path(
                np.full(rate, frequency), sample_rate=rate,
                waveform=waveform)
            a, b = _match(naive, blep)
            identities = ("naive", "polyblep")
            if swaps[len(pairs)]:
                a, b = b, a
                identities = ("polyblep", "naive")
            trial = f"trial-{len(pairs) + 1:02d}"
            sf.write(reviewer / f"{trial}-A.wav", a, rate,
                     subtype="PCM_24")
            sf.write(reviewer / f"{trial}-B.wav", b, rate,
                     subtype="PCM_24")
            pairs.append({
                "trial": trial, "waveform": waveform,
                "frequency_hz": frequency, "sample_rate": rate,
                "A": identities[0], "B": identities[1]})
            form.append({
                "trial": trial, "timbre_preference_A_B_equal": "",
                "artifact_salience_A_B_equal": "",
                "comments": ""})
    with (reviewer / "ratings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(form[0]))
        writer.writeheader()
        writer.writerows(form)
    (reviewer / "instructions.txt").write_text(
        "Use low volume to start; some trials may be harsh. "
        "Choose A, B or equal for each criterion independently. "
        "Do not view the organizer key until answers are recorded. "
        "No perceptual findings are implied by these files.\n")
    (organizer / "key.json").write_text(
        json.dumps({"seed": seed, "pairs": pairs}, indent=2) + "\n")
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="polyblep-listening")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    pairs = prepare(args.out, seed=args.seed)
    print(f"Prepared {len(pairs)} matched pairs; share only "
          f"{args.out}/reviewer/")


if __name__ == "__main__":
    main()
