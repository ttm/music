#!/usr/bin/env python3
"""Cross-check four SSTIM-named stimulus techniques against analytic signals.

This is NOT an RDF parser, SSTIM conformance test, or independent library.
The second renderer uses closed-form NumPy sinusoids rather than MUSIC LUTs,
so it can detect wrong frequencies, modulation, gating and channel shapes.
It cannot test assumptions shared by the two formulations.
"""
import argparse
import json

import numpy as np

import music

RATE = 44100
DURATION = 1.0

# The technique IRIs are from the published SSTIM vocabulary and the
# corresponding music.stimulation docstrings, not an exported SSTIM graph.
CASES = (
    ("techBinauralBeats", "binaural_beats",
     {"carrier_freq": 200.0, "beat_freq": 10.0}),
    ("techMonauralBeats", "monaural_beats",
     {"carrier_freq": 200.0, "beat_freq": 10.0}),
    ("techIsochronicTones", "isochronic_tones",
     {"carrier_freq": 200.0, "pulse_rate": 10.0, "duty_cycle": 0.5}),
    ("techAmplitudeModulation", "amplitude_modulation",
     {"carrier_freq": 200.0, "modulation_freq": 10.0,
      "modulation_depth": 0.75}),
)


def analytic(technique, params, rate=RATE, duration=DURATION):
    """Independent continuous-sinusoid rendering sampled in discrete time."""
    t = np.arange(int(duration * rate)) / rate
    carrier = params["carrier_freq"]
    if technique == "techBinauralBeats":
        half = params["beat_freq"] / 2
        return np.vstack((np.sin(2 * np.pi * (carrier - half) * t),
                          np.sin(2 * np.pi * (carrier + half) * t)))
    if technique == "techMonauralBeats":
        half = params["beat_freq"] / 2
        return (np.sin(2 * np.pi * (carrier - half) * t)
                + np.sin(2 * np.pi * (carrier + half) * t)) / 2
    tone = np.sin(2 * np.pi * carrier * t)
    if technique == "techIsochronicTones":
        gate = ((t * params["pulse_rate"]) % 1 < params["duty_cycle"])
        return tone * gate
    if technique == "techAmplitudeModulation":
        wave = np.sin(2 * np.pi * params["modulation_freq"] * t)
        depth = params["modulation_depth"]
        return tone * (1 - depth * (1 - wave) / 2)
    raise ValueError(f"unknown technique {technique!r}")


def compare(technique, implementation, parameters):
    """Numerical comparison and a machine-readable stimulus description."""
    expected = analytic(technique, parameters)
    actual = getattr(music, implementation)(**parameters, duration=DURATION,
                                           sample_rate=RATE)
    difference = np.asarray(actual) - expected
    return {
        "technique_iri": "https://w3id.org/sstim/vocab#" + technique,
        "implementation": implementation,
        "parameters": parameters,
        "sample_rate_hz": RATE,
        "duration_seconds": DURATION,
        "channels": int(1 if actual.ndim == 1 else actual.shape[0]),
        "rms_difference": float(np.sqrt(np.mean(difference ** 2))),
        "max_absolute_difference": float(np.max(np.abs(difference))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="Write JSON results to this file")
    args = parser.parse_args()
    report = [compare(*case) for case in CASES]
    print("| technique | channels | RMS error | maximum error |")
    print("|---|---:|---:|---:|")
    for row in report:
        print("| {technique_iri} | {channels} | {rms_difference:.5g} | "
              "{max_absolute_difference:.5g} |".format(**row))
    if args.json:
        from pathlib import Path
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main()
