#!/usr/bin/env python3
"""Reproducible spectral and runtime comparison for static-pitch oscillators.

FFT bins are exact here because each frequency is an integer number of
cycles in one second. Off-harmonic energy is a numerical alias proxy, not
a perceptual score; changes in alias energy do not establish preference.
Run: python tools/benchmark_spectral_aliasing.py [--json report.json]
"""
import argparse
import json
from time import perf_counter

import numpy as np

import music


def leakage(samples, frequency, sample_rate):
    """Fraction of spectral energy outside the permitted harmonics."""
    power = abs(np.fft.rfft(samples)) ** 2
    bins = np.arange(len(power))
    fundamental_bin = int(frequency * len(samples) / sample_rate)
    allowed = (bins > 0) & (bins % fundamental_bin == 0)
    return float(power[~allowed].sum() / power.sum())


def measure(frequency, waveform, sample_rate=44100, repeats=3):
    """Compare the original MASS table against the opt-in filtered table."""
    legacy_table = music.waveform_table(waveform)
    methods = {
        "MASS LUT": lambda: music.note(
            frequency, duration=1, sample_rate=sample_rate,
            waveform_table=legacy_table),
        "band-limited": lambda: music.bandlimited_note(
            frequency, duration=1, sample_rate=sample_rate,
            waveform=waveform),
    }
    rows = []
    for name, synth in methods.items():
        synth()  # cache / warmup is deliberately excluded from timings
        measurements = []
        for _ in range(repeats):
            start = perf_counter()
            result = synth()
            measurements.append(perf_counter() - start)
        rows.append({
            "waveform": waveform,
            "frequency_hz": frequency,
            "sample_rate_hz": sample_rate,
            "method": name,
            "off_harmonic_energy_fraction": leakage(
                result, frequency, sample_rate),
            "render_median_ms": float(np.median(measurements) * 1000),
            "peak_absolute": float(np.max(np.abs(result))),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="Optional JSON output path")
    parser.add_argument("--repeats", type=int, default=3)
    options = parser.parse_args()
    if options.repeats < 1:
        parser.error("--repeats must be positive")
    report = [
        row for waveform in ("sawtooth", "square", "triangle", "sine")
        for frequency in (1000, 10000)
        for row in measure(frequency, waveform, repeats=options.repeats)
    ]
    print("| waveform | Hz | renderer | off-harmonic energy | "
          "median ms | peak |")
    print("|---|---:|---|---:|---:|---:|")
    for row in report:
        print("| {waveform} | {frequency_hz} | {method} | "
              "{off_harmonic_energy_fraction:.3g} | "
              "{render_median_ms:.2f} | {peak_absolute:.3f} |".format(**row))
    if options.json:
        from pathlib import Path
        Path(options.json).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main()
