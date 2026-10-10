#!/usr/bin/env python3
"""Coherent-bin alias/leakage and CPU/memory diagnostics for rich oscillators.

This metric is **not** a perceptual score and cannot detect aliases that
coincide with permitted partials. Compare independent high-rate references
for nonstationary pitch, nonlinear effects or fast FM; the bin mask here
is valid only for a precisely periodic one-second static-pitch signal.
"""
import argparse
import json
from time import perf_counter
import tracemalloc

import numpy as np

from music.core.synths.frequency_path import bandlimited_frequency_path
from music.core.synths.polyblep import polyblep_frequency_path


def coherent_fraction(audio, fundamental_hz, sample_rate, waveform):
    """Energy outside DC and true Nyquist-admissible harmonic FFT bins."""
    signal = np.asarray(audio, dtype=float)
    if signal.shape != (sample_rate,):
        raise ValueError("required coherent one-second mono signal")
    if (not 0 < fundamental_hz < sample_rate / 2
            or waveform not in ("sawtooth", "square")):
        raise ValueError("unsupported frequency or waveform")
    power = abs(np.fft.rfft(signal)) ** 2
    mask = np.ones(len(power), dtype=bool)
    mask[0] = False
    for n in range(1, int((sample_rate / 2) // fundamental_hz) + 1):
        if waveform == "sawtooth" or n % 2:
            mask[n * fundamental_hz] = False
    return float(power[mask].sum() / power.sum())


def measure(waveform, rate=48000, repeats=3):
    """Compare a discontinuous oscillator, PolyBLEP and additive harmonic."""
    if waveform not in ("sawtooth", "square") or rate not in (
            44100, 48000, 96000) or repeats < 1:
        raise ValueError("invalid benchmark waveform, rate or repeats")
    f = int(round(rate * .20835))
    frequencies = np.full(rate, f, dtype=float)

    def naive():
        phase = (np.arange(rate) * f / rate) % 1.
        if waveform == "sawtooth":
            return 2 * phase - 1
        return np.where(phase < .5, 1., -1.)

    methods = {
        "naive": naive,
        "polyblep": lambda: polyblep_frequency_path(
            frequencies, sample_rate=rate, waveform=waveform),
        "additive": lambda: bandlimited_frequency_path(
            frequencies, sample_rate=rate, waveform=waveform, transition=0),
    }
    rows = []
    for name, render in methods.items():
        render()
        tracemalloc.start()
        try:
            result = render()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        times = []
        for _ in range(repeats):
            start = perf_counter()
            render()
            times.append(1000 * (perf_counter() - start))
        rows.append({
            "method": name, "waveform": waveform,
            "rate_hz": rate, "carrier_hz": f,
            "unwanted_energy_fraction": coherent_fraction(
                result, f, rate, waveform),
            "rms": float(np.sqrt(np.mean(result * result))),
            "render_median_ms": float(np.median(times)),
            "traced_peak_kib": int(np.ceil(peak / 1024)),
            "output_bytes": int(result.nbytes),
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rates", nargs="+", type=int,
                        default=[44100, 48000, 96000])
    args = parser.parse_args(argv)
    rows = [item for rate in args.rates
            for shape in ("sawtooth", "square")
            for item in measure(shape, rate, args.repeats)]
    print("| wave | Hz | rate | engine | unwanted energy | ms | traced KiB |")
    print("|---|---:|---:|---|---:|---:|---:|")
    for row in rows:
        print("| {waveform} | {carrier_hz} | {rate_hz} | "
              "{method} | {unwanted_energy_fraction:.5g} | "
              "{render_median_ms:.3f} | "
              "{traced_peak_kib} |".format(**row))
    if args.json:
        from pathlib import Path
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
    return rows


if __name__ == "__main__":
    main()
