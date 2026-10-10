#!/usr/bin/env python3
"""Reference-based dynamic alias tests for actual MUSIC and analytic DSP.

A high-rate continuous-time numerical construction is low-pass filtered
before decimation. The target is the *band-limited* version of the modeled
signal, not its direct samples. No assumption is made that every legacy
MUSIC generator is phase-exact relative to that independent construction.

Run: python tools/benchmark_dynamic_aliasing.py --json alias-report.json
"""
import argparse
import json
from time import perf_counter

import numpy as np
from scipy.signal import resample_poly

import music

CASES = ("fm_wide", "fm_extreme", "nonlinear", "hard_gate")
RATES = (44100, 48000, 96000)


def _parameters(case, rate):
    """Physical frequencies remain fixed as the internal rate changes."""
    if case == "fm_wide":
        return rate * .32, rate * .15, rate * .032, 0.0
    if case == "fm_extreme":
        # The intended FM frequency exceeds target Nyquist some of the time.
        return rate * .42, rate * .16, rate * .031, 0.0
    if case == "nonlinear":
        return rate * .29, 0.0, 0.0, 8.0
    if case == "hard_gate":
        return rate * .34, .37, rate * .034, 0.0
    raise ValueError("unsupported benchmark case")


def analytic(case, *, target_rate, sample_rate, number_of_samples):
    """Analytic zero-phase source, sampled directly at requested rate."""
    fc, extent, fm, drive = _parameters(case, target_rate)
    t = np.arange(number_of_samples, dtype=float) / sample_rate
    if case.startswith("fm_"):
        # Instantaneous deviation = extent*sin(2*pi*fm*t), exactly
        # integrated without the discretisation or table lookup of MUSIC.
        phase = (2 * np.pi * fc * t
                 + extent / fm * (1 - np.cos(2 * np.pi * fm * t)))
        return np.sin(phase)
    tone = np.sin(2 * np.pi * fc * t)
    if case == "nonlinear":
        return np.tanh(drive * tone) / np.tanh(drive)
    return tone * ((t * fm) % 1 < extent)


def actual_music(case, *, target_rate, sample_rate, number_of_samples):
    """Current MUSIC generator; do not substitute analytic source values."""
    fc, extent, fm, drive = _parameters(case, target_rate)
    if case.startswith("fm_"):
        return music.frequency_modulation(
            carrier_freq=fc, frequency_deviation=extent,
            modulation_freq=fm, number_of_samples=number_of_samples,
            sample_rate=sample_rate)
    if case == "nonlinear":
        x = music.note(freq=fc, number_of_samples=number_of_samples,
                       sample_rate=sample_rate)
        return np.tanh(drive * x) / np.tanh(drive)
    return music.isochronic_tones(
        carrier_freq=fc, pulse_rate=fm, duty_cycle=extent,
        number_of_samples=number_of_samples, sample_rate=sample_rate)


def _rms(samples):
    return float(np.sqrt(np.mean(samples * samples)))


def _short_time_spectral_error(samples, reference):
    """Windowed magnitude-distance; phase insensitive, not pure alias power."""
    numerator = 0.0
    denominator = 0.0
    for x, y in zip(np.array_split(samples, 8),
                    np.array_split(reference, 8)):
        window = np.hanning(len(x))
        a = np.abs(np.fft.rfft(x * window))
        b = np.abs(np.fft.rfft(y * window))
        numerator += float(np.sum((a - b) ** 2))
        denominator += float(np.sum(b ** 2))
    return float(np.sqrt(numerator / denominator))


def evaluate(case, rate=48000, count=4096, *,
             reference_factor=16, factors=(4, 8), repeats=3):
    """Compare direct and oversampled sources against a high-rate reference.

    Reference is independent of MUSIC. Output contains absolute RMS error
    and worst 1/8-window RMS error; the latter captures short transients
    that can disappear inside a global FFT. No perceptual claim follows.

    The reference has finite bandwidth too. Check convergence against a
    higher reference factor when hard gates/nonlinearities dominate.
    """
    if case not in CASES or rate not in RATES:
        raise ValueError("unknown case or sample rate")
    if (not isinstance(count, int) or count < 8
            or not isinstance(reference_factor, int)
            or reference_factor < 4 or reference_factor > 64
            or count * reference_factor > 4_000_000):
        raise ValueError("invalid or excessive reference sample budget")
    if (not isinstance(repeats, int) or repeats < 1 or repeats > 30
            or any(not isinstance(f, int) or f < 2
                   or f >= reference_factor for f in factors)):
        raise ValueError("invalid oversampling factors or repeats")
    raw_ref = analytic(
        case, target_rate=rate, sample_rate=rate * reference_factor,
        number_of_samples=count * reference_factor)
    reference = resample_poly(
        raw_ref, 1, reference_factor, window=("kaiser", 8.6))

    rows = []
    for engine, producer in (("analytic", analytic),
                             ("MUSIC", actual_music)):
        for factor in (1, *factors):
            def synth():
                if factor == 1:
                    return producer(
                        case, target_rate=rate, sample_rate=rate,
                        number_of_samples=count)
                return music.render_oversampled(
                    lambda *, sample_rate, number_of_samples: producer(
                        case, target_rate=rate, sample_rate=sample_rate,
                        number_of_samples=number_of_samples),
                    sample_rate=rate, number_of_samples=count,
                    factor=factor)
            samples = np.asarray(synth(), dtype=float)
            if (samples.shape != reference.shape or
                    not np.isfinite(samples).all()):
                raise ValueError("benchmark renderer produced invalid audio")
            timings = []
            for _ in range(repeats):
                start = perf_counter()
                synth()
                timings.append((perf_counter() - start) * 1000)
            errors = samples - reference
            chunks = np.array_split(errors, 8)
            rows.append({
                "case": case, "sample_rate_hz": rate,
                "engine": engine, "oversampling_factor": factor,
                "reference_factor": reference_factor,
                "reference_rms": _rms(reference),
                "error_rms": _rms(errors),
                "max_window_error_rms": max(map(_rms, chunks)),
                "short_time_magnitude_error": (
                    _short_time_spectral_error(samples, reference)),
                "peak_absolute": float(np.max(np.abs(samples))),
                "render_median_ms": float(np.median(timings)),
                "nominal_intermediate_frames": count * factor,
                "nominal_audio_buffer_bytes": count * factor * 8,
            })
    return rows


def main(argv=None):
    """Print markdown + optionally machine-readable JSON benchmark rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json")
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-factor", type=int, default=16)
    parser.add_argument("--rates", type=int, nargs="+", default=list(RATES))
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    args = parser.parse_args(argv)
    rows = [row for rate in args.rates for case in args.cases
            for row in evaluate(case, rate, args.count,
                                reference_factor=args.reference_factor,
                                repeats=args.repeats)]
    print("| case | rate | engine | x | RMS error | "
          "worst window RMS | spectral mag. | ms |")
    print("|---|---:|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print("| {case} | {sample_rate_hz} | {engine} | "
              "{oversampling_factor} | {error_rms:.5g} | "
              "{max_window_error_rms:.5g} | "
              "{short_time_magnitude_error:.5g} | "
              "{render_median_ms:.2f} |".format(**row))
    if args.json:
        from pathlib import Path
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
    return rows


if __name__ == "__main__":
    main()
