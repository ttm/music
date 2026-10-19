#!/usr/bin/env python3
"""Four oscillators for a rich wave, by folding, fidelity, time and memory.

A naive sawtooth or square, PolyBLEP, mipmapped tables and summed
partials, each rendered three ways:

- at a fixed whole-hertz pitch for exactly one second, so that every
  harmonic falls on a bin. The energy anywhere else is folding, or an
  oscillator's own error, and the magnitudes on the harmonics, against
  the Fourier series' own, say what the oscillator kept of the wave;
- along an exponential sweep, read in short windows: the energy away
  from each window's harmonics, and the difference from summed partials
  cut at the Nyquist frequency, the band-limited wave by definition;
- in blocks, for the two oscillators that keep their state between
  calls: the time each block takes against the time it lasts.

None of it says what a listener hears, and the times are one machine's.

Run: python tools/benchmark_oscillators.py --json oscillators.json
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import tracemalloc

import numpy as np

from music.core.synths.frequency_path import bandlimited_frequency_path
from music.core.synths.mipmap import (
    MipmapOscillator, mipmap_frequency_path, tables,
)
from music.core.synths.polyblep import (
    PolyBLEPOscillator, polyblep_frequency_path,
)

RATES = (44100, 48000, 96000)
WAVEFORMS = ("sawtooth", "square")
METHODS = ("naive", "polyblep", "mipmap", "additive")
# Pitches as fractions of the rate: 110, 440, 1762, 7042 and 10,001 Hz
# at 48 kHz.
PITCHES = (.00229, .00917, .0367, .1467, .20835)
BLOCKS = (64, 256, 1024, 4096)


def _naive(frequencies, rate, waveform):
    phase = np.r_[0., np.cumsum((frequencies[:-1] + frequencies[1:])
                                / (2 * rate))] % 1.
    if waveform == "sawtooth":
        # In the Fourier series' phase: zero, rising, at the start.
        return 2 * ((phase + .5) % 1.) - 1
    return np.where(phase < .5, 1., -1.)


def render(method, frequencies, rate, waveform):
    """One render of a frequency path, in the Fourier series' phase."""
    f = np.asarray(frequencies, dtype=float)
    if method == "naive":
        return _naive(f, rate, waveform)
    if method == "polyblep":
        return polyblep_frequency_path(
            f, sample_rate=rate, waveform=waveform,
            phase_cycles=.5 if waveform == "sawtooth" else 0.)
    if method == "mipmap":
        return mipmap_frequency_path(f, sample_rate=rate, waveform=waveform)
    if method == "additive":
        return bandlimited_frequency_path(
            f, sample_rate=rate, waveform=waveform, transition=0,
            max_harmonics=2048)
    raise ValueError("unknown oscillator")


def series(waveform, n):
    """The magnitude of the n-th partial's sine, from the Fourier series."""
    n = np.asarray(n)
    if waveform == "sawtooth":
        return 2 / (np.pi * n)
    return np.where(n % 2 == 1, 4 / (np.pi * n), 0.)


def _timed(call, repeats):
    times = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        times.append(1000 * (perf_counter() - start))
    return float(np.median(times))


def static(waveform, rate=48000, fraction=.20835, repeats=3):
    """Each oscillator at one fixed pitch, for exactly one second."""
    if (waveform not in WAVEFORMS or rate not in RATES
            or not 0 < fraction < .5 or repeats < 1):
        raise ValueError("invalid waveform, rate, pitch or repeats")
    hz = int(round(fraction * rate))
    path = np.full(rate, float(hz))
    harmonics = np.arange(1, (rate - 1) // 2 // hz + 1)
    harmonics = harmonics[harmonics * hz * 2 < rate]
    wanted = series(waveform, harmonics) / 2
    rows = []
    for method in METHODS:
        render(method, path, rate, waveform)
        tracemalloc.start()
        try:
            audio = render(method, path, rate, waveform)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        spectrum = np.fft.rfft(audio) / rate
        power = np.abs(spectrum) ** 2
        on = np.zeros(len(power), dtype=bool)
        on[harmonics * hz] = True
        kept = np.abs(spectrum[harmonics * hz])
        rows.append({
            "method": method, "waveform": waveform, "rate_hz": rate,
            "pitch_hz": hz,
            "unwanted_energy_fraction": float(
                power[~on].sum() / power.sum()),
            "harmonic_error": float(np.sqrt(
                np.sum((kept - wanted) ** 2) / np.sum(wanted ** 2))),
            "render_median_ms": _timed(
                lambda: render(method, path, rate, waveform), repeats),
            "traced_peak_kib": int(np.ceil(peak / 1024)),
            "table_kib": (sum(table.nbytes for table in tables(waveform))
                          // 1024 if method == "mipmap" else 0),
        })
    return rows


def sweep(waveform, rate=48000, seconds=2.0, window=2048):
    """Each oscillator along an exponential sweep, read in windows.

    The sweep runs from a five-hundredth of the rate to 45 % of it, which
    summed partials render within their bound of fifty million
    partial-samples at every rate here, two seconds at 96 kHz. In each
    window, half overlapping, a Blackman-Harris window keeps leakage 92 dB
    down, and every harmonic that sounds anywhere in the window is masked,
    across the band it sweeps and three bins either side. What is left is
    the energy away from the harmonics.
    """
    if (waveform not in WAVEFORMS or rate not in RATES
            or not .1 <= seconds <= 10 or window not in (1024, 2048, 4096)):
        raise ValueError("invalid waveform, rate, length or window")
    count = int(seconds * rate)
    path = np.geomspace(.002 * rate, .45 * rate, count)
    from scipy.signal.windows import blackmanharris
    taper = blackmanharris(window)
    renders = {method: render(method, path, rate, waveform)
               for method in METHODS}
    ideal = renders["additive"]
    rows = []
    for method, audio in renders.items():
        away = total = 0.
        for start in range(0, count - window + 1, window // 2):
            span = path[start:start + window]
            power = np.abs(np.fft.rfft(audio[start:start + window]
                                       * taper)) ** 2
            on = np.zeros(len(power), dtype=bool)
            low, high = span.min(), span.max()
            for n in range(1, int(rate / 2 // low) + 1):
                if waveform == "square" and n % 2 == 0:
                    continue
                first = max(int(n * low * window / rate) - 3, 0)
                last = int(min(n * high, rate / 2) * window / rate) + 4
                on[first:last] = True
            away += power[~on].sum()
            total += power.sum()
        rows.append({
            "method": method, "waveform": waveform, "rate_hz": rate,
            "from_hz": float(path[0]), "to_hz": float(path[-1]),
            "off_harmonic_fraction": float(away / total),
            "difference_from_cut_partials": float(
                np.sqrt(np.mean((audio - ideal) ** 2)
                        / np.mean(ideal ** 2))),
        })
    return rows


def stream(method, rate=48000, block=256, seconds=5.0):
    """One stateful oscillator, rendered in blocks of a fixed size.

    A vibrato of a semitone at 5 Hz around 440 Hz, in sawtooth. Each
    block's time is compared with the time it lasts; the blocks joined
    are compared with one render of the whole path.
    """
    oscillators = {"polyblep": PolyBLEPOscillator,
                   "mipmap": MipmapOscillator}
    if (method not in oscillators or rate not in RATES
            or block not in BLOCKS or not .5 <= seconds <= 60):
        raise ValueError("invalid oscillator, rate, block or length")
    count = int(seconds * rate)
    t = np.arange(count) / rate
    path = 440 * 2 ** (np.sin(2 * np.pi * 5 * t) / 12)
    whole = oscillators[method](sample_rate=rate).render(path)
    oscillator = oscillators[method](sample_rate=rate)
    pieces, times = [], []
    for start in range(0, count, block):
        begin = perf_counter()
        pieces.append(oscillator.render(path[start:start + block]))
        times.append(1e6 * (perf_counter() - begin))
    probe = oscillators[method](sample_rate=rate)
    probe.render(path[:block])
    tracemalloc.start()
    try:
        probe.render(path[block:2 * block])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    budget = 1e6 * block / rate
    times = np.array(times)
    return {
        "method": method, "rate_hz": rate, "block": block,
        "budget_us": budget,
        "median_us": float(np.median(times)),
        "p99_us": float(np.percentile(times, 99)),
        "max_us": float(times.max()),
        "median_load": float(np.median(times) / budget),
        "blocks_over_budget": int(np.sum(times > budget)),
        "blocks": len(times),
        "traced_block_kib": int(np.ceil(peak / 1024)),
        "join_error": float(np.max(np.abs(np.concatenate(pieces) - whole))),
    }


def main(argv=None):
    """Print three Markdown tables; optionally write the rows as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rates", nargs="+", type=int, default=list(RATES))
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="length of each streamed render")
    args = parser.parse_args(argv)
    report = {
        "static": [row for rate in args.rates for shape in WAVEFORMS
                   for fraction in PITCHES
                   for row in static(shape, rate, fraction, args.repeats)],
        "sweep": [row for rate in args.rates for shape in WAVEFORMS
                  for row in sweep(shape, rate)],
        "stream": [stream(method, rate, block, args.seconds)
                   for rate in args.rates for method in ("polyblep", "mipmap")
                   for block in BLOCKS],
    }
    print("| wave | Hz | rate | oscillator | unwanted energy | "
          "harmonic error | ms | traced KiB | table KiB |")
    print("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for row in report["static"]:
        print("| {waveform} | {pitch_hz} | {rate_hz} | {method} | "
              "{unwanted_energy_fraction:.3g} | {harmonic_error:.3g} | "
              "{render_median_ms:.2f} | {traced_peak_kib} | "
              "{table_kib} |".format(**row))
    print("\n| wave | rate | oscillator | off-harmonic | "
          "from cut partials |")
    print("|---|---:|---|---:|---:|")
    for row in report["sweep"]:
        print("| {waveform} | {rate_hz} | {method} | "
              "{off_harmonic_fraction:.3g} | "
              "{difference_from_cut_partials:.3g} |".format(**row))
    print("\n| oscillator | rate | block | budget us | median us | p99 us | "
          "max us | over budget | traced KiB | join error |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["stream"]:
        print("| {method} | {rate_hz} | {block} | {budget_us:.0f} | "
              "{median_us:.0f} | {p99_us:.0f} | {max_us:.0f} | "
              "{blocks_over_budget}/{blocks} | {traced_block_kib} | "
              "{join_error:.1e} |".format(**row))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main()
