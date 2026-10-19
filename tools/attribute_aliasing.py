#!/usr/bin/env python3
"""Attribute a render's folded energy to the partials that fold.

The four signals of ``benchmark_dynamic_aliasing.py``, and its gate
with ramped edges, have line spectra known in closed form. Frequency
modulation's sidebands are Bessel functions of its index; a tanh
waveshaper's harmonics are a Fourier series, computed here to machine
precision; a gated carrier's sidebands are a pulse train's series, a
rectangular one's or a trapezoidal one's. Here every line falls on a
whole hertz, so one second holds a whole number of periods and its DFT
puts each line on a bin. Where a line above the Nyquist frequency lands
is then arithmetic: the tool lists the lines that fold, where each lands
and with what power, and reads the package's render at those bins.

A render's energy is split three ways: on the bins of the band-limited
signal, on the bins only folded lines reach, and anywhere else, which is
neither the signal nor its folding -- a table's quantization, say. The
frequencies are chosen so that no folded line within 80 dB of the signal
shares a bin with it, and what folded power does is reported, since no
measurement can tell it from the signal.

An oversampled render is predicted as exactly. The closed form is
sampled at the higher rate, multiplied by the response of the filter
``render_oversampled`` uses, and folded down. What the filter takes from
the band-limited signal near the Nyquist frequency is reported too,
since less folding bought by suppressing the signal would not be less
folding.

Run: python tools/attribute_aliasing.py --json attribution.json
"""
import argparse
import json
from math import gcd
from pathlib import Path

import numpy as np
from scipy.special import jv

import music
from music.core.synths.oversampling import _decimation_filter
from music.utils import WAVEFORM_SINE

CASES = ("fm_wide", "fm_extreme", "nonlinear", "hard_gate", "ramped_gate")
RATES = (44100, 48000, 96000)
FACTORS = (1, 2, 4, 8, 16)

# benchmark_dynamic_aliasing's fractions of the sample rate: the carrier,
# then the deviation, drive or duty, then the modulation or pulse rate.
_NOMINAL = {
    "fm_wide": (.32, .15, .032),
    "fm_extreme": (.42, .16, .031),
    "nonlinear": (.29, 8.0, None),
    "hard_gate": (.34, .37, .034),
    "ramped_gate": (.34, .37, .034),
}
# The ramped gate's rise and fall, in samples at the rate: like the
# frequencies, a fixed fraction of the rate's own scale.
_RAMP_SAMPLES = 3
# Samples dropped at each end of a render: the filter reaches 65.
_PAD = 128
# Power, relative to the band-limited signal's, that may share a bin.
_SEPARATION = 1e-8
# Power below which a bin holds nothing, relative to the same.
_EMPTY = 1e-20


def parameters(case, rate):
    """Whole-hertz frequencies near the benchmark's, chosen to separate.

    The modulation or pulse rate is the first whole number at or above
    the nominal one with no factor of 2, 3, 5 or 7, so that it shares
    none with any rate used here, or any multiple of one. A pulse then
    starts exactly on a sample only at whole seconds, where the carrier
    is at zero, and FM's folded sidebands fall between its own. The
    carrier is the first at which no folded line within 80 dB of the
    band-limited signal lands on one of its bins.
    """
    if case not in CASES or rate not in RATES:
        raise ValueError("unknown case or sample rate")
    carrier, second, third = _NOMINAL[case]
    if third is not None:
        modulation = int(np.ceil(third * rate))
        while gcd(modulation, 210) != 1:
            modulation += 1
    for hz in range(int(np.ceil(carrier * rate)), rate // 2):
        if case.startswith("fm_"):
            chosen = {"carrier_hz": hz, "deviation_hz": second * rate,
                      "modulation_hz": modulation}
        elif case == "nonlinear":
            chosen = {"carrier_hz": hz, "drive": second}
        else:
            # A third of a sample off a whole one, so that no sample at
            # 1x, 2x, 4x, 8x or 16x falls on a falling edge, where a sampled
            # gate and its Fourier series disagree.
            chosen = {"carrier_hz": hz, "pulse_hz": modulation,
                      "duty": (round(second * rate) + 1 / 3) / rate}
            if case == "ramped_gate":
                chosen["ramp_s"] = _RAMP_SAMPLES / rate
        if _separated(case, chosen, rate):
            return chosen
    raise ValueError("no separating carrier")  # pragma: no cover


def lines(case, chosen, *, reach=4000):
    """The closed form's lines: frequencies in Hz, amplitudes, labels.

    A real signal's lines come in pairs, at f and -f with conjugate
    amplitudes; both are listed. A gate's series has no end, and only
    the pulse harmonics up to ``reach`` are: beyond 4000, each line is
    more than 80 dB below the signal.
    """
    fc = chosen["carrier_hz"]
    if case.startswith("fm_"):
        fm = chosen["modulation_hz"]
        index = chosen["deviation_hz"] / fm
        n = np.arange(-int(index) - 60, int(index) + 61)
        # sin(2 pi fc t + index (1 - cos 2 pi fm t)), by Jacobi-Anger.
        amplitude = np.exp(1j * index) * (-1j) ** n * jv(n, index) / 2j
        frequency = fc + n * fm
        labels = [f"sideband {k:+d}" for k in n]
    elif case == "nonlinear":
        period = 1 << 16
        theta = 2 * np.pi * np.arange(period) / period
        shaped = np.tanh(chosen["drive"] * np.sin(theta))
        coefficients = -2 * np.fft.rfft(shaped).imag / period
        n = np.arange(1, 400, 2)
        amplitude = (coefficients[n] / np.tanh(chosen["drive"])) / 2j
        frequency = n * fc
        labels = [f"harmonic {k}" for k in n]
    else:
        fp, duty = chosen["pulse_hz"], chosen["duty"]
        k = np.arange(-reach, reach + 1)
        nonzero = np.where(k == 0, 1, k)
        # A trapezoid is a rectangle as long as its plateau and one ramp,
        # smeared by a box as long as the ramp.
        ramp = chosen.get("ramp_s", 0.0) * fp
        if 2 * ramp > duty:
            raise ValueError("a gate's ramps overlap, and it is no trapezoid")
        width = duty - ramp
        series = np.where(
            k == 0, width,
            (1 - np.exp(-2j * np.pi * k * width)) / (2j * np.pi * nonzero))
        if ramp:
            series = series * np.where(
                k == 0, 1, (1 - np.exp(-2j * np.pi * k * ramp))
                / (2j * np.pi * nonzero * ramp))
        # sin(2 pi fc t) g(t): each pulse harmonic beside each of the
        # carrier's two lines.
        amplitude = np.concatenate((series / 2j, -series / 2j))
        frequency = np.concatenate((fc + k * fp, -fc + k * fp))
        labels = ([f"carrier {j:+d}·pulse" for j in k]
                  + [f"-carrier {j:+d}·pulse" for j in k])
        return frequency, amplitude, labels
    # The other half of each pair.
    return (np.concatenate((frequency, -frequency)),
            np.concatenate((amplitude, np.conj(amplitude))),
            labels + [f"{label}, mirrored" for label in labels])


def _separated(case, chosen, rate):
    """Whether no strong folded line shares a bin with the signal."""
    frequency, amplitude, _ = lines(case, chosen)
    power = np.abs(amplitude) ** 2
    inside = np.abs(frequency) * 2 < rate
    reference = power[inside].sum()
    if np.any((np.abs(frequency) * 2 == rate) & (power > _EMPTY * reference)):
        return False
    occupied = np.zeros(rate, dtype=bool)
    occupied[frequency[inside & (power > _EMPTY * reference)] % rate] = True
    strong = ~inside & (power > _SEPARATION * reference)
    return not occupied[frequency[strong] % rate].any()


def analytic(case, chosen, *, number_of_samples, sample_rate):
    """The closed form at every sample, in exact integer phase."""
    n = np.arange(number_of_samples, dtype=np.int64)
    carrier = 2 * np.pi * ((n * chosen["carrier_hz"]) % sample_rate
                           / sample_rate)
    if case.startswith("fm_"):
        fm = chosen["modulation_hz"]
        index = chosen["deviation_hz"] / fm
        swing = 2 * np.pi * ((n * fm) % sample_rate / sample_rate)
        return np.sin(carrier + index * (1 - np.cos(swing)))
    if case == "nonlinear":
        drive = chosen["drive"]
        return np.tanh(drive * np.sin(carrier)) / np.tanh(drive)
    phase = (n * chosen["pulse_hz"]) % sample_rate / sample_rate
    gate = (phase < chosen["duty"]).astype(float)
    ramp = chosen.get("ramp_s", 0.0) * chosen["pulse_hz"]
    if ramp:
        edge = np.minimum(phase, chosen["duty"] - phase)
        gate *= np.clip(edge / ramp, 0, 1)
    return np.sin(carrier) * gate


def actual_music(case, chosen, *, number_of_samples, sample_rate):
    """The package's own render of the same signal."""
    fc = chosen["carrier_hz"]
    if case.startswith("fm_"):
        return music.frequency_modulation(
            carrier_freq=fc, frequency_deviation=chosen["deviation_hz"],
            modulation_freq=chosen["modulation_hz"],
            number_of_samples=number_of_samples, sample_rate=sample_rate)
    if case == "nonlinear":
        drive = chosen["drive"]
        tone = music.note(freq=fc, waveform_table=WAVEFORM_SINE,
                          number_of_samples=number_of_samples,
                          sample_rate=sample_rate)
        return np.tanh(drive * tone) / np.tanh(drive)
    # isochronic_tones truncates the ramp to whole samples, and a hair
    # over keeps rounding from taking one away at any factor.
    return music.isochronic_tones(
        carrier_freq=fc, pulse_rate=chosen["pulse_hz"],
        duty_cycle=chosen["duty"],
        ramp_duration=chosen.get("ramp_s", 0.0) * (1 + 1e-9),
        number_of_samples=number_of_samples, sample_rate=sample_rate)


def response(factor, rate):
    """The zero-phase response of render_oversampled's filter, per Hz.

    Read at every whole hertz of the higher rate: entry m is the gain at
    m Hz, or at m minus that rate above its Nyquist frequency.
    """
    high = factor * rate
    taps = _decimation_filter(factor)
    half = len(taps) // 2
    centred = np.zeros(high)
    centred[:len(taps)] = taps
    return np.fft.fft(np.roll(centred, -half)).real


def predict(case, chosen, rate, factor):
    """One second's spectra: the whole render's, and the signal's own.

    All three are complex amplitudes per Hz, the DFT over the number of
    samples. The first is what a render of the closed form leaves; the
    second is what the lines below the Nyquist frequency contribute to it,
    after the filter; the third is what they are before it.
    """
    high = factor * rate
    whole = np.fft.fft(analytic(case, chosen, number_of_samples=high,
                                sample_rate=high)) / high
    gain = response(factor, rate) if factor > 1 else np.ones(high)
    whole = (whole * gain).reshape(factor, rate).sum(axis=0)
    frequency, amplitude, _ = lines(case, chosen)
    inside = np.abs(frequency) * 2 < rate
    bins = frequency[inside] % rate
    signal = np.zeros(rate, dtype=complex)
    np.add.at(signal, bins, amplitude[inside] * gain[frequency[inside] % high])
    ideal = np.zeros(rate, dtype=complex)
    np.add.at(ideal, bins, amplitude[inside])
    return whole, signal, ideal


def _render(engine, case, chosen, rate, factor):
    """One second of a render, from the middle of a longer one."""
    producer = analytic if engine == "analytic" else actual_music

    def generator(*, number_of_samples, sample_rate):
        return producer(case, chosen, number_of_samples=number_of_samples,
                        sample_rate=sample_rate)

    count = rate + 2 * _PAD
    if factor == 1:
        audio = generator(number_of_samples=count, sample_rate=rate)
    else:
        audio = music.render_oversampled(
            generator, sample_rate=rate, number_of_samples=count,
            factor=factor)
    return np.asarray(audio, dtype=float)[_PAD:_PAD + rate]


def _db(power):
    return None if power <= 0 else float(10 * np.log10(power))


def attribute(case, rate=48000, factor=1, engine="MUSIC", *, top=5):
    """Split one render's energy, and name the lines that fold most."""
    if factor not in (1, 2, 4, 8, 16) or engine not in ("MUSIC", "analytic"):
        raise ValueError("unknown oversampling factor or engine")
    chosen = parameters(case, rate)
    whole, signal, ideal = predict(case, chosen, rate, factor)
    folded = whole - signal
    reference = float(np.sum(np.abs(ideal) ** 2))
    present = np.abs(ideal) ** 2 > _EMPTY * reference
    reached = ~present & (np.abs(folded) ** 2 > _EMPTY * reference)
    elsewhere = ~present & ~reached
    measured = np.fft.fft(_render(engine, case, chosen, rate, factor)) / rate
    power = np.abs(measured) ** 2

    frequency, amplitude, labels = lines(case, chosen)
    high = factor * rate
    gain = response(factor, rate) if factor > 1 else np.ones(high)
    outside = (frequency > 0) & (frequency * 2 > rate)
    strength = 2 * np.abs(amplitude * gain[frequency % high]) ** 2
    ranked = sorted(np.flatnonzero(outside),
                    key=lambda i: -strength[i])[:top]
    named = []
    for i in ranked:
        landing = int(frequency[i] % rate)
        named.append({
            "line": labels[i], "from_hz": int(frequency[i]),
            "to_hz": min(landing, rate - landing),
            "predicted_db": _db(strength[i] / reference),
            "measured_db": _db(2 * power[landing] / reference),
        })
    magnitude = np.abs(np.abs(measured[present]) - np.abs(ideal[present]))
    filtered = np.abs(np.abs(signal[present]) - np.abs(ideal[present]))
    return {
        "case": case, "sample_rate_hz": rate, "engine": engine,
        "oversampling_factor": factor, **chosen,
        "folded_db": _db(power[reached].sum() / reference),
        "folded_predicted_db": _db(
            np.sum(np.abs(folded[reached]) ** 2) / reference),
        "shared_db": _db(np.sum(np.abs(folded[present]) ** 2) / reference),
        "other_db": _db(power[elsewhere].sum() / reference),
        "signal_change_db": _db(np.sum(magnitude ** 2) / reference),
        "filter_change_db": _db(np.sum(filtered ** 2) / reference),
        "lines": named,
    }


def _cell(value):
    return "--" if value is None else f"{value:.1f}"


def main(argv=None):
    """Print Markdown tables, and optionally write the rows as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json")
    parser.add_argument("--rates", type=int, nargs="+", default=list(RATES))
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--factors", type=int, nargs="+",
                        default=list(FACTORS))
    parser.add_argument("--engines", nargs="+", default=["MUSIC", "analytic"])
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args(argv)
    rows = [attribute(case, rate, factor, engine, top=args.top)
            for rate in args.rates for case in args.cases
            for engine in args.engines for factor in args.factors]
    print("Power in dB relative to the band-limited signal's.\n")
    print("| case | rate | engine | x | folded | predicted | shared | "
          "other | signal change | of which filter |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print("| {case} | {sample_rate_hz} | {engine} | "
              "{oversampling_factor} | ".format(**row)
              + " | ".join(_cell(row[key]) for key in (
                  "folded_db", "folded_predicted_db", "shared_db",
                  "other_db", "signal_change_db", "filter_change_db"))
              + " |")
    print("\n| case | rate | engine | x | line | from Hz | to Hz | "
          "predicted | measured |")
    print("|---|---:|---|---:|---|---:|---:|---:|---:|")
    for row in rows:
        for line in row["lines"]:
            print(f"| {row['case']} | {row['sample_rate_hz']} | "
                  f"{row['engine']} | {row['oversampling_factor']} | "
                  f"{line['line']} | {line['from_hz']} | {line['to_hz']} | "
                  f"{_cell(line['predicted_db'])} | "
                  f"{_cell(line['measured_db'])} |")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
    return rows


if __name__ == "__main__":
    main()
