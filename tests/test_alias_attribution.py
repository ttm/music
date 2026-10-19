"""The attribution must account for every folded line, and only those."""

from math import gcd

import numpy as np
import pytest

pytest.importorskip("scipy")

from tools.attribute_aliasing import (  # noqa: E402
    CASES, RATES, analytic, attribute, lines, main, parameters, predict,
    response,
)


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("case", CASES)
def test_frequencies_are_whole_and_keep_the_folding_apart(case, rate):
    chosen = parameters(case, rate)
    assert isinstance(chosen["carrier_hz"], int)
    assert chosen["carrier_hz"] * 2 < rate
    for key in ("modulation_hz", "pulse_hz"):
        if key in chosen:
            assert gcd(chosen[key], 210) == 1
    frequency, amplitude, _ = lines(case, chosen)
    power = np.abs(amplitude) ** 2
    inside = np.abs(frequency) * 2 < rate
    reference = power[inside].sum()
    signal = set(frequency[inside & (power > 1e-20 * reference)] % rate)
    strong = ~inside & (power > 1e-8 * reference)
    assert not signal & set(frequency[strong] % rate)


@pytest.mark.parametrize("case", ("fm_wide", "fm_extreme", "nonlinear"))
def test_finite_line_spectra_are_the_closed_form_sampled(case):
    """Every line, folded where sampling puts it, and nothing else."""
    rate = 48000
    chosen = parameters(case, rate)
    sampled = np.fft.fft(analytic(case, chosen, number_of_samples=rate,
                                  sample_rate=rate)) / rate
    frequency, amplitude, _ = lines(case, chosen)
    placed = np.zeros(rate, dtype=complex)
    np.add.at(placed, frequency % rate, amplitude)
    assert np.max(np.abs(sampled - placed)) < 1e-12


@pytest.mark.parametrize("case", ("hard_gate", "ramped_gate"))
def test_a_gate_s_series_is_its_sampled_closed_form_below_nyquist(case):
    """A gate's series has no end: its lines below Nyquist must match."""
    rate = 48000
    chosen = parameters(case, rate)
    _, signal, ideal = predict(case, chosen, rate, 1)
    np.testing.assert_array_equal(signal, ideal)
    whole = np.fft.fft(analytic(case, chosen, number_of_samples=rate,
                                sample_rate=rate)) / rate
    present = np.abs(ideal) > 0
    # What differs is what folds onto the signal's bins, which is small.
    assert np.sum(np.abs(whole - ideal)[present] ** 2) < 1e-7


def test_the_response_is_render_oversampled_s_filter():
    """A line's gain, read off a render, is the response at its Hz."""
    import music

    rate, factor = 48000, 4
    gain = response(factor, rate)
    for hz in (1000, 21000, 21600, 23000, 24000, 30000, 95000):
        def tone(*, number_of_samples, sample_rate, hz=hz):
            n = np.arange(number_of_samples)
            return np.cos(2 * np.pi * (n * hz % sample_rate) / sample_rate)
        out = music.render_oversampled(tone, sample_rate=rate,
                                       number_of_samples=rate + 256,
                                       factor=factor)[128:128 + rate]
        landing = hz % rate
        measured = 2 * abs(np.fft.fft(out)[landing]) / rate
        if landing in (0, rate // 2):
            measured /= 2
        assert measured == pytest.approx(abs(gain[hz]), abs=1e-9)
    assert gain[21600] == pytest.approx(1, abs=2e-5)
    assert abs(gain[24000]) < 1.1e-5


@pytest.mark.parametrize("factor", (1, 4))
@pytest.mark.parametrize("case", CASES)
def test_the_closed_form_s_render_is_predicted_exactly(case, factor):
    row = attribute(case, 48000, factor, "analytic", top=3)
    assert row["folded_db"] == pytest.approx(row["folded_predicted_db"],
                                             abs=0.01)
    for line in row["lines"]:
        assert line["measured_db"] == pytest.approx(line["predicted_db"],
                                                    abs=0.05)


@pytest.mark.parametrize("case", CASES)
def test_the_package_s_render_folds_as_predicted(case):
    row = attribute(case, 48000, 1, "MUSIC", top=1)
    assert row["folded_db"] == pytest.approx(row["folded_predicted_db"],
                                             abs=0.2)
    line, = row["lines"]
    assert line["measured_db"] == pytest.approx(line["predicted_db"],
                                                abs=0.2)


def test_oversampling_removes_fm_s_folding_and_names_what_it_costs():
    direct = attribute("fm_wide", 48000, 1, "MUSIC", top=1)
    four = attribute("fm_wide", 48000, 4, "MUSIC", top=1)
    assert direct["folded_db"] > -20
    assert four["folded_predicted_db"] < -130
    assert four["folded_db"] < -110
    # Its sidebands end below three times the Nyquist frequency: 2x will do.
    two = attribute("fm_wide", 48000, 2, "analytic", top=0)
    assert two["folded_db"] < -130
    # The top tenth of the band goes with it, and is reported.
    assert direct["filter_change_db"] is None
    assert -20 < four["filter_change_db"] < -10


def test_a_hard_gate_folds_at_the_higher_rate_and_a_ramp_cuts_it():
    hard = [attribute("hard_gate", 48000, x, "analytic", top=0)["folded_db"]
            for x in (4, 8)]
    ramped = [attribute("ramped_gate", 48000, x, "analytic",
                        top=0)["folded_db"] for x in (4, 8)]
    # Each doubling halves the hard gate's tail, whose lines fall as 1/k.
    assert hard[0] - hard[1] == pytest.approx(6, abs=0.5)
    assert ramped[0] - ramped[1] > 10
    assert ramped[0] < hard[0] - 30


def test_the_first_lines_named_are_the_ones_that_fold_most():
    row = attribute("fm_extreme", 48000, 1, "analytic", top=3)
    named = [line["line"] for line in row["lines"]]
    assert named == ["sideband +4", "sideband +3", "sideband +5"]
    first = row["lines"][0]
    assert first["from_hz"] == parameters("fm_extreme", 48000)[
        "carrier_hz"] + 4 * parameters("fm_extreme", 48000)["modulation_hz"]
    assert first["to_hz"] == 48000 - first["from_hz"]


def test_refusals():
    with pytest.raises(ValueError, match="unknown case"):
        parameters("missing", 48000)
    with pytest.raises(ValueError, match="unknown case"):
        parameters("fm_wide", 47000)
    with pytest.raises(ValueError, match="factor or engine"):
        attribute("fm_wide", 48000, 3)
    with pytest.raises(ValueError, match="factor or engine"):
        attribute("fm_wide", 48000, 1, "other")
    chosen = dict(parameters("ramped_gate", 48000))
    chosen["ramp_s"] = chosen["duty"] / chosen["pulse_hz"]
    with pytest.raises(ValueError, match="overlap"):
        lines("ramped_gate", chosen)


def test_the_cli_writes_both_tables_and_json(tmp_path, capsys):
    target = tmp_path / "attribution.json"
    rows = main(["--json", str(target), "--rates", "48000", "--cases",
                 "nonlinear", "--factors", "1", "4", "--engines", "MUSIC",
                 "--top", "2"])
    assert len(rows) == 2 and len(rows[0]["lines"]) == 2
    assert target.is_file()
    out = capsys.readouterr().out
    assert "| nonlinear | 48000 | MUSIC | 4 |" in out
    assert "harmonic 3" in out
