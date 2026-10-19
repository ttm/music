"""The oscillator benchmark must measure what it says it measures."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from tools.benchmark_oscillators import (  # noqa: E402
    METHODS, main, render, series, static, stream, sweep,
)


def test_every_oscillator_renders_the_series_phase():
    """At a pitch where only the fundamental fits, all four agree."""
    rate = 48000
    path = np.full(480, 15000.)
    expected = 2 / np.pi * np.sin(2 * np.pi * 15000 * np.arange(480) / rate)
    for method in ("mipmap", "additive"):
        np.testing.assert_allclose(render(method, path, rate, "sawtooth"),
                                   expected, atol=1e-6)
    # The naive and PolyBLEP sawtooths keep their folded partials, but
    # cross zero rising where the series does.
    for method in ("naive", "polyblep"):
        out = render(method, np.full(480, 1000.), rate, "sawtooth")
        assert abs(out[0]) < 1e-9 and out[1] > 0
    with pytest.raises(ValueError, match="unknown oscillator"):
        render("other", path, rate, "sawtooth")


def test_the_series_magnitudes():
    assert series("sawtooth", 3) == pytest.approx(2 / (3 * np.pi))
    assert series("square", 3) == pytest.approx(4 / (3 * np.pi))
    assert series("square", 2) == 0


@pytest.mark.parametrize("waveform", ("sawtooth", "square"))
def test_a_fixed_pitch_ranks_what_folds_and_what_is_kept(waveform):
    rows = {row["method"]: row for row in static(waveform, 48000, repeats=1)}
    assert list(rows) == list(METHODS)
    assert {row["pitch_hz"] for row in rows.values()} == {10001}
    folded = {name: row["unwanted_energy_fraction"]
              for name, row in rows.items()}
    assert folded["naive"] > .15
    assert 1e-3 < folded["polyblep"] < .01
    assert folded["mipmap"] < 1e-12
    assert folded["additive"] < 1e-20
    assert rows["additive"]["harmonic_error"] < 1e-12
    assert rows["polyblep"]["harmonic_error"] > .05
    assert rows["mipmap"]["table_kib"] == 1792
    assert rows["polyblep"]["table_kib"] == 0
    assert all(row["render_median_ms"] >= 0 and row["traced_peak_kib"] > 0
               for row in rows.values())


def test_a_sweep_separates_folding_from_cutting():
    rows = {row["method"]: row for row in sweep("sawtooth", 48000,
                                                seconds=.5, window=1024)}
    assert rows["additive"]["difference_from_cut_partials"] == 0
    assert rows["naive"]["off_harmonic_fraction"] > .01
    assert (rows["mipmap"]["off_harmonic_fraction"]
            < rows["polyblep"]["off_harmonic_fraction"] / 1000)
    assert rows["mipmap"]["from_hz"] == pytest.approx(96)
    assert rows["mipmap"]["to_hz"] == pytest.approx(21600)


@pytest.mark.parametrize("method", ("polyblep", "mipmap"))
def test_a_stream_reports_its_blocks_against_their_budget(method):
    row = stream(method, 48000, 256, seconds=.5)
    assert row["blocks"] == 94
    assert row["budget_us"] == pytest.approx(1e6 * 256 / 48000)
    assert row["median_load"] == pytest.approx(
        row["median_us"] / row["budget_us"])
    assert 0 <= row["blocks_over_budget"] <= row["blocks"]
    assert row["median_us"] <= row["p99_us"] <= row["max_us"]
    assert row["join_error"] < 1e-8
    assert row["traced_block_kib"] > 0


def test_refusals():
    for call in (lambda: static("triangle"), lambda: static("square", 47000),
                 lambda: static("square", fraction=.5),
                 lambda: static("square", repeats=0),
                 lambda: sweep("triangle"), lambda: sweep("square", 47000),
                 lambda: sweep("square", seconds=.05),
                 lambda: sweep("square", window=1000),
                 lambda: stream("additive"), lambda: stream("mipmap", 47000),
                 lambda: stream("mipmap", block=100),
                 lambda: stream("mipmap", seconds=.1)):
        with pytest.raises(ValueError):
            call()


def test_the_cli_prints_three_tables_and_writes_json(tmp_path, capsys,
                                                      monkeypatch):
    import tools.benchmark_oscillators as benchmark

    monkeypatch.setattr(benchmark, "PITCHES", (.20835,))
    monkeypatch.setattr(benchmark, "BLOCKS", (4096,))
    target = tmp_path / "oscillators.json"
    report = main(["--json", str(target), "--rates", "48000",
                   "--repeats", "1", "--seconds", "0.5"])
    assert len(report["static"]) == 8
    assert len(report["sweep"]) == 8
    assert len(report["stream"]) == 2
    assert target.is_file()
    out = capsys.readouterr().out
    assert "| sawtooth | 10001 | 48000 | mipmap |" in out
    assert "join error" in out
