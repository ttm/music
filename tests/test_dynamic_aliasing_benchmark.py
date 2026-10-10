"""High-rate, short-time comparison to an independently integrated signal."""

import numpy as np
import pytest

from tools.benchmark_dynamic_aliasing import (
    CASES, RATES, actual_music, analytic, evaluate, main,
)


@pytest.mark.parametrize("case", CASES)
def test_oversampling_reduces_analytic_reference_error(case):
    """A finite-rate decimator must not reintroduce direct-sample folding."""
    data = evaluate(case, rate=48000, count=1024,
                    reference_factor=16, factors=(4, 8), repeats=1)
    reference_rows = {x["oversampling_factor"]: x for x in data
                      if x["engine"] == "analytic"}
    direct = reference_rows[1]["error_rms"]
    four = reference_rows[4]["error_rms"]
    eight = reference_rows[8]["error_rms"]
    assert direct > 0
    assert four < direct / 2
    assert eight < four
    assert all(x["reference_rms"] > 0 and
               x["max_window_error_rms"] >= x["error_rms"] - 1e-12
               for x in data)


@pytest.mark.parametrize("rate", RATES)
def test_fm_high_rate_reference_matrix(rate):
    rows = evaluate("fm_wide", rate=rate, count=1024,
                    reference_factor=16, factors=(4,), repeats=1)
    assert len(rows) == 4
    assert {x["engine"] for x in rows} == {"MUSIC", "analytic"}
    assert {x["sample_rate_hz"] for x in rows} == {rate}
    assert all(np.isfinite(x["error_rms"]) for x in rows)


def test_analytic_and_music_are_separate_renderers():
    a = analytic("fm_wide", target_rate=48000, sample_rate=48000,
                 number_of_samples=300)
    b = actual_music("fm_wide", target_rate=48000, sample_rate=48000,
                     number_of_samples=300)
    assert a.shape == b.shape
    assert not np.array_equal(a, b)


def test_bound_execution_and_invalid_cases():
    with pytest.raises(ValueError, match="unsupported"):
        analytic("missing", target_rate=48000, sample_rate=48000,
                 number_of_samples=100)
    for kwargs in (
            {"case": "x"},
            {"rate": 47000},
            {"count": 0},
            {"reference_factor": 2},
            {"count": 4_000_001},
            {"repeats": 0},
            {"factors": (1,)},
            {"factors": (16,)},
            {"factors": (2.1,)},
            {"reference_factor": 65}):
        arguments = dict(case="fm_wide", rate=48000, count=100,
                         repeats=1)
        arguments.update(kwargs)
        with pytest.raises(ValueError):
            evaluate(**arguments)


def test_cli_writes_machine_readable_and_markdown(tmp_path, capsys):
    target = tmp_path / "alias.json"
    result = main(["--json", str(target), "--cases", "fm_wide",
                   "--rates", "48000", "--count", "512",
                   "--repeats", "1"])
    assert len(result) == 6
    assert target.is_file()
    assert "RMS error" in capsys.readouterr().out
