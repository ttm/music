"""The benchmark itself must measure the regression it describes."""

from tools.benchmark_spectral_aliasing import measure


def test_spectral_benchmark_reports_aliasing_and_render_time():
    mass, filtered = measure(10000, "sawtooth", repeats=1)
    assert mass["method"] == "MASS LUT"
    assert filtered["method"] == "band-limited"
    assert mass["off_harmonic_energy_fraction"] > .20
    assert filtered["off_harmonic_energy_fraction"] < 1e-6
    assert mass["render_median_ms"] >= 0
    assert filtered["render_median_ms"] >= 0
    assert mass["frequency_hz"] == filtered["frequency_hz"] == 10000
