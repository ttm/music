"""Executable external SSTIM interoperability harness and safety limits."""

import json
from pathlib import Path

import pytest

pytest.importorskip("rdflib")
import tools.sstim_interop_check as harness


FIXTURES = Path(__file__).parent / "fixtures"
TURTLE = FIXTURES / "sstim_third_party_am.ttl"
SIDECAR = FIXTURES / "sstim_third_party_am.contract.json"


def test_third_party_turtle_cli_reports_stable_nonportable_pcm(tmp_path):
    first = harness.check(TURTLE, SIDECAR)
    second = harness.check(TURTLE, SIDECAR)
    assert first == second
    assert first["shape"] == [1920]
    assert first["generator"] == "amplitude_modulation"
    assert first["sstim_version"] == "0.19.0"
    assert first["sstim_full_profile_conforms"] is None
    assert first["exact_pcm_cross_vendor"] is False
    assert first["rms"] > 0
    assert len(first["render_sha256"]) == 64
    assert len(first["sidecar_sha256"]) == 64
    output = tmp_path / "render.wav"
    result = harness.check(TURTLE, SIDECAR, wav=output)
    assert result["render_sha256"] == first["render_sha256"]
    import soundfile as sf
    rendered, rate = sf.read(output)
    assert rate == 48000
    assert rendered.shape == (1920,)


def test_cli_json_report_and_console_output(tmp_path, capsys):
    output = tmp_path / "report.json"
    result = harness.main([
        str(TURTLE), str(SIDECAR), "--json", str(output)])
    assert json.loads(output.read_text()) == result
    assert json.loads(capsys.readouterr().out) == result


def test_sidecar_input_is_bounded_and_shape_checked(tmp_path):
    with pytest.raises(ValueError, match="existing local"):
        harness.check(TURTLE, tmp_path / "none.json")
    long_file = tmp_path / "long.json"
    long_file.write_text("x" * 8193)
    with pytest.raises(ValueError, match="8 KiB"):
        harness.check(TURTLE, long_file)
    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="invalid sidecar"):
        harness.check(TURTLE, invalid)
    invalid.write_text("{bad json}")
    with pytest.raises(ValueError, match="invalid sidecar"):
        harness.check(TURTLE, invalid)
    for data in ([], {"generator": "amplitude_modulation"},
                 {"generator": "amplitude_modulation",
                  "parameters": [], "duration": .04,
                  "sample_rate": 48000}):
        invalid.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="sidecar shape"):
            harness.check(TURTLE, invalid)


def test_opt_in_full_profile_validation_can_gate_rendering(monkeypatch):
    class Result:
        def __init__(self, ok):
            self.ok = ok

    monkeypatch.setattr(
        harness, "validate_sstim", lambda *a, **k: Result(True))
    report = harness.check(TURTLE, SIDECAR, official=True)
    assert report["sstim_full_profile_conforms"] is True
    monkeypatch.setattr(
        harness, "validate_sstim", lambda *a, **k: Result(False))
    with pytest.raises(ValueError, match="Full-profile"):
        harness.check(TURTLE, SIDECAR, official=True)


def test_invalid_output_from_engine_is_rejected(monkeypatch):
    class Dummy:
        pass
    monkeypatch.setattr(harness, "render_sstim_contract",
                        lambda *args: [float("nan")])
    with pytest.raises(ValueError, match="invalid samples"):
        harness.check(TURTLE, SIDECAR)


@pytest.mark.parametrize("stem,shape", [
    ("sstim_third_party_am", [1920]),
    ("sstim_third_party_noise", [1920]),
    ("sstim_third_party_spatial", [2, 1920]),
])
def test_three_independently_authored_external_profiles(stem, shape):
    turtle = FIXTURES / (stem + ".ttl")
    contract = FIXTURES / (stem + ".contract.json")
    result = harness.check(turtle, contract)
    assert result["shape"] == shape
    assert result["exact_pcm_cross_vendor"] is False
    assert harness.check(turtle, contract)["render_sha256"] == (
        result["render_sha256"])


@pytest.mark.network
@pytest.mark.skipif(__import__("os").environ.get("SSTIM_LIVE") != "1",
                    reason="pinned Full profile on Python 3.12 only")
def test_three_independently_authored_external_profiles_pass_official():
    for stem in ("sstim_third_party_am", "sstim_third_party_noise",
                 "sstim_third_party_spatial"):
        result = harness.check(
            FIXTURES / (stem + ".ttl"),
            FIXTURES / (stem + ".contract.json"),
            official=True)
        assert result["sstim_full_profile_conforms"] is True
