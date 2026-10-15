"""The tool that renders another tool's description, and its limits."""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("rdflib")
import tools.sstim_interop_check as harness  # noqa: E402
from music.stimulation.sstim import to_sstim_graph  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
TURTLE = FIXTURES / "sstim_third_party_am.ttl"
ARGUMENTS = FIXTURES / "sstim_third_party_am.render.json"


def test_the_report_is_stable_and_claims_no_portable_samples(tmp_path):
    first = harness.check(TURTLE, ARGUMENTS)
    assert first == harness.check(TURTLE, ARGUMENTS)
    assert first["shape"] == [1920]
    assert first["generator"] == "amplitude_modulation"
    assert first["rendered_by"] == "music.amplitude_modulation"
    assert first["sstim_version"] == "0.19.0"
    assert first["sstim_full_profile_conforms"] is None
    assert first["exact_pcm_cross_vendor"] is False
    assert first["rms"] > 0
    assert len(first["render_sha256"]) == 64
    assert len(first["arguments_sha256"]) == 64
    output = tmp_path / "render.wav"
    assert harness.check(TURTLE, ARGUMENTS, wav=output)["render_sha256"] == (
        first["render_sha256"])
    import soundfile as sf
    rendered, rate = sf.read(output)
    assert (rendered.shape, rate) == ((1920,), 48000)


def test_the_report_is_printed_and_written_as_json(tmp_path, capsys):
    output = tmp_path / "report.json"
    result = harness.main([str(TURTLE), str(ARGUMENTS), "--json",
                           str(output)])
    assert json.loads(output.read_text()) == result
    assert json.loads(capsys.readouterr().out) == result


def test_the_arguments_are_bounded_and_checked(tmp_path):
    with pytest.raises(ValueError, match="existing local JSON"):
        harness.check(TURTLE, tmp_path / "none.json")
    long_file = tmp_path / "long.json"
    long_file.write_text("x" * 8193)
    with pytest.raises(ValueError, match="under 8 KiB"):
        harness.check(TURTLE, long_file)
    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="not valid JSON"):
        harness.check(TURTLE, invalid)
    invalid.write_text("{bad json}")
    with pytest.raises(ValueError, match="not valid JSON"):
        harness.check(TURTLE, invalid)
    for data in ([], {"generator": "amplitude_modulation"},
                 {"sample_rate": 48000, "controls": []}):
        invalid.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="must be an object of"):
            harness.check(TURTLE, invalid)


def test_validation_is_opt_in_and_gates_the_render(monkeypatch):
    class Result:
        def __init__(self, ok):
            self.ok = ok

    monkeypatch.setattr(harness, "validate_sstim",
                        lambda *a, **k: Result(True))
    assert harness.check(TURTLE, ARGUMENTS, official=True)[
        "sstim_full_profile_conforms"] is True
    monkeypatch.setattr(harness, "validate_sstim",
                        lambda *a, **k: Result(False))
    with pytest.raises(ValueError, match="does not conform"):
        harness.check(TURTLE, ARGUMENTS, official=True)


def test_invalid_samples_are_refused(monkeypatch):
    monkeypatch.setattr(harness, "render_sstim",
                        lambda *args, **kwargs: [float("nan")])
    with pytest.raises(ValueError, match="invalid samples"):
        harness.check(TURTLE, ARGUMENTS)


def test_a_beat_renders_by_its_profile(tmp_path):
    turtle = tmp_path / "beat.ttl"
    to_sstim_graph("binaural_beats", duration=.01,
                   sample_rate=48000).serialize(turtle, format="turtle")
    arguments = tmp_path / "beat.json"
    arguments.write_text(json.dumps(
        {"profile": "zero-phase-equal-gain-beats-v1"}))
    report = harness.check(turtle, arguments)
    assert report["rendered_by"] == "profile zero-phase-equal-gain-beats-v1"
    assert report["shape"] == [2, 480]


def test_a_session_is_not_this_tools(tmp_path):
    import music
    session = music.StimulationSession(sample_rate=8000)
    session.add(music.binaural_beats, duration=.1)
    turtle = tmp_path / "session.ttl"
    to_sstim_graph(session).serialize(turtle, format="turtle")
    arguments = tmp_path / "none.json"
    arguments.write_text("{}")
    with pytest.raises(ValueError, match="single stimulus"):
        harness.check(turtle, arguments)


@pytest.mark.parametrize("stem,shape", [
    ("sstim_third_party_am", [1920]),
    ("sstim_third_party_noise", [1920]),
    ("sstim_third_party_spatial", [2, 1920]),
])
def test_three_independently_written_descriptions(stem, shape):
    turtle = FIXTURES / (stem + ".ttl")
    arguments = FIXTURES / (stem + ".render.json")
    result = harness.check(turtle, arguments)
    assert result["shape"] == shape
    assert harness.check(turtle, arguments)["render_sha256"] == (
        result["render_sha256"])


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SSTIM_LIVE") != "1",
                    reason="opt-in live SSTIM validation on one CI runner")
def test_the_three_conform_to_sstims_full_profile():
    for stem in ("sstim_third_party_am", "sstim_third_party_noise",
                 "sstim_third_party_spatial"):
        result = harness.check(FIXTURES / (stem + ".ttl"),
                               FIXTURES / (stem + ".render.json"),
                               official=True)
        assert result["sstim_full_profile_conforms"] is True
