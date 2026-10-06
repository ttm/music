"""Level controls and review artifacts of the optional timing experiment."""
import json
import sys
import types

import numpy as np
import pytest

from tools import compare_singing_timing as comparison


def test_matching_preserves_equal_rms_when_one_recording_has_sharp_peaks():
    soft = np.full(1000, 0.02)
    sharp = np.zeros(1000)
    sharp[0] = 1.0
    matched, gains = comparison.level_match([soft, sharp])
    assert comparison.rms(matched[0]) == pytest.approx(
        comparison.rms(matched[1]))
    assert np.max(np.abs(matched[1])) == pytest.approx(0.98)
    assert gains[0]["shared_attenuation"] < 1
    for original, output, gain in zip([soft, sharp], matched, gains):
        np.testing.assert_allclose(output, original * gain["gain"])
        assert gain["matched_rms"] == pytest.approx(gain["shared_rms"])


def test_matching_retains_stereo_balance_and_handles_silence():
    stereo = np.array([[0.1, -0.1], [0.05, -0.05]])
    matched, gains = comparison.level_match([stereo, np.zeros(8)])
    np.testing.assert_allclose(matched[0][0], 2 * matched[0][1])
    assert comparison.rms(matched[0]) == pytest.approx(0.1)
    assert gains[1]["silent"]
    np.testing.assert_array_equal(matched[1], np.zeros(8))
    assert np.isfinite(gains[1]["gain_db"])
    assert comparison.level_match([]) == ([], [])


@pytest.mark.parametrize("sound", [[np.nan], [np.inf]])
def test_matching_rejects_nonfinite_audio(sound):
    with pytest.raises(ValueError, match="non-finite"):
        comparison.level_match([sound])


@pytest.mark.parametrize("options", [
    {"target_rms": 0}, {"target_rms": np.nan},
    {"ceiling": 0}, {"ceiling": 1.1},
])
def test_matching_rejects_invalid_level_controls(options):
    with pytest.raises(ValueError):
        comparison.level_match([[0.1]], **options)


def test_metrics_report_unmeasured_notes_without_hiding_them():
    score = dict(text="la la", notes=(0, 7), durs=(1, 1))
    row = comparison.metrics(np.zeros(44100), score, "silent", "vowel", 0)
    assert row["unmeasured_notes"] == 2
    assert row["cents_by_note"] == [None, None]
    assert row["max_abs_cents"] is None
    assert row["length_error_ms"] == 0


@pytest.mark.parametrize("effect", ["tremolo", "melt"])
def test_metrics_allow_the_effects_intended_reverberation_tail(effect):
    score = dict(text="la", notes=(0,), durs=(2,), effect=effect)
    row = comparison.metrics(np.zeros((2, 88200)), score, "tail", "vowel", 0)
    assert row["length_error_ms"] == 0
    assert row["expected_tail_seconds"] == 1


def test_review_page_escapes_lyrics_names_and_audio_attributes():
    name = "<script>bad()</script>|score"
    row = dict(score=name, backend="baseline", max_abs_cents=None,
               median_abs_cents=None, unmeasured_notes=1,
               length_error_ms=0, render_seconds=0,
               matched_file="matched/it's <a>.wav")
    report = dict(scores={name: {"text": "<img src=x>"}}, results=[row],
                  environment={"speaker": "<speaker>"})
    page = comparison.review_html(report)
    assert "<script>" not in page
    assert "<img src=x>" not in page
    assert "&lt;script&gt;" in page
    assert "it&#x27;s &lt;a&gt;.wav" in page
    assert "&lt;speaker&gt;" in page
    assert "slowed is the psola backend's own timing" in page
    assert "\\|score" in comparison.markdown(report)


def test_comparison_writes_each_render_once_and_preserves_level_matching(
        tmp_path, monkeypatch):
    import soundfile
    import music.singing.perform as perform

    calls = []
    score = dict(text="la", notes=(0,), durs=(1,))
    signal = np.sin(2 * np.pi * 130.81 * np.arange(22050) / 44100)

    def render(score, variant, speaker):
        calls.append((variant, speaker))
        return signal * (0.01 if variant == "baseline" else 0.8), [
            {"syllable": "la", "variant": variant}]

    fake = types.SimpleNamespace(TIMINGS=("baseline", "combined"),
                                 render=render)
    monkeypatch.setitem(sys.modules, "tools.singing_timing", fake)
    monkeypatch.setattr(comparison, "_version", lambda command: "version")
    monkeypatch.setattr(comparison, "_revision", lambda: "test-revision")
    seeded = []
    monkeypatch.setattr(comparison, "_seed_praat", lambda: seeded.append(42))

    def sing(**kwargs):
        calls.append((kwargs["backend"], None))
        return signal * 0.5

    monkeypatch.setattr(perform, "sing", sing)
    report = comparison.compare(tmp_path, {"one": score}, "/bin/espeak",
                                include_ecantorix=True)
    assert calls == [("baseline", "/bin/espeak"),
                     ("combined", "/bin/espeak"), ("ecantorix", None)]
    assert seeded == [42, 42]
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["package_timing"] == "slowed"
    assert saved["environment"]["git_head"] == "test-revision"
    levels = []
    for row in report["results"]:
        raw, rate = soundfile.read(tmp_path / row["raw_file"])
        matched, matched_rate = soundfile.read(tmp_path / row["matched_file"])
        assert rate == matched_rate == 44100
        levels.append(comparison.rms(matched))
        assert np.abs(matched).max() <= 0.98
        np.testing.assert_allclose(
            matched, raw * row["level_matching"]["gain"], atol=1 / 32768)
        assert row["unmeasured_notes"] == 0
    np.testing.assert_allclose(levels, [0.1] * 3, atol=1 / 32768)
    assert (tmp_path / "index.html").is_file()
    assert (tmp_path / "report.md").is_file()
