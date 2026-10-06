"""The timing experiment: preserved transitions, bounded fitting, pitch.

These checks establish signal properties, not improved intelligibility.
The optional real-engine cases also ensure the comparison's ``vibrato``
is the package's actual singer, including its effects and silent
syllables, and that it sings a note shorter than its syllable as
``baseline``, the timing it replaced, did.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from music.singing import psola
from tools import singing_timing as timing
from tools.compare_singing import (SCORES, cents, expected_pitches,
                                   expected_seconds, note_pitches)


@pytest.mark.parametrize("target, durations, chosen, tried, fastest", [
    (0.6, {175: 0.6}, 175, [175], 450),
    (0.6, {175: 0.3, 88: 0.6}, 88, [175, 88], 450),
    (4.0, {175: 0.3, 80: 0.7}, 80, [175, 80], 450),
    (0.01, {175: 0.3, 450: 0.2}, 450, [175, 450], 450),
    (0.6, {175: 0.3, 88: 0.75, 110: 0.7, 128: 0.8}, 110,
     [175, 88, 110, 128], 450),
    # Never faster than espeak's default: a syllable longer than its
    # note is said once, as the baseline says it.
    (0.01, {175: 0.3}, 175, [175], 175),
    (0.2, {175: 0.3}, 175, [175], 175),
    (0.6, {175: 0.3, 88: 0.6}, 88, [175, 88], 175),
    # 146 WPM comes out too long, and would ask for 219.
    (0.6, {175: 0.5, 146: 0.9}, 175, [175, 146], 175),
    (0.6, {175: 0.5, 146: 0.9, 219: 0.59}, 219, [175, 146, 219], 450),
])
def test_rate_search_is_bounded_and_keeps_closest_measured_syllable(
        monkeypatch, tmp_path, target, durations, chosen, tried, fastest):
    sounds = {rate: SimpleNamespace(get_total_duration=lambda d=d: d)
              for rate, d in durations.items()}
    calls = []

    def speak(program, syllable, voice, path, speed):
        assert (program, syllable, voice, path) == (
            "espeak", "lamb", "en", tmp_path / "note.wav")
        calls.append(speed)
        return sounds[speed]

    monkeypatch.setattr(psola, "_speak", speak)
    result, report = timing.fit_speech("espeak", "lamb", "en",
                                       tmp_path / "note.wav", target,
                                       fastest)
    assert result is sounds[chosen]
    assert calls == tried
    assert report["wpm"] == chosen
    assert report["attempts"] == [
        {"wpm": r, "seconds": durations[r]} for r in tried]


def test_an_explicit_speaker_does_not_need_a_speaker_on_path(monkeypatch):
    monkeypatch.setattr(timing.shutil, "which", lambda name: (
        name if name == "/custom/espeak" else None))
    monkeypatch.setattr(psola.importlib.util, "find_spec", lambda name: True)
    speakers = []
    monkeypatch.setattr(timing.perform, "sung_phonemes", lambda *args: (
        speakers.append(args[0]) or {}))
    sound, notes = timing.render(dict(text="", notes=(), durs=()),
                                 "combined", speaker="/custom/espeak")
    assert speakers == ["/custom/espeak"]
    assert sound.size == 0 and notes == []


REAL = pytest.mark.skipif(bool(psola.missing_requirements()),
                          reason="timing audio needs espeak and Parselmouth")
FLITE = pytest.mark.skipif(bool(psola.missing_requirements("flite")),
                           reason="timing audio needs flite and Parselmouth")


@REAL
@pytest.mark.parametrize("effect, lang", [
    (None, "en"), (None, "fr"), ("tremolo", "en"), ("melt", "en"),
    pytest.param("flite", "slt", marks=FLITE),
])
def test_experiment_vibrato_is_sample_exact_with_the_package(effect, lang):
    from parselmouth.praat import run

    # Short notes and long ones, so that some syllables are slowed and
    # held and some are not; the experiment finds its speeds by a search
    # of its own.
    score = dict(text="Jac-ques ! la", notes=(4, 0, 0, 7),
                 durs=(0.5, 1, 0.2, 2), effect=effect, lang=lang)
    # Praat uses random phase for unvoiced regions. Equal inputs only
    # give equal samples when its generator starts at the same place.
    try:
        run("random_initializeWithSeedUnsafelyButPredictably (42)")
        actual, notes = timing.render(score, "vibrato")
        run("random_initializeWithSeedUnsafelyButPredictably (42)")
        np.testing.assert_array_equal(actual, psola.sing(**score))
    finally:
        run("random_initializeSafelyAndUnpredictably ()")
    long_note = notes[-1]
    assert long_note["held_nucleus"] is not None
    assert (long_note["wpm"] is None) == (effect == "flite")
    if effect != "flite":
        assert long_note["wpm"] < psola.SPEED


@REAL
@pytest.mark.parametrize("lang", ["en", "fr"])
def test_slowed_sings_notes_shorter_than_their_syllables_as_the_baseline(
        lang):
    from parselmouth.praat import run

    # Each of these is said in more than 0.125 s at espeak's default
    # speed, so nothing is slowed and nothing held; and the vibrato sets
    # in later than that.
    score = dict(text="bright blue Jac-ques", notes=(0, 4, 7, 2),
                 durs=(0.25,) * 4, lang=lang)
    sung = {}
    try:
        for variant in ("slowed", "vibrato", "baseline"):
            run("random_initializeWithSeedUnsafelyButPredictably (42)")
            sung[variant], notes = timing.render(score, variant)
            if variant == "slowed":
                slowed_notes = notes
    finally:
        run("random_initializeSafelyAndUnpredictably ()")
    assert all(note["source_seconds"] > note["seconds"]
               for note in slowed_notes)
    assert [note["wpm"] for note in slowed_notes] == [175] * 4
    assert [note["nucleus"] for note in slowed_notes] == [None] * 4
    np.testing.assert_array_equal(sung["slowed"], sung["baseline"])
    np.testing.assert_array_equal(sung["vibrato"], sung["baseline"])


@REAL
@pytest.mark.parametrize("variant", ["speed", "vowel", "combined", "slowed",
                                     "vibrato", "envelope", "shaped"])
@pytest.mark.parametrize("score", [
    SCORES["mary"], SCORES["quick"], SCORES["test song"], SCORES["german"],
    dict(text="Jac-ques", notes=(4, 0), durs=(1, 1), lang="fr"),
])
def test_timing_candidates_keep_pitch_and_score_duration(variant, score):
    sound, diagnostics = timing.render(score, variant)
    seconds = expected_seconds(score)
    assert len(sound) == round(sum(seconds) * psola.RATE)
    pitches = note_pitches(sound, seconds)
    measured = [(p, e) for p, e in zip(pitches, expected_pitches(score))
                if p is not None]
    assert len(measured) >= len(pitches) - 1
    assert all(abs(cents(p, e)) < 35 for p, e in measured)
    assert np.isfinite(sound).all()
    assert len(diagnostics) == len(score["notes"])


@REAL
@pytest.mark.parametrize("variant", ["combined", "slowed"])
@pytest.mark.parametrize("effect", [None, "melt", "tremolo"])
def test_combined_long_note_stays_voiced_and_effect_timing_is_preserved(
        effect, variant):
    score = dict(text="lamb", notes=(9,), durs=(8,), effect=effect)
    sound, diagnostics = timing.render(score, variant)
    assert sound.shape[-1] == psola.RATE * (4 + bool(effect))
    mono = sound.mean(axis=0) if sound.ndim == 2 else sound
    early, late = mono[22050:44100], mono[132300:154350]
    assert np.sqrt(np.mean(late ** 2)) > 0.1 * np.sqrt(np.mean(early ** 2))
    pitch, = note_pitches(mono[:4 * psola.RATE], [4.0])
    assert abs(cents(pitch, 220.0)) < 35
    assert diagnostics[0]["held_nucleus"] is not None
    shift = 1.25 if effect == "melt" else 1
    assert diagnostics[0]["target_seconds"] == 4 * shift


@FLITE
def test_flite_bypasses_espeak_speed_search():
    _, notes = timing.render(dict(text="la", notes=(0,), durs=(2,),
                                  effect="flite", lang="slt"), "combined")
    assert notes[0]["attempts"] == []
    assert notes[0]["wpm"] is None


@REAL
@pytest.mark.parametrize("effect", [None, "melt"])
def test_the_vibrato_is_sung_at_its_rate_and_depth_on_a_held_vowel(effect):
    """Drawn where Praat reads the pitch, through the stretch, the vibrato
    keeps its 5.5 Hz on a vowel held many times as long as it was said,
    and through melt's resampling."""
    import parselmouth

    sound, notes = timing.render(dict(text="laa", notes=(9,), durs=(4,),
                                      effect=effect), "vibrato")
    assert notes[0]["held_nucleus"] is not None
    mono = sound.mean(axis=0) if sound.ndim == 2 else sound
    pitch = parselmouth.Sound(mono[:2 * psola.RATE], psola.RATE).to_pitch_ac(
        time_step=0.005, pitch_floor=100, pitch_ceiling=400)
    times, hertz = pitch.xs(), pitch.selected_array["frequency"]
    grown = (times > 0.6) & (times < 1.85) & (hertz > 0)
    off = 1200 * np.log2(hertz[grown] / 220)
    assert abs(off.mean()) < 3
    assert 30 < np.percentile(np.abs(off - off.mean()), 95) < 42
    turns = np.sum(np.diff(np.sign(off - off.mean())) > 0)
    assert turns / (times[grown][-1] - times[grown][0]) == pytest.approx(
        5.5, abs=0.3)


@REAL
def test_the_envelope_is_the_package_s_adsr_on_each_note():
    from parselmouth.praat import run

    score = SCORES["mary"]
    try:
        run("random_initializeWithSeedUnsafelyButPredictably (42)")
        plain, _ = timing.render(score, "slowed")
        run("random_initializeWithSeedUnsafelyButPredictably (42)")
        shaped, _ = timing.render(score, "envelope")
    finally:
        run("random_initializeSafelyAndUnpredictably ()")
    from music.core.filters.adsr import adsr

    edges = np.round(np.cumsum([0.0] + expected_seconds(score))
                     * psola.RATE).astype(int)
    envelope = np.concatenate([
        adsr(sonic_vector=np.ones(end - start), **timing.ENVELOPE)
        for start, end in zip(edges, edges[1:])])
    expected = plain * envelope
    np.testing.assert_allclose(shaped, expected / np.abs(expected).max(),
                               atol=1e-12)
