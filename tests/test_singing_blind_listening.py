"""The listening evaluation must really hide identities and match levels."""

import csv
import json

import numpy as np
import pytest
import soundfile as sf

from tools.prepare_singing_listening import prepare


def test_blind_trial_reproducible_and_backend_key_separate(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    t = np.arange(8000) / 8000
    for name in ("mary", "quick"):
        sf.write(source / f"{name}.psola.wav",
                 np.sin(2 * np.pi * 200 * t) * .2, 8000)
        sf.write(source / f"{name}.ecantorix.wav",
                 np.sin(2 * np.pi * 220 * t) * .4, 8000)
    trial1 = prepare(source, tmp_path / "first", seed=91)
    trial2 = prepare(source, tmp_path / "second", seed=91)
    assert trial1 == trial2
    assert len(trial1) == 2
    reviewer = tmp_path / "first" / "reviewer"
    assert not list(reviewer.glob("*key*"))
    for i in range(1, 3):
        a, _ = sf.read(reviewer / f"trial-{i:02d}-A.wav")
        b, _ = sf.read(reviewer / f"trial-{i:02d}-B.wav")
        assert np.max(np.abs(a)) <= .951
        assert np.max(np.abs(b)) <= .951
        def rms(x):
            return np.sqrt(np.mean(x*x))
        assert rms(a) / rms(b) == pytest.approx(1, abs=.01)
    with (reviewer / "ratings.csv").open(newline="") as f:
        ratings = list(csv.DictReader(f))
    assert len(ratings) == 2
    assert all(not x["naturalness_preference_A_B_equal"] for x in ratings)
    key = json.loads(
        (tmp_path / "first" / "organizer" / "key.json").read_text())
    assert key["trials"] == trial1


def test_listening_preparation_fails_on_missing_pairs(tmp_path):
    with pytest.raises(ValueError, match="need paired"):
        prepare(tmp_path, tmp_path / "out")


def test_level_match_refuses_silence(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    sf.write(source / "silent.psola.wav", np.zeros(100), 8000)
    sf.write(source / "silent.ecantorix.wav", np.ones(100), 8000)
    with pytest.raises(ValueError, match="silent"):
        prepare(source, tmp_path / "out")


def test_recordings_must_be_mono_and_share_rate(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    sf.write(source / "duet.psola.wav", np.ones((100, 2)) * .1, 8000)
    sf.write(source / "duet.ecantorix.wav", np.ones(100) * .1, 8000)
    with pytest.raises(ValueError, match="mono"):
        prepare(source, tmp_path / "out")
    sf.write(source / "duet.psola.wav", np.ones(100) * .1, 44100)
    with pytest.raises(ValueError, match="sample rates"):
        prepare(source, tmp_path / "out")
