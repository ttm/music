"""Blinded PolyBLEP-versus-naive listening stimulus preparation."""

import json

import numpy as np
import pytest
import soundfile as sf

from tools.prepare_polyblep_listening import _match, prepare


def test_prepared_pairs_have_matching_rms_without_exposing_key(tmp_path):
    first = prepare(tmp_path / "first", seed=42, rates=(48000,))
    second = prepare(tmp_path / "second", seed=42, rates=(48000,))
    assert first == second
    assert len(first) == 2
    review = tmp_path / "first" / "reviewer"
    assert not list(review.glob("*key*"))
    for trial in first:
        a, rate_a = sf.read(review / (trial["trial"] + "-A.wav"))
        b, rate_b = sf.read(review / (trial["trial"] + "-B.wav"))
        assert rate_a == rate_b == 48000
        assert a.shape == b.shape == (48000,)
        assert max(abs(a).max(), abs(b).max()) <= .901
        rms_a = float(np.sqrt(np.mean(a*a)))
        rms_b = float(np.sqrt(np.mean(b*b)))
        assert rms_a / rms_b == pytest.approx(1, abs=.001)
    recorded = json.loads(
        (tmp_path / "first" / "organizer" / "key.json").read_text())
    assert recorded["pairs"] == first
    assert "timbre_preference" in (review / "ratings.csv").read_text()
    assert "low volume" in (review / "instructions.txt").read_text()


@pytest.mark.parametrize("bad", [0., float("nan"), float("inf")])
def test_match_refuses_silence_or_invalid_samples(bad):
    with pytest.raises(ValueError, match="finite and nonsilent"):
        _match(np.array([bad, bad]), np.array([.2, -.2]))


@pytest.mark.parametrize("kwargs", [
    {"seed": True},
    {"seed": 1.5},
    {"rates": (22050,)},
])
def test_preparation_rejects_invalid_metadata(tmp_path, kwargs):
    with pytest.raises(ValueError, match="invalid seed"):
        prepare(tmp_path / "bad", **kwargs)
