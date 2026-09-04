import warnings

import numpy as np
import pytest

import music
from music import utils
from music.core import functions

def test_db_amp_conversion():
    values = np.array([-12, -6, 0, 6, 12])
    amps = utils.db_to_amp(values)
    back = utils.amp_to_db(amps)
    assert np.allclose(back, values)


def test_hz_midi_conversion():
    freqs = np.array([220.0, 440.0, 880.0])
    midis = utils.hz_to_midi(freqs)
    back = utils.midi_to_hz(midis)
    assert np.allclose(back, freqs)


def test_horizontal_stack_and_convert_to_stereo():
    m1 = np.arange(4)
    m2 = np.arange(4) + 4
    stereo = np.vstack((np.arange(4), np.arange(4) + 10))
    stacked = utils.horizontal_stack(m1, stereo, m2)
    assert stacked.shape == (2, 12)
    conv = utils.convert_to_stereo(m1)
    assert conv.shape == (2, 4)
    assert np.allclose(conv[0], m1)
    multi = np.vstack((m1, m1 + 10, m1 + 20))
    conv_multi = utils.convert_to_stereo(multi)
    expected = np.vstack((multi[0] + multi[2], multi[1] + multi[2]))
    assert np.allclose(conv_multi, expected)


def test_mix_and_normalize():
    a = np.ones(5)
    b = np.arange(3)
    mixed = utils.mix(a, b)
    expected = a.copy()
    expected[:3] += b
    assert np.allclose(mixed, expected)
    norm = functions.normalize_mono(mixed)
    assert np.max(norm) <= 1 and np.min(norm) >= -1


def test_mix_many_basic():
    a = np.array([1, 1, 1])
    b = np.array([1, 2])
    mixed = utils.mix_many([a, b])
    assert np.allclose(mixed, np.array([2, 3, 1]))


def test_mix_many_offset_and_end():
    a = np.array([1, 1])
    b = np.array([1, 1, 1])
    out = utils.mix_many([a, b], end=True)
    assert np.allclose(out, np.array([1, 2, 2]))

    out_offset = utils.mix_many([a, b], offset=[0, 1], sample_rate=1)
    assert np.allclose(out_offset, np.array([1, 2, 1, 1]))


def test_hz_to_midi_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        utils.hz_to_midi(np.array([0.0, 440.0]))


def test_profile_sorts_a_namespace_by_what_the_names_hold():
    """It was exported, documented as returning a dictionary, and had a
    fully commented-out body, so every call quietly returned None; then it
    raised, which said the same thing where a caller could hear it. The
    design in its docstring was a specification. It is now a description.
    """
    summary = utils.profile({
        "sound": music.note(duration=0.1),
        "freqs": np.array([220.0, 440.0]),
        "durations": [1, 2],
        "options": {"a": 1},
        "seen": {1, 2},
        "count": 3,
        "label": "x",
        "writer": music.write_wav_mono,
    })

    assert sorted(summary["type"]["scalar"]) == ["count", "label"]
    assert sorted(summary["type"]["collections"]) == [
        "durations", "freqs", "options", "seen", "sound"]
    assert summary["type"]["other"] == ["writer"]
    # Only the arrays get measured.
    assert sorted(summary["analyses"]["ndarray"]) == ["freqs", "sound"]


def test_profile_measures_each_array_against_the_sample_rate():
    sound = music.note(freq=440, duration=0.5, sample_rate=44100)
    measured = utils.profile({"s": sound})["analyses"]["ndarray"]["s"]

    assert measured["samples"] == len(sound)
    assert measured["seconds"] == pytest.approx(0.5)
    assert measured["shape"] == sound.shape
    assert measured["rms"] == pytest.approx(
        float(np.sqrt(np.mean(sound ** 2))))
    assert measured["mean"] == pytest.approx(float(sound.mean()))
    assert measured["minimum"] == pytest.approx(float(sound.min()))
    assert measured["maximum"] == pytest.approx(float(sound.max()))
    # A steady tone has the same RMS in every block.
    assert measured["block_rms_std"] < 1e-3

    halved = utils.profile({"s": sound}, sample_rate=22050)
    assert halved["analyses"]["ndarray"]["s"]["seconds"] == pytest.approx(1.0)


def test_profile_hears_the_discontinuity_a_steady_tone_does_not_have():
    """The spread of the block RMS is what the specification asked for."""
    steady = music.note(duration=0.5)
    broken = np.hstack([steady[:len(steady) // 2],
                        np.zeros(len(steady) // 2)])

    measured = utils.profile({"a": steady, "b": broken})["analyses"]["ndarray"]
    # A tone whose blocks do not divide into whole periods has a little
    # spread; a sound that stops halfway has an order of magnitude more.
    assert measured["a"]["block_rms_std"] < 0.02 * measured["a"]["rms"]
    assert measured["b"]["block_rms_std"] > 10 * measured["a"]["block_rms_std"]


def test_profile_reads_a_long_centred_bounded_array_as_pcm():
    guesses = dict(utils.profile({"s": music.note(duration=0.2)})["guesses"])
    readings = [reading for reading, _reason in guesses["s"]]
    assert "pcm samples" in readings
    # Every guess states what produced it.
    for _reading, reason in guesses["s"]:
        assert reason


def test_profile_reads_a_short_offset_array_as_parametrisation():
    guesses = utils.profile({"p": np.array([2.0, 3.0, 5.0])})["guesses"]["p"]
    assert "parametrisation" in [reading for reading, _ in guesses]


def test_profile_reads_large_values_as_frequencies():
    guesses = utils.profile({"f": np.array([220.0, 440.0, 880.0])})
    readings = [r for r, _ in guesses["guesses"]["f"]]
    assert "frequencies in Hz" in readings


def test_profile_tells_pitches_from_decibels_by_how_far_they_step():
    pitches = utils.profile({"m": np.array([60.0, 62.0, 64.0, 65.0])})
    decibels = utils.profile({"d": np.array([0.0, 90.0, 5.0, 120.0,
                                             10.0, 100.0, 3.0, 130.0])})
    assert "MIDI pitches or semitone intervals" in [
        r for r, _ in pitches["guesses"]["m"]]
    assert "decibels" in [r for r, _ in decibels["guesses"]["d"]]


def test_profile_reads_a_power_of_two_bound_as_pcm():
    """16-bit PCM read as integers is centred and bounded by 2**15."""
    samples = (music.note(duration=0.2) * (2 ** 15 - 1)).astype(np.int16)
    readings = [r for r, _ in utils.profile({"s": samples})["guesses"]["s"]]
    assert "pcm samples" in readings


@pytest.mark.parametrize("value, expect_numeric", [
    (np.array([]), False),
    (np.array(["a", "b"]), False),
    (np.array([1.0, 2.0]), True),
])
def test_profile_measures_what_it_can_and_says_when_it_cannot(
        value, expect_numeric):
    measured = utils.profile({"v": value})["analyses"]["ndarray"]["v"]
    assert measured["numeric"] is expect_numeric
    if not expect_numeric:
        assert "rms" not in measured
        assert utils.profile({"v": value})["guesses"]["v"] == []


def test_profile_survives_an_array_with_no_finite_values():
    measured = utils.profile(
        {"v": np.array([np.nan, np.inf])})["analyses"]["ndarray"]["v"]
    assert measured["numeric"] is True
    assert measured["finite"] is False
    assert "rms" not in measured
    assert utils.profile({"v": np.array([np.nan])})["guesses"]["v"] == []


def test_profile_of_an_empty_namespace_is_empty():
    summary = utils.profile({})
    assert summary["type"] == {"scalar": [], "collections": [], "other": []}
    assert summary["analyses"]["ndarray"] == {}
    assert summary["guesses"] == {}
