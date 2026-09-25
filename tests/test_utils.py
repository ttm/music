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


def test_as_sonic_vector_promotes_integer_samples_to_float64():
    converted = utils.as_sonic_vector([1, -2])

    assert converted.dtype == np.float64
    np.testing.assert_array_equal(converted, [1.0, -2.0])


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


def test_convert_to_stereo_does_not_warn_for_an_existing_stereo_vector():
    stereo = np.array([[1.0, 2.0], [3.0, 4.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converted = utils.convert_to_stereo(stereo)

    np.testing.assert_array_equal(converted, stereo)


def test_convert_to_stereo_keeps_multichannel_two_sample_inputs():
    multichannel = np.array([[1, 2], [3, 4], [5, 6]])

    converted = utils.convert_to_stereo(multichannel)

    np.testing.assert_array_equal(converted, [[6, 8], [8, 10]])


def test_convert_to_stereo_sums_integer_pcm_without_overflow():
    multichannel = np.array([[20000], [30000], [10000]], dtype=np.int16)

    with pytest.warns(UserWarning):
        converted = utils.convert_to_stereo(multichannel)

    assert converted.dtype == np.float64
    np.testing.assert_array_equal(converted, [[30000], [40000]])


def test_mix_stereo_sums_integer_pcm_without_overflow():
    mono = np.array([20000, -20000], dtype=np.int16)

    converted = utils.mix_stereo(mono)

    assert converted.dtype == np.float64
    np.testing.assert_array_equal(
        converted, [[40000, -40000], [40000, -40000]])


def test_mix_and_normalize():
    a = np.ones(5)
    b = np.arange(3)
    mixed = utils.mix(a, b)
    expected = a.copy()
    expected[:3] += b
    assert np.allclose(mixed, expected)
    norm = functions.normalize_mono(mixed)
    assert np.max(norm) <= 1 and np.min(norm) >= -1


def test_mix_sums_when_the_second_sound_is_longer():
    mixed = utils.mix(np.array([1.0, 2.0]), np.array([10.0, 20.0, 30.0]))

    np.testing.assert_array_equal(mixed, [11.0, 22.0, 30.0])


def test_mix_does_not_modify_the_longer_input():
    first = np.array([1.0, 2.0])
    second = np.array([10.0, 20.0, 30.0])
    original = second.copy()

    utils.mix(first, second)

    np.testing.assert_array_equal(second, original)


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


def test_mix_many_converts_seconds_to_samples_at_the_given_rate():
    mixed = utils.mix_many([np.array([1.0]), np.array([2.0])],
                           offset=[0, .5], sample_rate=4)

    np.testing.assert_array_equal(mixed, [1.0, 0.0, 2.0])


def test_mix_many_uses_its_documented_default_sample_rate():
    mixed = utils.mix_many([np.array([1.0]), np.array([2.0])],
                           offset=[0, 1])

    assert len(mixed) == 44_101
    assert mixed[0] == 1.0
    assert mixed[-1] == 2.0


def test_mix_with_offset_uses_its_zero_offset_default():
    mixed = utils.mix_with_offset([1.0], [2.0])

    np.testing.assert_array_equal(mixed, [3.0])


def test_mix_with_offset_default_rate_places_a_one_second_delay():
    mixed = utils.mix_with_offset([1.0], [2.0], duration=1)

    assert len(mixed) == 44_101
    assert mixed[0] == 1.0
    assert mixed[-1] == 2.0


def test_mix_many_with_offsets_places_each_sound_at_its_sample_offset():
    mixed = utils.mix_many_with_offsets(
        np.array([1.0, 0.0]), 0,
        np.array([0.0, 2.0]), 1 / 44100,
        np.array([4.0, 0.0]), 2 / 44100)

    np.testing.assert_array_equal(mixed, [1.0, 0.0, 6.0, 0.0])


def test_mix_many_with_offsets_sums_three_unoffset_vectors():
    mixed = utils.mix_many_with_offsets(
        np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0]), np.array([6.0]))

    np.testing.assert_array_equal(mixed, [11.0, 7.0, 3.0])


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
    assert measured["dtype"] == str(sound.dtype)
    assert measured["rms"] == pytest.approx(
        float(np.sqrt(np.mean(sound ** 2))))
    assert measured["mean"] == pytest.approx(float(sound.mean()))
    assert measured["minimum"] == pytest.approx(float(sound.min()))
    assert measured["maximum"] == pytest.approx(float(sound.max()))
    # A steady tone has the same RMS in every block.
    assert measured["block_rms_std"] < 1e-3

    halved = utils.profile({"s": sound}, sample_rate=22050)
    assert halved["analyses"]["ndarray"]["s"]["seconds"] == pytest.approx(1.0)


def test_profile_counts_stereo_frames_once_for_duration():
    stereo = np.vstack((np.ones(20), -np.ones(20)))

    measured = utils.profile({"s": stereo}, sample_rate=20)[
        "analyses"]["ndarray"]["s"]

    assert measured["samples"] == 20
    assert measured["seconds"] == pytest.approx(1.0)


def test_profile_flattens_integer_stereo_before_measuring_blocks():
    channel = np.arange(35, dtype=np.int16)
    stereo = np.vstack((channel, channel))

    measured = utils.profile({"s": stereo})["analyses"]["ndarray"]["s"]

    assert measured["samples"] == 35
    assert measured["seconds"] == pytest.approx(35 / 44100)
    assert measured["mean_square"] == pytest.approx(
        np.mean(np.square(stereo.astype(np.float64))))


def test_profile_measures_short_arrays_one_sample_per_block():
    samples = np.array([3.0, 4.0, 0.0])

    measured = utils.profile({"s": samples})["analyses"]["ndarray"]["s"]

    assert measured["block_rms_mean"] == pytest.approx(7 / 3)
    assert measured["block_rms_std"] == pytest.approx(np.std([3.0, 4.0, 0.0]))


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


def test_profile_pcm_guess_starts_at_its_documented_sample_count():
    short = np.tile([-1.0, 1.0], 499)
    threshold = np.tile([-1.0, 1.0], 500)

    short_readings = [r for r, _ in
                      utils.profile({"s": short})["guesses"]["s"]]
    threshold_readings = [r for r, _ in
                          utils.profile({"s": threshold})["guesses"]["s"]]

    assert "pcm samples" not in short_readings
    assert "pcm samples" in threshold_readings


def test_profile_pcm_guess_requires_a_mean_strictly_inside_five_percent():
    samples = np.concatenate((np.ones(50), np.zeros(950)))

    readings = [r for r, _ in utils.profile({"s": samples})["guesses"]["s"]]

    assert "pcm samples" not in readings
    assert "parametrisation" not in readings


def test_profile_pcm_guess_does_not_round_small_non_power_of_two_bounds_up():
    samples = np.tile([-6.0, 6.0], 500)

    readings = [r for r, _ in utils.profile({"s": samples})["guesses"]["s"]]

    assert "pcm samples" not in readings


def test_profile_does_not_guess_a_nonnegative_scale_for_negative_values():
    readings = [r for r, _ in
                utils.profile({"x": np.array([-1.0, 1.0])})["guesses"]["x"]]

    assert "MIDI pitches or semitone intervals" not in readings
    assert "decibels" not in readings


def test_profile_uses_ten_as_the_decibel_step_boundary():
    description = {
        "numeric": True,
        "finite": True,
        "samples": 4,
        "mean": 50.25,
        "minimum": 0.25,
        "maximum": 100.25,
        "block_rms_std": 10.0,
    }

    readings = [r for r, _ in utils._guess_role(description)]

    assert "decibels" in readings
    assert "MIDI pitches or semitone intervals" not in readings


def test_profile_frequency_guess_starts_above_150_hz():
    below = utils.profile({"f": np.array([148.0, 149.0])})
    at_boundary = utils.profile({"f": np.array([149.0, 150.0])})
    above = utils.profile({"f": np.array([150.0, 151.0])})

    assert "frequencies in Hz" not in [r for r, _ in below["guesses"]["f"]]
    assert "frequencies in Hz" in [
        r for r, _ in at_boundary["guesses"]["f"]]
    assert "frequencies in Hz" in [r for r, _ in above["guesses"]["f"]]


def test_profile_reads_a_power_of_two_bound_as_pcm():
    """16-bit PCM read as integers is centred and bounded by 2**15."""
    samples = (music.note(duration=0.2) * (2 ** 15 - 1)).astype(np.int16)
    readings = [r for r, _ in utils.profile({"s": samples})["guesses"]["s"]]
    assert "pcm samples" in readings


def test_profile_reads_a_two_sample_peak_bound_as_pcm():
    samples = np.tile([-2.0, 2.0], 500)

    readings = [r for r, _ in utils.profile({"s": samples})["guesses"]["s"]]

    assert "pcm samples" in readings


def test_profile_tolerates_float_noise_in_widely_spaced_integer_pitches():
    pitches = utils.profile({
        "p": np.array([0.0000005, 20.0000005, 40.0000005, 60.0000005])})

    readings = [r for r, _ in pitches["guesses"]["p"]]

    assert "MIDI pitches or semitone intervals" in readings


@pytest.mark.parametrize("samples, expected_mean_square, expected_rms", [
    (np.array([20000, -20000], dtype=np.int16), 400_000_000, 20_000),
    (np.array([1e30, -1e30], dtype=np.float32), 1e60, 1e30),
], ids=["int16", "float32"])
def test_profile_squares_in_wide_enough_precision(
        samples, expected_mean_square, expected_rms):
    measured = utils.profile({"s": samples})["analyses"]["ndarray"]["s"]

    assert measured["mean_square"] == pytest.approx(expected_mean_square)
    assert measured["rms"] == pytest.approx(expected_rms)


def test_profile_declines_to_measure_complex_arrays_as_real_pcm():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        measured = utils.profile(
            {"s": np.array([1 + 2j, 3 + 4j])})[
                "analyses"]["ndarray"]["s"]

    assert measured["numeric"] is False
    assert "rms" not in measured
    assert utils.profile({"s": np.array([1 + 2j])})["guesses"]["s"] == []


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


def test_default_pitch_to_freq_starts_from_220_hz():
    pitches = utils.pitch_to_freq(semitones=(0, 12))

    assert pitches == [220.0, 440.0]


def test_default_rhythm_uses_its_documented_duration_and_pattern():
    durations = utils.rhythm_to_durations()

    assert durations == pytest.approx(
        [1, .5, .5, 1, .25, .25, .25, .25, .5, .5, 1])


@pytest.mark.parametrize("kind, expected", [
    ("sine", 0.0),
    ("sawtooth", -1.0),
    ("square", -1.0),
    ("triangle", -1.0),
])
def test_waveform_table_accepts_the_smallest_positive_size(kind, expected):
    table = utils.waveform_table(kind, size=1)

    assert table.shape == (1,)
    assert table[0] == expected


def test_a_single_channel_row_becomes_two_channels():
    """Mono written as one row was returned as it was: one channel, from
    a routine whose whole job is to return two."""
    row = np.array([[.1, .2, .3]])
    np.testing.assert_array_equal(utils.convert_to_stereo(row),
                                  [[.1, .2, .3], [.1, .2, .3]])
