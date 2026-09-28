"""What the `sequencer` mutation audit found untested or wrong."""

import math
import re

import numpy as np
import pytest

import music
from music import Sequencer
from music.sequencer import NoteEvent


def _exactly(message):
    return "^" + re.escape(message) + "$"


@pytest.mark.parametrize("start", [math.nan, math.inf])
def test_a_note_starts_a_finite_time_into_the_sequence(start):
    """NaN and infinity were accepted and failed in round() on render."""
    sequencer = Sequencer()
    with pytest.raises(ValueError, match=_exactly(
            "a note starts a finite number of seconds into the sequence; "
            f"got start={start}")):
        sequencer.add_note(440, start, 0.1)
    assert sequencer.events == []


def test_a_note_before_the_sequence_is_refused_by_its_start():
    with pytest.raises(ValueError, match=_exactly(
            "a note cannot start before the sequence does; got "
            "start=-0.5")):
        Sequencer().add_note(440, -0.5, 0.1)


@pytest.mark.parametrize("name", ["adsr_params", "spatial"])
@pytest.mark.parametrize("keys, named", [
    ({"sample_rate": 8000}, "sample_rate"),
    ({"sonic_vector": [0.0]}, "sonic_vector"),
    ({"sample_rate": 8000, "sonic_vector": [0.0]},
     "sample_rate, sonic_vector"),
])
def test_the_sequencer_keeps_the_arguments_it_supplies(name, keys, named):
    """A second sample_rate was a TypeError when the note rendered."""
    with pytest.raises(ValueError, match=_exactly(
            f"{name} cannot set {named}: the sequencer passes the note and "
            "its own sample rate")):
        Sequencer().add_note(440, 0, 0.1, **{name: keys})


@pytest.mark.parametrize("sample_rate", [0, -8000])
def test_a_sequencer_needs_a_positive_sample_rate(sample_rate):
    with pytest.raises(ValueError, match=_exactly(
            f"sample_rate must be positive; got {sample_rate}")):
        Sequencer(sample_rate=sample_rate)


def test_writing_no_notes_says_so(tmp_path):
    """The empty render reached the normalization, which blamed a
    duration computed as zero."""
    with pytest.raises(ValueError, match=_exactly(
            "there are no notes to write; add some with add_note")):
        Sequencer().write(str(tmp_path / "nothing.wav"))
    assert not (tmp_path / "nothing.wav").exists()


# --------------------------------------------------------------------------
# What a sequence renders, sample for sample, against the routines it
# hands each note to
# --------------------------------------------------------------------------

RATE = 8000


def _alone(**note):
    sequencer = Sequencer(sample_rate=RATE)
    sequencer.add_note(**note)
    return sequencer.render()


def test_a_note_is_rendered_at_its_frequency_duration_and_rate():
    np.testing.assert_array_equal(
        _alone(freq=330, start=0, duration=0.1),
        music.note(freq=330, duration=0.1, sample_rate=RATE))


def test_a_note_with_a_vibrato_is_rendered_with_it():
    np.testing.assert_array_equal(
        _alone(freq=330, start=0, duration=0.1, vibrato_freq=7,
               max_pitch_dev=3),
        music.note_with_vibrato(freq=330, duration=0.1, vibrato_freq=7,
                                max_pitch_dev=3, sample_rate=RATE))


@pytest.mark.parametrize("vibrato_freq, max_pitch_dev", [(7, 0), (0, 2)])
def test_a_vibrato_needs_both_a_rate_and_a_depth(vibrato_freq,
                                                 max_pitch_dev):
    np.testing.assert_array_equal(
        _alone(freq=330, start=0, duration=0.1, vibrato_freq=vibrato_freq,
               max_pitch_dev=max_pitch_dev),
        music.note(freq=330, duration=0.1, sample_rate=RATE))


def test_a_note_s_envelope_is_the_adsr_it_was_given():
    params = dict(attack_duration=10, decay_duration=10,
                  release_duration=20, sustain_level=-6)
    np.testing.assert_array_equal(
        _alone(freq=330, start=0, duration=0.1, adsr_params=params),
        music.adsr(sonic_vector=music.note(330, 0.1, sample_rate=RATE),
                   sample_rate=RATE, **params))


def test_a_note_s_place_is_the_localization_it_was_given():
    spatial = dict(x=0.5, y=1.0)
    np.testing.assert_array_equal(
        _alone(freq=330, start=0, duration=0.1, spatial=spatial),
        music.localize(sonic_vector=music.note(330, 0.1, sample_rate=RATE),
                       sample_rate=RATE, **spatial))


def test_overlapping_notes_add():
    sequencer = Sequencer(sample_rate=RATE)
    sequencer.add_note(330, start=0.05, duration=0.1)
    sequencer.add_note(220, start=0, duration=0.1)
    first = music.note(220, 0.1, sample_rate=RATE)
    second = music.note(330, 0.1, sample_rate=RATE)
    expected = np.zeros(400 + 800)
    expected[:800] += first
    expected[400:] += second
    np.testing.assert_allclose(sequencer.render(), expected, atol=1e-12)


def test_overlapping_stereo_notes_add_on_each_channel():
    left, right = dict(x=-1.0, y=0.5), dict(x=1.0, y=0.5)
    sequencer = Sequencer(sample_rate=RATE)
    sequencer.add_note(220, start=0, duration=0.1, spatial=left)
    sequencer.add_note(330, start=0.05, duration=0.1, spatial=right)
    first = music.localize(sonic_vector=music.note(220, 0.1,
                                                   sample_rate=RATE),
                           sample_rate=RATE, **left)
    second = music.localize(sonic_vector=music.note(330, 0.1,
                                                    sample_rate=RATE),
                            sample_rate=RATE, **right)
    expected = np.zeros((2, max(first.shape[1], 400 + second.shape[1])))
    expected[:, :first.shape[1]] += first
    expected[:, 400:400 + second.shape[1]] += second
    np.testing.assert_allclose(sequencer.render(), expected, atol=1e-12)


def test_a_mono_note_joins_a_stereo_one_on_both_channels():
    spatial = dict(x=1.0, y=0.5)
    sequencer = Sequencer(sample_rate=RATE)
    sequencer.add_note(220, start=0, duration=0.1, spatial=spatial)
    sequencer.add_note(330, start=0.02, duration=0.05)
    placed = music.localize(sonic_vector=music.note(220, 0.1,
                                                    sample_rate=RATE),
                            sample_rate=RATE, **spatial)
    plain = music.note(330, 0.05, sample_rate=RATE)
    expected = placed.copy()
    expected[:, 160:160 + len(plain)] += plain
    np.testing.assert_allclose(sequencer.render(), expected, atol=1e-12)


def test_a_note_of_no_samples_at_the_start_renders_no_samples():
    assert _alone(freq=330, start=0, duration=0).shape == (0,)


def test_a_stereo_note_first_starts_a_stereo_sequence():
    sequencer = Sequencer(sample_rate=RATE)
    sequencer.add_note(330, 0.01, 0.05, spatial={"x": 1.0})
    placed = music.localize(sonic_vector=music.note(330, 0.05,
                                                    sample_rate=RATE),
                            sample_rate=RATE, x=1.0)
    expected = np.zeros((2, 80 + placed.shape[1]))
    expected[:, 80:] = placed
    np.testing.assert_array_equal(sequencer.render(), expected)


def test_nothing_scheduled_renders_an_empty_mono_array():
    rendered = Sequencer().render()
    assert rendered.shape == (0,) and rendered.dtype == np.float64


def test_add_note_records_every_field_it_is_given():
    sequencer = Sequencer()
    sequencer.add_note(330, 0.5, 0.25, vibrato_freq=6, max_pitch_dev=3,
                       adsr_params={"attack_duration": 5},
                       spatial={"x": 1.0})
    sequencer.add_note(220, 0, 1)
    assert sequencer.events == [
        NoteEvent(330, 0.5, 0.25, 6, 3, {"attack_duration": 5},
                  {"x": 1.0}),
        NoteEvent(220, 0, 1, 0.0, 0.0, None, None)]


def test_a_sequencer_at_one_hertz_is_a_sequencer():
    assert Sequencer(sample_rate=1).sample_rate == 1


@pytest.mark.parametrize("spatial, channels", [(None, 1), ({"x": 1.0}, 2)])
def test_written_audio_is_the_render_at_the_sequencer_s_rate(
        tmp_path, spatial, channels):
    soundfile = pytest.importorskip("soundfile")
    sequencer = Sequencer(sample_rate=RATE)
    sequencer.add_note(330, 0, 0.1, spatial=spatial)
    sequencer.add_note(220, 0.05, 0.1)
    path = tmp_path / "sequence.wav"
    sequencer.write(str(path), bit_depth=24)
    info = soundfile.info(str(path))
    assert (info.samplerate, info.channels, info.subtype) == (
        RATE, channels, "PCM_24")
    rendered = sequencer.render()
    normalized = (music.normalize_mono(rendered) if channels == 1
                  else music.normalize_stereo(rendered))
    np.testing.assert_allclose(music.read_audio(str(path)), normalized,
                               atol=2 ** -22)
