import sys
import warnings

import numpy as np
from unittest.mock import patch
import pytest
import soundfile as sf

import music


def test_invalid_bit_depth_mono():
    with pytest.raises(ValueError):
        music.write_wav_mono(np.zeros(10), filename="tmp.wav", bit_depth=12)


def test_invalid_bit_depth_stereo():
    with pytest.raises(ValueError):
        music.write_wav_stereo(np.zeros((2, 10)), filename="tmp.wav",
                               bit_depth=12)


def test_read_wav_16bit(tmp_path):
    path = tmp_path / "s16.wav"
    data = np.array([0, 1000, -1000], dtype=np.int16)
    sf.write(str(path), data, 8000, subtype="PCM_16")
    out = music.read_audio(str(path))
    assert np.allclose(out, data.astype(np.float64) / (2 ** 15))


def test_read_wav_32bit(tmp_path):
    path = tmp_path / "s32.wav"
    data = np.array([0, 100000, -100000], dtype=np.int32)
    sf.write(str(path), data, 8000, subtype="PCM_32")
    out = music.read_audio(str(path))
    assert np.allclose(out, data.astype(np.float64) / (2 ** 31))


def test_play_audio_invocation():
    import types
    from unittest.mock import MagicMock

    sd = types.SimpleNamespace(play=MagicMock(), wait=MagicMock())
    with patch.dict(sys.modules, {"sounddevice": sd}):
        music.play_audio(np.zeros(4), sample_rate=8000)
    sd.play.assert_called_once()
    sd.wait.assert_called_once()


# ---------------------------------------------------------------------
# A file's rate, and a file at the level it was given
# ---------------------------------------------------------------------

def _at(tmp_path, rate):
    path = tmp_path / f"at-{rate}.wav"
    sf.write(str(path), np.linspace(-0.5, 0.5, 10), rate, subtype="PCM_16")
    return str(path)


def test_read_audio_gives_the_files_rate_when_asked(tmp_path):
    samples, rate = music.read_audio(_at(tmp_path, 8000),
                                     return_sample_rate=True)
    assert rate == 8000 and type(rate) is int
    assert samples.shape == (10,)


def test_reading_a_file_at_another_rate_without_it_says_so(tmp_path):
    """It came back with nothing to say what rate it was at, so a 48 kHz
    file written back at the default was 9 % long and a semitone and a
    half flat."""
    with pytest.warns(UserWarning,
                      match=r"48000 Hz.*return_sample_rate=True"):
        music.read_audio(_at(tmp_path, 48000))


def test_a_file_whose_rate_is_known_reads_quietly(tmp_path):
    path = _at(tmp_path, 48000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        music.read_audio(path, return_sample_rate=True)
        music.read_audio(_at(tmp_path, 44100))
        with music.set_sample_rate(48000):
            music.read_audio(path)


@pytest.mark.parametrize("write,shape,name", [
    (music.write_wav_mono, (2205,), "m.wav"),
    (music.write_wav_stereo, (2, 2205), "s.wav"),
    (music.write_audio, (2205,), "a.flac"),
    (music.write_audio, (2, 2205), "a.wav"),
])
def test_normalize_false_writes_the_level_it_is_given(tmp_path, write,
                                                      shape, name):
    """Every file used to arrive at full scale, so a piece written a
    phrase to a file lost its dynamics, and scaling by hand first was
    undone by the writer."""
    quiet = 0.01 * np.broadcast_to(music.note(440, 0.05), shape)
    path = str(tmp_path / name)
    write(quiet, path, normalize=False)
    np.testing.assert_allclose(music.read_audio(path), quiet,
                               atol=2 ** -15)


def test_normalize_false_keeps_the_offset_too(tmp_path):
    """Removing the bias is part of normalizing."""
    path = str(tmp_path / "offset.wav")
    music.write_wav_mono(0.3 + 0.1 * music.note(440, 0.05), path,
                         normalize=False)
    assert music.read_audio(path).mean() == pytest.approx(0.3, abs=1e-3)


def test_normalize_false_still_fades(tmp_path):
    path = str(tmp_path / "faded.wav")
    music.write_wav_mono(0.5 * np.ones(4410), path, normalize=False,
                         fades_ms=(20, 20))
    written = music.read_audio(path)
    assert written[0] == 0 and written[-1] == pytest.approx(0, abs=1e-3)
    assert written[2205] == pytest.approx(0.5, abs=1e-4)


@pytest.mark.parametrize("write", [music.write_wav_mono,
                                   music.write_wav_stereo])
def test_normalize_false_refuses_a_sound_past_full_scale(tmp_path, write):
    """Rather than clip it quietly, which is the level problem this
    option exists to avoid, moved somewhere harder to hear."""
    loud = 1.5 * np.ones((2, 10)) if write is music.write_wav_stereo \
        else 1.5 * np.ones(10)
    with pytest.raises(ValueError, match="reaches 1.5"):
        write(loud, str(tmp_path / "loud.wav"), normalize=False)


def test_normalize_false_refuses_nothing_as_normalizing_does(tmp_path):
    with pytest.raises(ValueError, match="nothing here to write"):
        music.write_wav_mono(np.array([]), str(tmp_path / "e.wav"),
                             normalize=False)


def test_full_scale_itself_is_written(tmp_path):
    path = str(tmp_path / "full.wav")
    music.write_wav_mono(np.array([-1.0, 0.0, 1.0]), path, normalize=False)
    np.testing.assert_allclose(music.read_audio(path), [-1, 0, 1],
                               atol=2 ** -15)


@pytest.mark.parametrize("fades", [-5, (10,), (10, 20, 30), (10, -20)])
@pytest.mark.parametrize("writer", ["write_wav_mono", "write_wav_stereo",
                                    "write_audio"])
def test_the_writers_refuse_fades_they_cannot_apply(tmp_path, writer,
                                                    fades):
    """One value in a sequence failed as an IndexError, a third was
    ignored, and a negative one was refused in words about an `attack_ms`
    nobody gave."""
    with pytest.raises(ValueError, match="fades_ms is milliseconds of fade"):
        getattr(music, writer)(music.note(duration=0.1),
                               str(tmp_path / "faded.wav"), fades_ms=fades)


def test_a_fade_at_one_end_only_leaves_the_other_alone(tmp_path):
    """A pair holding a zero reaches the fades, where all zeros skip them:
    nothing had asked for a fade out alone."""
    tone = music.note(duration=0.2)
    path = str(tmp_path / "fade_out.wav")
    music.write_wav_mono(tone, path, fades_ms=(0, 50))
    written = music.read_audio(path)
    np.testing.assert_allclose(written[:100], music.normalize_mono(tone)[:100],
                               atol=2 ** -15)
    assert abs(written[-10:]).max() < 1e-3
