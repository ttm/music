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
# A file's rate
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
