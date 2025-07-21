import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
from scipy.io import wavfile

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import music


def test_invalid_bit_depth_mono():
    with pytest.raises(ValueError):
        music.write_wav_mono(np.zeros(10), filename="tmp.wav", bit_depth=24)


def test_invalid_bit_depth_stereo():
    with pytest.raises(ValueError):
        music.write_wav_stereo(np.zeros((2, 10)), filename="tmp.wav", bit_depth=24)


def test_read_wav_16bit(tmp_path):
    path = tmp_path / "s16.wav"
    data = np.array([0, 1000, -1000], dtype=np.int16)
    wavfile.write(path, 8000, data)
    out = music.read_wav(str(path))
    assert np.allclose(out, data.astype(np.float64) / (2 ** 15))


def test_read_wav_32bit(tmp_path):
    path = tmp_path / "s32.wav"
    data = np.array([0, 100000, -100000], dtype=np.int32)
    wavfile.write(path, 8000, data)
    out = music.read_wav(str(path))
    assert np.allclose(out, data.astype(np.float64) / (2 ** 31))


def test_play_audio_invocation():
    import types
    from unittest.mock import MagicMock

    sd = types.SimpleNamespace(play=MagicMock(), wait=MagicMock())
    with patch.dict(sys.modules, {"sounddevice": sd}):
        music.play_audio(np.zeros(4), sample_rate=8000)
    sd.play.assert_called_once()
    sd.wait.assert_called_once()
