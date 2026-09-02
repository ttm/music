"""The reading and writing paths that the round-trip tests do not reach."""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

import music
from music.core.io import _fade_pair


# --------------------------------------------------------------------------
# fades
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fades, expected", [
    (20, (20, 20)),
    (20.0, (20, 20)),
    (np.int64(20), (20, 20)),
    ((10, 30), (10, 30)),
    ([10, 30], (10, 30)),
    (np.array([10, 30]), (10, 30)),
])
def test_fade_pair_resolves_every_documented_form(fades, expected):
    """A scalar applies to both ends; a pair is taken as given."""
    assert _fade_pair(fades) == expected


@pytest.mark.parametrize("stereo", [False, True])
def test_a_scalar_fade_is_applied_at_both_ends(stereo, tmp_path):
    """Regression: a scalar `fades` used to be silently ignored."""
    path = tmp_path / "faded.wav"
    tone = music.note(440, 0.2)
    signal = np.vstack((tone, tone)) if stereo else tone
    writer = music.write_wav_stereo if stereo else music.write_wav_mono

    writer(signal, filename=str(path), fades=20)
    faded = music.read_wav(str(path))
    writer(signal, filename=str(path), fades=0)
    plain = music.read_wav(str(path))

    channel = (lambda a: a[0]) if stereo else (lambda a: a)
    # The opening of the faded render is quieter than the unfaded one.
    opening = slice(0, 200)
    assert (np.abs(channel(faded)[opening]).max()
            < np.abs(channel(plain)[opening]).max())


# --------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------

@pytest.mark.parametrize("writer, channels", [
    (music.write_wav_mono, 1),
    (music.write_wav_stereo, 2),
])
def test_the_writers_render_noise_when_given_nothing(writer, channels,
                                                     tmp_path):
    """The default is a couple of seconds of uniform noise, generated on
    the call rather than at import."""
    path = tmp_path / "default.wav"
    writer(filename=str(path))

    restored = music.read_wav(str(path))
    assert (restored.ndim == 1 if channels == 1 else restored.shape[0] == 2)
    assert restored.size > 0


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def test_a_float_wav_is_scaled_by_its_own_peak(tmp_path):
    path = tmp_path / "float.wav"
    sf.write(str(path), np.array([0.0, 0.25, -0.5], dtype=np.float32), 8000,
             subtype="FLOAT")

    out = music.read_wav(str(path))

    assert out.max() == pytest.approx(0.5)
    assert out.min() == pytest.approx(-1.0)


def test_an_all_zero_float_wav_does_not_divide_by_zero(tmp_path):
    path = tmp_path / "silent.wav"
    sf.write(str(path), np.zeros(8, dtype=np.float32), 8000,
             subtype="FLOAT")

    assert np.array_equal(music.read_wav(str(path)), np.zeros(8))


def test_an_encoding_the_reader_does_not_support_is_reported(tmp_path):
    """A real file in a real encoding, rather than a mocked return value.

    libsndfile will happily decode ADPCM, but a WAV whose samples are not
    linear PCM has no full scale this package's normalization is defined
    against, so it is refused rather than silently rescaled.
    """
    path = tmp_path / "odd.wav"
    sf.write(str(path), np.zeros(64, dtype=np.int16), 8000,
             subtype="IMA_ADPCM")

    with pytest.raises(ValueError, match="unsupported WAV encoding"):
        music.read_wav(str(path))


# --------------------------------------------------------------------------
# playback
# --------------------------------------------------------------------------

def test_play_audio_transposes_stereo_for_the_device():
    """sounddevice wants (samples, channels); the package uses
    (channels, samples)."""
    device = types.SimpleNamespace(play=MagicMock(), wait=MagicMock())
    stereo = np.vstack((np.ones(8), np.zeros(8)))

    with patch.dict(sys.modules, {"sounddevice": device}):
        music.play_audio(stereo, sample_rate=8000)

    passed = device.play.call_args[0][0]
    assert passed.shape == (8, 2)


def test_play_audio_can_skip_normalization():
    device = types.SimpleNamespace(play=MagicMock(), wait=MagicMock())
    quiet = np.full(8, 0.25)

    with patch.dict(sys.modules, {"sounddevice": device}):
        music.play_audio(quiet, sample_rate=8000, normalize=False)

    assert np.allclose(device.play.call_args[0][0], quiet)


def test_play_audio_says_so_when_sounddevice_is_missing(caplog):
    """It is an optional dependency, so this is a warning, not an error."""
    with patch.dict(sys.modules, {"sounddevice": None}):
        music.play_audio(np.zeros(4))
    assert "sounddevice" in caplog.text
