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
    assert faded.shape == signal.shape

    channel = (lambda a: a[0]) if stereo else (lambda a: a)
    # The opening of the faded render is quieter than the unfaded one.
    opening = slice(0, 200)
    assert (np.abs(channel(faded)[opening]).max()
            < np.abs(channel(plain)[opening]).max())


@pytest.mark.parametrize("sample_rate", [8000, 44100, 48000, 96000])
@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("extension", ["wav", "flac"])
def test_written_fade_durations_follow_the_file_sample_rate(
        sample_rate, stereo, extension, tmp_path):
    """Read an actual file's envelope at the requested time boundaries."""
    tone = np.tile([-1.0, 1.0], sample_rate * 3 // 4)
    signal = np.vstack((tone, tone * 0.25)) if stereo else tone
    writer = music.write_wav_stereo if stereo else music.write_wav_mono
    if extension == "flac":
        writer = music.write_audio
    path = tmp_path / f"faded.{extension}"

    writer(signal, filename=str(path), sample_rate=sample_rate,
           fades=(100, 150))

    restored = music.read_audio(str(path))
    assert restored.shape == signal.shape
    channel = restored[0] if stereo else restored
    full_scale = np.flatnonzero(np.abs(channel) > 0.9999)
    # The attack ends at full scale and the release starts there; the
    # endpoint samples belong to their respective ramps.
    assert full_scale[0] == sample_rate * 100 // 1000 - 1
    assert full_scale[-1] == len(tone) - sample_rate * 150 // 1000
    assert abs(channel[0]) < 0.001
    assert abs(channel[-1]) < 0.001
    assert sf.info(str(path)).samplerate == sample_rate
    plateau = slice(sample_rate * 100 // 1000,
                    len(tone) - sample_rate * 150 // 1000)
    # An envelope alone has the right absolute values on this carrier,
    # but the writer must preserve the signed audio underneath it.
    assert np.allclose(channel[plateau], tone[plateau],
                       rtol=0, atol=1 / 32768)
    if stereo:
        assert np.allclose(restored[1], restored[0] * 0.25,
                           rtol=0, atol=1 / 32768)


@pytest.mark.parametrize("stereo", [False, True])
def test_numpy_fade_pair_writes_the_same_sound_as_a_tuple(stereo, tmp_path):
    tone = np.tile([-1.0, 1.0], 8000)
    signal = np.vstack((tone, tone)) if stereo else tone
    writer = music.write_wav_stereo if stereo else music.write_wav_mono
    tuple_path = tmp_path / "tuple.wav"
    array_path = tmp_path / "array.wav"

    writer(signal, filename=str(tuple_path), fades=(10, 30))
    writer(signal, filename=str(array_path), fades=np.array([10, 30]))

    assert np.array_equal(music.read_audio(str(array_path)),
                          music.read_audio(str(tuple_path)))


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("fades", [0, False, np.int64(0), (0, 0), [0, 0],
                                  np.zeros(2, dtype=int), []])
def test_disabled_fades_leave_the_samples_untouched(stereo, fades, tmp_path):
    tone = np.tile([-1.0, 1.0], 16)
    signal = np.vstack((tone, tone)) if stereo else tone
    writer = music.write_wav_stereo if stereo else music.write_wav_mono
    plain_path = tmp_path / "plain.wav"
    faded_path = tmp_path / "disabled.wav"

    writer(signal, filename=str(plain_path))
    writer(signal, filename=str(faded_path), fades=fades)

    assert np.array_equal(music.read_audio(str(faded_path)),
                          music.read_audio(str(plain_path)))


# --------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------

@pytest.mark.parametrize("writer, channels", [
    (music.write_wav_mono, 1),
    (music.write_wav_stereo, 2),
    (music.write_audio, 1),
])
def test_writers_use_the_documented_default_file_and_rate(
        writer, channels, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    writer([-1.0, 0.0, 1.0])

    assert [path.name for path in tmp_path.iterdir()] == ["asound.wav"]
    info = sf.info(str(tmp_path / "asound.wav"))
    assert info.samplerate == 44100
    assert info.channels == channels
    assert info.frames == 3


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
    assert np.std(restored) > 0.1
    assert sf.info(str(path)).samplerate == 44100


@pytest.mark.parametrize("writer", [music.write_wav_mono, music.write_audio])
@pytest.mark.parametrize("remove_bias, expected", [
    (True, [-0.8, -0.2, 1.0]),
    (False, [-1.0, -1 / 3, 1.0]),
])
def test_mono_writers_honor_the_normalization_choice(
        writer, remove_bias, expected, tmp_path):
    path = tmp_path / "normalization.wav"
    writer([1.0, 2.0, 4.0], filename=str(path), remove_bias=remove_bias)
    assert np.allclose(music.read_audio(str(path)), expected,
                       rtol=0, atol=1 / 32768)


@pytest.mark.parametrize("options, expected", [
    ({}, [[-0.4, -0.1, 0.5], [-0.8, -0.2, 1.0]]),
    ({"remove_bias": True, "normalize_separately": False},
     [[-0.4, -0.1, 0.5], [-0.8, -0.2, 1.0]]),
    ({"remove_bias": False, "normalize_separately": False},
     [[-1.0, -5 / 7, -1 / 7], [-5 / 7, -1 / 7, 1.0]]),
    ({"remove_bias": True, "normalize_separately": True},
     [[-0.8, -0.2, 1.0], [-0.8, -0.2, 1.0]]),
    ({"remove_bias": False, "normalize_separately": True},
     [[-1.0, -1 / 3, 1.0], [-1.0, -1 / 3, 1.0]]),
])
def test_stereo_writer_honors_both_normalization_choices(
        options, expected, tmp_path):
    path = tmp_path / "normalization.wav"
    music.write_wav_stereo([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]],
                           filename=str(path), **options)
    assert np.allclose(music.read_audio(str(path)), expected,
                       rtol=0, atol=1 / 32768)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

@pytest.mark.parametrize("subtype", ["FLOAT", "DOUBLE"])
def test_a_float_wav_is_scaled_by_its_own_peak(subtype, tmp_path):
    path = tmp_path / "float.wav"
    sf.write(str(path), np.array([0.0, 0.25, -0.5], dtype=np.float32), 8000,
             subtype=subtype)

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


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("sample_rate", [None, 8000])
def test_playback_preserves_samples_channels_and_rate(
        stereo, normalize, sample_rate):
    """Device calls must carry the sound, not just have the right shape."""
    signal = np.array([0, 1, 3], dtype=np.int16)
    expected = np.array([-0.8, -0.2, 1.0])
    if stereo:
        signal = np.vstack((signal, signal * 2))
        expected = np.vstack((expected * 0.5, expected)).T
    if not normalize:
        expected = signal.T
    options = {} if normalize else {"normalize": False}
    if sample_rate is not None:
        options["sample_rate"] = sample_rate
    device = types.SimpleNamespace(play=MagicMock(), wait=MagicMock())

    with patch.dict(sys.modules, {"sounddevice": device}):
        music.play_audio(signal, **options)

    device.play.assert_called_once()
    data = device.play.call_args.args[0]
    assert data.shape == expected.shape
    assert data.dtype == np.float64
    assert np.allclose(data, expected, rtol=0, atol=1e-15)
    assert device.play.call_args.kwargs["samplerate"] == (
        44100 if sample_rate is None else sample_rate)
    device.wait.assert_called_once()


def test_play_audio_says_so_when_sounddevice_is_missing(caplog):
    """It is an optional dependency, so this is a warning, not an error."""
    with patch.dict(sys.modules, {"sounddevice": None}):
        music.play_audio(np.zeros(4))
    assert "sounddevice" in caplog.text
