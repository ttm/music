"""Containers: what this package will write, and what it refuses to.

FLAC earns its place by being lossless -- the point of it here is a
smaller file and not a different sound, so the tests that matter are the
ones showing a FLAC round trip is exactly as accurate as the WAV round
trip at the same depth. If that ever stops being true, the compression
has started discarding the thing this package exists to preserve.
"""

import numpy as np
import pytest
import soundfile as sf

import music
from music.core.functions import normalize_mono

DEPTHS = [8, 16, 24]


@pytest.mark.parametrize("bit_depth", DEPTHS)
def test_flac_is_exactly_as_accurate_as_wav_at_the_same_depth(bit_depth,
                                                              tmp_path):
    """Lossless means lossless: same depth, same samples, smaller file."""
    signal = music.note(freq=440, duration=0.05)
    stored = normalize_mono(signal, True)

    wav = tmp_path / f"t{bit_depth}.wav"
    flac = tmp_path / f"t{bit_depth}.flac"
    music.write_audio(signal, str(wav), bit_depth=bit_depth)
    music.write_audio(signal, str(flac), bit_depth=bit_depth)

    from_wav = music.read_audio(str(wav))
    from_flac = music.read_audio(str(flac))

    assert np.array_equal(from_wav, from_flac)
    assert np.abs(from_flac - stored).max() <= 2.0 ** -(bit_depth - 1)
    assert flac.stat().st_size < wav.stat().st_size


@pytest.mark.parametrize("bit_depth, subtype", [
    (8, "PCM_S8"), (16, "PCM_16"), (24, "PCM_24"),
])
def test_flac_uses_its_own_encoding_for_each_depth(bit_depth, subtype,
                                                   tmp_path):
    """FLAC stores 8-bit signed where WAV stores it unsigned, so the two
    containers cannot share one subtype table."""
    path = tmp_path / "t.flac"
    music.write_audio(music.note(duration=0.02), str(path),
                      bit_depth=bit_depth)
    assert sf.info(str(path)).subtype == subtype


def test_flac_has_no_32_bit_form_and_says_so(tmp_path):
    """libsndfile's own message for this is 'Invalid combination of
    format, subtype and endian', which tells a caller nothing."""
    with pytest.raises(ValueError, match="allowed for FLAC are only"):
        music.write_audio(music.note(duration=0.02),
                          str(tmp_path / "t.flac"), bit_depth=32)


def test_wav_still_has_its_32_bit_form(tmp_path):
    path = tmp_path / "t.wav"
    music.write_audio(music.note(duration=0.02), str(path), bit_depth=32)
    assert sf.info(str(path)).subtype == "PCM_32"


def test_an_extension_we_do_not_write_is_refused(tmp_path):
    """Named, rather than written as a WAV with a misleading name."""
    with pytest.raises(ValueError, match="unsupported audio file extension"):
        music.write_audio(music.note(duration=0.02),
                          str(tmp_path / "t.mp3"))


def test_a_file_with_no_extension_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unsupported audio file extension"):
        music.write_audio(music.note(duration=0.02), str(tmp_path / "t"))


def test_write_audio_picks_the_writer_from_the_array(tmp_path):
    """A caller with a sound and a path should not have to dispatch."""
    mono = tmp_path / "mono.flac"
    stereo = tmp_path / "stereo.flac"
    music.write_audio(music.note(duration=0.05), str(mono))
    music.write_audio(music.localize_linear(music.note(duration=0.05)),
                      str(stereo))

    assert music.read_audio(str(mono)).ndim == 1
    assert music.read_audio(str(stereo)).shape[0] == 2


def test_write_audio_with_nothing_writes_noise_like_the_others(tmp_path):
    path = tmp_path / "default.wav"
    music.write_audio(filename=str(path))
    assert music.read_audio(str(path)).ndim == 1


def test_fades_reach_the_writer_through_write_audio(tmp_path):
    """It forwards its arguments rather than quietly dropping some."""
    path = tmp_path / "faded.wav"
    music.write_audio(music.note(duration=0.2), str(path), fades=50)
    out = music.read_audio(str(path))
    assert abs(out[0]) < abs(out[len(out) // 2])


def test_the_stereo_writers_still_take_flac(tmp_path):
    """The container follows the extension, whatever the writer is
    called; the wav in the name predates FLAC."""
    path = tmp_path / "s.flac"
    music.write_wav_stereo(music.localize_linear(music.note(duration=0.05)),
                           filename=str(path))
    assert sf.info(str(path)).format == "FLAC"


def test_read_wav_is_still_bound_and_reads_flac_too(tmp_path):
    """The old name kept working, and got more capable rather than less."""
    path = tmp_path / "t.flac"
    music.write_audio(music.note(duration=0.05), str(path))
    assert music.read_wav is music.read_audio
    assert music.read_wav(str(path)).ndim == 1
