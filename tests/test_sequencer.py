
import numpy as np
import pytest

import music


def test_sequencer_basic_mono():
    seq = music.Sequencer(sample_rate=1000)
    seq.add_note(freq=440, start=0.0, duration=0.01)
    seq.add_note(freq=440, start=0.02, duration=0.01)
    data = seq.render()
    assert data.ndim == 1
    # length should accommodate the last note at 0.02 sec plus its
    # duration of 0.01 => 30 samples
    assert len(data) >= 30


def test_sequencer_stereo_spatial():
    seq = music.Sequencer(sample_rate=1000)
    seq.add_note(
        freq=440,
        start=0.0,
        duration=0.01,
        spatial={"x": 0.1, "y": 0.1},
    )
    out = seq.render()
    assert out.shape[0] == 2


def test_sequencer_write(tmp_path):
    seq = music.Sequencer(sample_rate=1000)
    seq.add_note(freq=220, start=0.0, duration=0.005)
    out_file = tmp_path / "seq.wav"
    seq.write(str(out_file))
    assert out_file.exists()


def test_a_note_cannot_start_before_the_sequence():
    """Rendered at a negative offset, it failed with a broadcast error."""
    sequencer = music.Sequencer()
    with pytest.raises(ValueError, match="cannot start before"):
        sequencer.add_note(440, start=-0.05, duration=0.1)


@pytest.mark.parametrize("suffix", [".wav", ".flac"])
def test_a_sequence_is_written_as_write_audio_writes_it(tmp_path, suffix):
    """It wrote WAV alone, normalized to full scale whatever the notes'
    level. A note held at -12 dB from its first sample peaks near 0.25,
    which normalizing would have taken to 1."""
    seq = music.Sequencer()
    seq.add_note(220, 0, 0.2, adsr_params={"attack_ms": 0, "decay_ms": 0,
                                           "sustain_level": -12})
    assert abs(seq.render()).max() < 0.3
    path = tmp_path / ("sequence" + suffix)
    seq.write(str(path), normalize=False)
    np.testing.assert_allclose(music.read_audio(str(path)), seq.render(),
                               atol=2 ** -15)
