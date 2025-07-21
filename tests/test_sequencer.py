import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import music


def test_sequencer_basic_mono():
    seq = music.Sequencer(sample_rate=1000)
    seq.add_note(freq=440, start=0.0, duration=0.01)
    seq.add_note(freq=440, start=0.02, duration=0.01)
    data = seq.render()
    assert data.ndim == 1
    # length should accommodate last note at 0.02 sec plus duration 0.01 => 30 samples
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
