
import time

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


def test_a_render_costs_the_length_of_the_piece_not_its_square():
    """Regression: the mix grew by a copy for every note, each the length
    of everything before it, so 2,000 notes in a row -- 200 s of audio --
    took half a minute to render, a tenth of a second of it synthesis.

    Four times the notes in a row is four times the piece: that cost
    about 13x, and costs 4x now. The threshold is loose because this is
    wall-clock on shared CI.
    """
    def best_of_three(count):
        seq = music.Sequencer(sample_rate=8000)
        for i in range(count):
            seq.add_note(220, start=i * .05, duration=.05)
        times = []
        for _ in range(3):
            start = time.perf_counter()
            seq.render()
            times.append(time.perf_counter() - start)
        return min(times)

    best_of_three(10)  # warm up
    small, large = best_of_three(400), best_of_three(1600)
    assert large / small < 8, (
        f"four times the notes cost {large / small:.1f}x, which looks "
        f"quadratic rather than linear")


def test_a_note_of_no_samples_still_reaches_to_where_it_starts():
    """The mix is as long as the note that ends last, and one of no
    samples ends where it starts, as it did when the mix grew note by
    note."""
    seq = music.Sequencer(sample_rate=1000)
    seq.add_note(220, 0, 0.01)
    seq.add_note(220, 0.05, 0)
    assert seq.render().shape == (50,)
