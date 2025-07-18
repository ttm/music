import numpy as np
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import music


def test_note_and_phase_consistency():
    dur = 0.01
    n = music.note(freq=440, duration=dur)
    n_phase = music.note_with_phase(freq=440, duration=dur, phase=0)
    assert len(n) == int(dur * 44100)
    assert np.allclose(n, n_phase)


def test_note_with_fm_output_shape():
    dur = 0.01
    n_fm = music.note_with_fm(freq=440, duration=dur, fm=0, max_fm_deviation=0)
    assert len(n_fm) == int(dur * 44100)
    assert n_fm.max() <= 1 and n_fm.min() >= -1


def test_glissando_and_vibrato_lengths():
    dur = 0.01
    g = music.note_with_glissando(start_freq=330, end_freq=330, duration=dur)
    assert len(g) == int(dur * 44100)

    g2 = music.note_with_glissando_vibrato(
        start_freq=220, end_freq=220, duration=dur, max_pitch_dev=0
    )
    assert len(g2) == int(dur * 44100)


def test_noise_and_silence_generation():
    sil = music.silence(duration=0.005)
    assert np.allclose(sil, np.zeros_like(sil))

    white = music.noise('white', duration=0.005)
    assert len(white) == int(0.005 * 44100)
    assert white.max() <= 1 and white.min() >= -1

    gauss = music.gaussian_noise(duration=1)
    assert len(gauss) == 44100
    assert gauss.max() <= 1 and gauss.min() >= -1


def test_note_with_doppler_stereo_shape():
    data = music.note_with_doppler(number_of_samples=100, stereo=True)
    assert data.shape[0] == 2
    assert data.shape[1] >= 100
