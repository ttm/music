import numpy as np
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import music


def _dominant_freq(samples, sample_rate=44100):
    freqs = np.fft.rfftfreq(len(samples), 1 / sample_rate)
    spectrum = np.fft.rfft(samples)
    return freqs[np.argmax(np.abs(spectrum))]


def test_note_peak_frequency():
    note = music.note(freq=440, duration=0.1)
    peak = _dominant_freq(note)
    assert abs(peak - 440) <= 5


def test_note_with_fm_peak_frequency():
    note = music.note_with_fm(freq=440, duration=0.1, fm=5, max_fm_deviation=5)
    peak = _dominant_freq(note)
    assert abs(peak - 440) <= 5


def test_note_with_glissando_peak_frequency():
    note = music.note_with_glissando(start_freq=430, end_freq=450, duration=0.1)
    peak = _dominant_freq(note)
    assert abs(peak - 440) <= 5
