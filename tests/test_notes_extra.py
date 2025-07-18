import numpy as np
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from music.core.synths.notes import (
    note_with_doppler,
    note_with_fm,
    note_with_phase,
    note_with_glissando,
    note_with_glissando_vibrato,
    note_with_two_vibratos_glissando,
    note_with_vibrato,
    note_with_two_vibratos,
)
from music.core.synths.envelopes import tremolo, tremolos


def test_extra_note_functions_shapes():
    params = dict(number_of_samples=10, sample_rate=100)
    assert note_with_doppler(**params).shape == (2, 10)
    assert note_with_fm(fm=0, max_fm_deviation=0, **params).shape == (10,)
    assert note_with_phase(phase=0, **params).shape == (10,)
    assert note_with_glissando(start_freq=220, end_freq=220, **params).shape == (10,)
    assert note_with_glissando_vibrato(
        start_freq=220,
        end_freq=220,
        vibrato_freq=0,
        max_pitch_dev=0,
        **params
    ).shape == (10,)
    assert note_with_two_vibratos_glissando(
        start_freq=220,
        end_freq=220,
        vibrato_freq=0,
        secondary_vibrato_freq=0,
        max_pitch_dev=0,
        **params
    ).shape == (10,)
    assert note_with_vibrato(vibrato_freq=0, max_pitch_dev=0, **params).shape == (10,)
    assert note_with_two_vibratos(
        vibrato_freq=0,
        secondary_vibrato_freq=0,
        nu1=0,
        nu2=0,
        **params
    ).shape == (10,)
    assert tremolo(number_of_samples=10, tremolo_freq=0, max_db_dev=0, sample_rate=100).shape == (10,)
    assert tremolos(
        number_of_samples=[[5, 5]],
        tremolo_freqs=[[0, 0]],
        max_db_devs=[[0, 0]],
        sample_rate=100,
    ).shape == (10,)

