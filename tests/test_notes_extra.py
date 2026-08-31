
import numpy as np
import pytest

from music.core.synths.notes import (
    _fit_to_samples,
    note_with_doppler,
    note_with_fm,
    note_with_phase,
    note_with_glissando,
    note_with_glissando_vibrato,
    note_with_two_vibratos_glissando,
    note_with_vibrato,
    note_with_two_vibratos,
    note_with_vibrato_seq_localization,
    note_with_vibratos_glissandos,
)
from music.core.synths.envelopes import tremolo, tremolos


def test_extra_note_functions_shapes():
    params = dict(number_of_samples=10, sample_rate=100)
    assert note_with_doppler(**params).shape == (2, 10)
    assert note_with_fm(fm=0, max_fm_deviation=0, **params).shape == (10,)
    assert note_with_phase(phase=0, **params).shape == (10,)
    assert note_with_glissando(start_freq=220, end_freq=220,
                               **params).shape == (10,)
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
    assert note_with_vibrato(vibrato_freq=0, max_pitch_dev=0,
                             **params).shape == (10,)
    assert note_with_two_vibratos(
        vibrato_freq=0,
        secondary_vibrato_freq=0,
        nu1=0,
        nu2=0,
        **params
    ).shape == (10,)
    assert tremolo(number_of_samples=10, tremolo_freq=0, max_db_dev=0,
                   sample_rate=100).shape == (10,)
    assert tremolos(
        number_of_samples=[[5, 5]],
        tremolo_freqs=[[0, 0]],
        max_db_devs=[[0, 0]],
        sample_rate=100,
    ).shape == (10,)


# --------------------------------------------------------------------------
# number_of_samples on the sequence routines
#
# Both routines declared `number_of_samples`, documented it as "the number
# of samples of the sound", and never read it: a caller asking for a length
# got whatever the durations happened to sum to. One carried a FIXME that
# had drifted onto the private helper above it; the other carried nothing.
# --------------------------------------------------------------------------

def test_fit_to_samples_passes_the_vector_through_when_unset():
    vector = np.arange(5.0)
    assert _fit_to_samples(vector, 0) is vector


def test_fit_to_samples_truncates_a_longer_vector():
    assert np.allclose(_fit_to_samples(np.arange(5.0), 3), [0, 1, 2])


def test_fit_to_samples_pads_a_shorter_one_with_silence():
    assert np.allclose(_fit_to_samples(np.ones(3), 5), [1, 1, 1, 0, 0])


def test_fit_to_samples_fits_stereo_per_channel():
    """The fit runs along the last axis, so a (2, n) result stays stereo."""
    out = _fit_to_samples(np.ones((2, 3)), 5)
    assert out.shape == (2, 5)
    assert np.allclose(out[:, :3], 1)
    assert np.allclose(out[:, 3:], 0)


@pytest.mark.parametrize("routine, ndim", [
    (note_with_vibrato_seq_localization, 2),
    (note_with_vibratos_glissandos, 1),
])
def test_sequence_routines_honour_number_of_samples(routine, ndim):
    out = routine(number_of_samples=1000)
    assert out.ndim == ndim
    assert out.shape[-1] == 1000
