"""Every oscillator refuses to read its table at a phase that is no number.

The oscillators read their waveform tables at the whole part of a phase,
cast to an integer, and NaN and infinity cast to the most negative one:
the same entry at every sample. So a note at a frequency of NaN, or a
vibrato of infinite depth, came back as a constant read from the table,
finite and plausible, with numpy's "invalid value encountered in cast"
for the only notice. The glissandos already refused the frequencies that
did this to them; these hold every reading of a table to it.
"""

import math
import warnings

import numpy as np
import pytest

import music
from music.utils import _entries

MESSAGE = ("the waveform table is read at a phase that is not a finite "
           "number: a frequency, a depth or a position given is NaN or "
           "infinite")


NAN = math.nan
CASES = {
    "note": lambda: music.note(NAN, 0.1),
    "note, infinite": lambda: music.note(math.inf, 0.1),
    "note_with_phase": lambda: music.note_with_phase(220, 0.1, phase=NAN),
    "note_with_fm": lambda: music.note_with_fm(220, 0.1, fm=NAN),
    "note_with_fm, its depth": lambda: music.note_with_fm(
        220, 0.1, max_fm_deviation=NAN),
    "note_with_doppler": lambda: music.note_with_doppler(NAN, 0.1),
    "note_with_doppler, a position": lambda: music.note_with_doppler(
        220, 0.1, x=(NAN, 1)),
    "note_with_doppler, mono": lambda: music.note_with_doppler(
        220, 0.1, x=(NAN, 1), stereo=False),
    "note_with_glissando, linear": lambda: music.note_with_glissando(
        220, NAN, 0.1, method="lin"),
    "note_with_glissando_vibrato": lambda: music.note_with_glissando_vibrato(
        220, 440, 0.1, vibrato_freq=NAN),
    "note_with_vibrato": lambda: music.note_with_vibrato(220, 0.1, NAN),
    "note_with_vibrato, its depth": lambda: music.note_with_vibrato(
        220, 0.1, 5, math.inf),
    "note_with_two_vibratos": lambda: music.note_with_two_vibratos(
        220, 0.1, vibrato_freq=NAN),
    "note_with_two_vibratos, the second": lambda: music.note_with_two_vibratos(
        220, 0.1, secondary_vibrato_freq=NAN),
    "note_with_two_vibratos_glissando":
        lambda: music.note_with_two_vibratos_glissando(
            220, 440, 0.1, vibrato_freq=NAN),
    "note_with_two_vibratos_glissando, the second":
        lambda: music.note_with_two_vibratos_glissando(
            220, 440, 0.1, secondary_vibrato_freq=NAN),
    "note_with_vibratos_glissandos":
        lambda: music.note_with_vibratos_glissandos(
            vibratos_freqs=((2, 6, NAN), (0.5, 15, 2, 6, 3))),
    "note_with_vibrato_seq_localization":
        lambda: music.note_with_vibrato_seq_localization(
            vibratos_freqs=((2, 6, NAN), (0.5, 15, 2, 6, 3))),
    "trill": lambda: music.trill((NAN, 440), duration=0.2),
    "am": lambda: music.am(0.1, NAN),
    "tremolo": lambda: music.tremolo(0.1, NAN),
    "binaural_beats": lambda: music.binaural_beats(NAN, duration=0.1),
    "frequency_modulation": lambda: music.frequency_modulation(
        220, NAN, duration=0.1),
    "Bonds": lambda: music.Bonds().note(NAN, 0.1),
}


@pytest.mark.parametrize("call", CASES.values(), ids=CASES.keys())
def test_an_oscillator_refuses_a_phase_that_is_no_number(call):
    with music.set_sample_rate(8000), warnings.catch_warnings():
        # numpy may warn on the way, multiplying zero by infinity; what
        # matters is that the render stops rather than comes back.
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError) as refusal:
            call()
    assert str(refusal.value) == MESSAGE


def test_a_nan_frequency_read_one_entry_for_every_sample():
    """What the refusal replaces: a constant, taken from the table."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        index = np.array([math.nan, math.nan]).astype(np.int64)
    assert index[0] == index[1]


@pytest.mark.parametrize("phase", [[], [0.0, 2.7, 16383.9], [-3.5]])
def test_a_finite_phase_is_its_whole_part(phase):
    np.testing.assert_array_equal(_entries(phase),
                                  np.array(phase).astype(np.int64))
    assert _entries(phase).dtype == np.int64


@pytest.mark.parametrize("phase", [[math.nan], [0, math.inf], [-math.inf, 1]])
def test_a_phase_with_one_value_that_is_no_number_is_refused(phase):
    with pytest.raises(ValueError) as refusal:
        _entries(np.array(phase))
    assert str(refusal.value) == MESSAGE
