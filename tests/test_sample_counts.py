"""A count of samples, read one way by every routine that takes one.

Each routine read ``number_of_samples`` its own way. `note` and the
envelopes handed it to ``np.arange``, which rounds a fraction up, so 5.5
rendered six samples; `silence`, the band-limited note and the
stimulation routines cut it down to five; `adsr`, `noise` and the
sequence routines refused even 5.0, in numpy's words; and the sequence
routines read a negative count as a slice's end, so five fewer than five
samples came back as all but the last five of a fourteen-second render.
A note and its envelope, asked for one count, could differ by a sample.

These hold every routine that takes a count to one rule: a whole number,
5.0 and NumPy's integers included, and no fraction, NaN, infinity or
boolean, each refused in the same words.
"""

import inspect
import math
from fractions import Fraction

import numpy as np
import pytest

import music
from music._rate import render_length, whole_count

RATE = 8000


def _counted():
    """Every public routine with a ``number_of_samples``."""
    names = []
    for name in music.__all__:
        routine = getattr(music, name)
        if callable(routine) and not inspect.isclass(routine):
            if "number_of_samples" in inspect.signature(routine).parameters:
                names.append(name)
    return names


COUNTED = _counted()

#: What else a routine needs to be called with a count alone.
NEEDS = {
    "render_oversampled": {"renderer": music.binaural_beats},
    "mix_with_offset": {"first_sonic_vector": np.ones(20),
                        "second_sonic_vector": np.ones(5)},
    "noise": {"seed": 1},
    "gaussian_noise": {"seed": 1},
    "modulated_noise": {"seed": 1},
}

#: The routines that take a count for each part, and how to lay one out.
LAYOUTS = {
    "louds": lambda count: [count] * 3,
    "tremolos": lambda count: ((count,) * 3, (count,) * 4),
}

#: The routines that refuse a negative count, as they refuse a negative
#: duration; every other one renders no samples for either.
STRICT = {"bandlimited_note", "render_oversampled"}


def _call(name, count):
    layout = LAYOUTS.get(name, lambda given: given)
    with music.set_sample_rate(RATE):
        return getattr(music, name)(number_of_samples=layout(count),
                                    **NEEDS.get(name, {}))


def test_the_routines_that_take_a_count_are_the_ones_held_to_it():
    # A new routine with a count joins every test below on its own; this
    # says which ones there are, so one that leaves is noticed too.
    assert set(COUNTED) == {
        "adsr", "adsr_stereo", "am", "amplitude_modulation",
        "bandlimited_note", "binaural_beats", "fade",
        "frequency_modulation", "gaussian_noise", "isochronic_tones",
        "loud", "louds", "mix_with_offset", "modulated_noise",
        "monaural_beats", "noise", "note", "note_with_doppler",
        "note_with_fm", "note_with_glissando",
        "note_with_glissando_vibrato", "note_with_phase",
        "note_with_two_vibratos", "note_with_two_vibratos_glissando",
        "note_with_vibrato", "note_with_vibrato_seq_localization",
        "note_with_vibratos_glissandos", "render_oversampled", "silence",
        "spatial_motion", "tremolo", "tremolos", "trill"}


@pytest.mark.parametrize("name", COUNTED)
@pytest.mark.parametrize("count", [5.5, -0.5, math.nan, math.inf, True,
                                   Fraction(11, 2)],
                         ids=["fraction", "negative-fraction", "nan", "inf",
                              "bool", "Fraction"])
def test_a_count_that_is_not_whole_is_refused_by_name(name, count):
    with pytest.raises(ValueError) as refusal:
        _call(name, count)
    assert str(refusal.value) == (
        f"number_of_samples is a count of samples, a whole number; got "
        f"{count!r}")


@pytest.mark.parametrize("name", COUNTED)
@pytest.mark.parametrize("count", [5.0, np.int64(5), np.float32(5),
                                   Fraction(10, 2)],
                         ids=["float", "int64", "float32", "Fraction"])
def test_a_whole_count_of_any_type_is_that_count(name, count):
    np.testing.assert_array_equal(_call(name, count), _call(name, 5))


@pytest.mark.parametrize("name", sorted(set(COUNTED) - STRICT
                                        - {"louds", "mix_with_offset"}))
def test_a_negative_count_is_no_samples(name):
    assert _call(name, -5).shape == _call(name, 0).shape
    assert _call(name, 0).shape[-1] == 0


@pytest.mark.parametrize("name", sorted(STRICT))
def test_the_band_limited_routines_refuse_a_negative_count(name):
    with pytest.raises(ValueError, match="number_of_samples"):
        _call(name, -5)


@pytest.mark.parametrize("name", ["note_with_vibrato_seq_localization",
                                  "note_with_vibratos_glissandos"])
def test_a_sequence_is_fitted_to_a_count_not_cut_from_its_end(name):
    """`_fit_to_samples` sliced the render with the count, and a slice's
    negative end counts back from the end: -5 gave all but the last five
    samples of the render, 635,035 of them for the defaults at 44.1 kHz."""
    with music.set_sample_rate(RATE):
        whole = getattr(music, name)()
        assert getattr(music, name)(number_of_samples=-5).shape == \
            whole.shape[:-1] + (0,)
        np.testing.assert_array_equal(
            getattr(music, name)(number_of_samples=7.0), whole[..., :7])


@pytest.mark.parametrize("name", ["note_with_vibrato_seq_localization",
                                  "note_with_vibratos_glissandos"])
def test_a_sequence_refuses_a_count_before_rendering(name, monkeypatch):
    # The count was read only by the slice that fits the render, after the
    # whole render was made; now it is read first.
    from music.core.synths import notes

    def no_render(*args, **kwargs):
        raise AssertionError("rendered before the count was read")

    monkeypatch.setattr(notes, "_integrate_phase", no_render)
    with pytest.raises(ValueError, match="^number_of_samples is a count"):
        getattr(music, name)(number_of_samples=5.5)


def test_a_note_and_its_envelope_asked_for_one_count_agree():
    """`note` rounded 5.5 up and `adsr` refused 5.0, so a note and the
    envelope meant for it could not both be made from one float count."""
    for count in (5.0, 44100.0, np.float64(321)):
        note = music.note(number_of_samples=count)
        assert len(note) == len(music.adsr(number_of_samples=count)) == \
            len(music.silence(number_of_samples=count)) == int(count)


def test_adsr_answers_a_negative_length_with_no_samples():
    """The one envelope that did not: a negative duration or count
    reached numpy as the sustain's length, and numpy said "negative
    dimensions are not allowed"."""
    assert music.adsr(duration=-1).shape == (0,)
    assert music.adsr(number_of_samples=-3).shape == (0,)
    assert music.adsr_stereo(duration=-1).shape == (2, 0)


@pytest.mark.parametrize("count", [5, np.int64(5), np.uint8(5), 5.0,
                                   np.float32(5), Fraction(5), -5, -5.0, 0,
                                   10 ** 400])
def test_whole_count_reads_any_whole_number(count):
    # 10**400 is past every float: an integer is read as one, never by
    # way of a float.
    assert whole_count(count) == int(count)
    assert type(whole_count(count)) is int


@pytest.mark.parametrize("count", [5.5, math.nan, -math.inf, True, False,
                                   None, "5", np.array([5]), 1e-9])
def test_whole_count_refuses_what_counts_nothing(count):
    with pytest.raises(ValueError) as refusal:
        whole_count(count, "taps")
    assert str(refusal.value) == (
        f"taps is a count of samples, a whole number; got {count!r}")


def test_render_length_counts_a_duration_when_no_count_is_given():
    assert render_length(0.35, 44100) == 15435
    assert render_length(0.35, 44100, None) == 15435
    assert render_length(0.35, 44100, 7) == 7
    assert render_length(0.35, 44100, -7.0) == -7
    with pytest.raises(ValueError, match="no number of samples"):
        render_length(math.nan, 44100)


@pytest.mark.parametrize("name,given", [
    ("number_of_samples", {"number_of_samples": 5}),
    ("durations", {"durations": 5}),
])
def test_louds_refuses_one_number_where_a_sequence_belongs(name, given):
    """One number for every transition was read as the sequence it is
    not, and refused in Python's words: an int "has no len()"."""
    with pytest.raises(ValueError) as refusal:
        music.louds(**given)
    assert str(refusal.value) == (
        f"louds takes {name} as a sequence, one for each transition; got "
        f"{given[name]!r}")


@pytest.mark.parametrize("name,given", [
    ("number_of_samples", {"number_of_samples": 5}),
    ("durations", {"durations": 5}),
    ("durations", {"durations": (3, 4, 5)}),
])
def test_tremolos_refuses_a_count_not_laid_out_in_lines(name, given):
    """One number, or a single line of them, was read as lines, and an
    int "is not iterable" named nothing."""
    with pytest.raises(ValueError) as refusal:
        music.tremolos(**given)
    assert str(refusal.value) == (
        f"tremolos takes {name} as lines of segments, a sequence of "
        f"sequences such as ((3, 4, 5), (2, 3, 7, 4)); got {given[name]!r}")


def test_mix_with_offset_refuses_an_offset_between_samples():
    """The offset was cut down to the sample below it."""
    with pytest.raises(ValueError, match="^number_of_samples is a count"):
        music.mix_with_offset(np.ones(4), np.ones(2), number_of_samples=1.5)
    np.testing.assert_array_equal(
        music.mix_with_offset(np.ones(4), np.ones(2), number_of_samples=-1.0),
        [1, 1, 1, 2, 1])
