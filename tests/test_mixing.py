"""Combining sonic vectors.

`mix_stereo`, `mix_many_with_offsets` (once `mix_with_offset_`) and
`resolve_stereo` had no test exercising their bodies at all, and each turned
out to have a defect.
"""

import re
import time

import numpy as np
import pytest

import music
from music.utils import (mix_many_with_offsets, mix_stereo,
                         resolve_stereo)


# --------------------------------------------------------------------------
# mix_stereo
# --------------------------------------------------------------------------

def test_mix_stereo_promotes_mono_to_both_channels():
    """Mixed with nothing, as it was before a cleanup in 2024 mixed it
    with itself, which this test then pinned: each channel doubled."""
    mono = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(mix_stereo(mono), [mono, mono])
    stereo = np.array([mono, -mono])
    np.testing.assert_array_equal(mix_stereo(stereo), stereo)


@pytest.mark.parametrize("end", [False, True])
def test_mix_stereo_pads_the_shorter_vector(end):
    """`end` chooses which side the padding goes on."""
    longer = np.ones(5)
    shorter = np.ones(3) * 2

    out = mix_stereo(longer, shorter, end=end)

    assert out.shape == (2, 5)
    if end:
        # the short one is pushed to the back
        assert np.allclose(out[0], [1, 1, 3, 3, 3])
    else:
        assert np.allclose(out[0], [3, 3, 3, 1, 1])


def test_mix_stereo_handles_either_argument_being_longer():
    a, b = np.ones(5), np.ones(3) * 2
    assert mix_stereo(a, b).shape == (2, 5)
    assert mix_stereo(b, a).shape == (2, 5)


def test_mix_stereo_keeps_a_real_stereo_vector_as_is():
    stereo = np.vstack((np.ones(4), np.ones(4) * 3))
    out = mix_stereo(stereo, np.zeros(4))
    assert np.allclose(out[0], 1.0)
    assert np.allclose(out[1], 3.0)


@pytest.mark.parametrize("end", [False, True])
def test_mix_stereo_keeps_a_stereo_second_vector_as_is(end):
    first = np.array([[1.0, 2.0], [10.0, 20.0]])
    second = np.array([[100.0, 200.0, 300.0],
                       [1000.0, 2000.0, 3000.0]])

    out = mix_stereo(first, second, end=end)

    expected = ([[100.0, 201.0, 302.0],
                 [1000.0, 2010.0, 3020.0]] if end else
                [[101.0, 202.0, 300.0],
                 [1010.0, 2020.0, 3000.0]])
    np.testing.assert_array_equal(out, expected)


def test_mix_stereo_does_not_mistake_a_two_sample_mono_for_stereo():
    """Regression: the stereo test was `len(x) != 2`, and a mono vector of
    two samples also has len 2. Its 'channels' were then scalars, and
    len() of a scalar raises."""
    out = mix_stereo(np.array([1.0, 2.0]), np.zeros(3))
    assert out.shape == (2, 3)
    assert np.allclose(out[0][:2], [1.0, 2.0])


@pytest.mark.parametrize("first, second", [
    (np.ones((3, 5)), np.ones((2, 4))),
    (np.ones((2, 5)), np.ones((3, 4))),
    (np.ones((3, 5)), None),
])
def test_mix_stereo_refuses_a_third_channel(first, second):
    """As every routine that keeps two channels does. numpy refused it
    in its own words, a broadcast between a pair and a triple, even with
    no second sound to mix."""
    with pytest.raises(ValueError, match="^" + re.escape(
            "mix_stereo takes a mono or a stereo sound; got an array of "
            "shape ") + r"\(3, \d\)\. convert_to_stereo folds more channels "
                        r"into two$"):
        mix_stereo(first, second)


def test_mix_stereo_puts_a_single_row_in_both_channels():
    out = mix_stereo(np.array([[1.0, 2.0, 3.0]]), np.zeros(2))
    np.testing.assert_array_equal(out, [[1, 2, 3], [1, 2, 3]])


@pytest.mark.parametrize("sound", [np.zeros(0), np.zeros((2, 0))])
def test_mix_stereo_of_one_empty_sound_is_an_empty_pair(sound):
    """No second sound is a pair of no samples, which pads nothing: a
    mutant that made it one sample long survived the audit, since every
    sound mixed alone here had samples of its own."""
    assert mix_stereo(sound).shape == (2, 0)


# --------------------------------------------------------------------------
# mix_many_with_offsets
# --------------------------------------------------------------------------

def test_mix_many_with_offsets_sums_vectors_given_without_offsets():
    out = mix_many_with_offsets(np.ones(5), np.ones(3) * 2)
    assert out.shape == (5,)
    assert np.allclose(out, [3, 3, 3, 1, 1])


def test_mix_many_with_offsets_reads_a_scalar_between_vectors_as_a_delay():
    """The arguments alternate vector, offset-in-seconds, vector..., and
    an offset delays the vector before it, as the docstring says.

    This comment once read the 0.5 as the second vector's delay and
    explained the length by a rounding "twice over": the second vector
    starting at sample 22,050 would end at 22,053. It is the first that
    is delayed, five samples from 22,050 to 22,054, which is 22,055 with
    no rounding at all, and the second has no offset after it and starts
    at the beginning.
    """
    out = mix_many_with_offsets(np.ones(5), 0.5, np.ones(3) * 2)

    assert out.shape[0] == 22055
    np.testing.assert_array_equal(out[22050:], np.ones(5))
    np.testing.assert_array_equal(out[:3], [2, 2, 2])
    assert not out[3:22050].any()


def test_mix_many_with_offsets_accepts_any_sequence():
    """Regression: an exact `type(a) not in (np.ndarray, list)` check
    rejected tuples, though the parameters are array_like."""
    from_tuples = mix_many_with_offsets((1.0,) * 5, (2.0,) * 3)
    from_lists = mix_many_with_offsets([1.0] * 5, [2.0] * 3)
    assert np.allclose(from_tuples, from_lists)


def test_mix_many_with_offsets_rejects_a_scalar_where_a_vector_belongs():
    with pytest.raises(ValueError, match="sequence of numbers"):
        mix_many_with_offsets(3.0)


def test_mix_many_with_offsets_passes_a_single_vector_through():
    assert np.allclose(mix_many_with_offsets(np.ones(4)), np.ones(4))


# --------------------------------------------------------------------------
# resolve_stereo
# --------------------------------------------------------------------------

def test_resolve_stereo_applies_a_mono_function_per_channel():
    stereo = np.vstack((np.ones(64), np.ones(64) * 2))

    out = resolve_stereo(music.fade, {"sonic_vector": stereo})

    assert out.shape == (2, 64)
    # Each channel is the mono result for that channel.
    assert np.allclose(out[0], music.fade(sonic_vector=stereo[0]))
    assert np.allclose(out[1], music.fade(sonic_vector=stereo[1]))


def test_resolve_stereo_promotes_a_mono_argument_first():
    mono = np.ones(64)
    out = resolve_stereo(music.fade, {"sonic_vector": mono})
    assert out.shape == (2, 64)
    assert np.allclose(out[0], out[1])


def test_resolve_stereo_does_not_replace_the_callers_arguments():
    mono = np.array([1.0, 2.0, 3.0])
    arguments = {"sonic_vector": mono}

    resolve_stereo(music.fade, arguments)

    assert arguments["sonic_vector"] is mono
    np.testing.assert_array_equal(arguments["sonic_vector"], [1.0, 2.0, 3.0])


def test_the_old_mixer_names_still_work_and_say_what_replaced_them():
    """`mix2` and `mix_with_offset_` said nothing about what they did or
    how they differed from `mix_with_offset`. Both were renamed, and both
    stay bound so that existing callers keep working -- now warning that
    the name they use is going."""
    first, second = music.note(220, 0.1), music.note(330, 0.05)
    with pytest.warns(DeprecationWarning, match=r"music\.mix_many\b"):
        mixed = music.mix2([first, second])
    np.testing.assert_array_equal(mixed, music.mix_many([first, second]))
    with pytest.warns(DeprecationWarning,
                      match=r"music\.mix_many_with_offsets"):
        offset = music.mix_with_offset_(first, 0.02, second)
    np.testing.assert_array_equal(
        offset, music.mix_many_with_offsets(first, 0.02, second))


def test_no_sounds_mix_or_stack_to_no_samples():
    """`horizontal_stack()` failed in numpy, and `mix_many([])` in
    Python's `max()`, where `mix([], [])` and `mix_many_with_offsets()`
    gave nothing for nothing."""
    assert music.horizontal_stack().shape == (0,)
    assert music.mix_many([]).shape == (0,)


@pytest.mark.parametrize("end", [False, True])
def test_mix_many_mixes_stereo_sounds_of_any_length(end):
    """It measured each sound with len(), which is a stereo sound's
    channel count: two of different lengths failed in numpy, and so did a
    mono one with a stereo one."""
    stereo = np.array([[1., 1., 1., 1.], [2., 2., 2., 2.]])
    mono = np.array([10., 20.])
    padded = [0, 0, 10, 20] if end else [10, 20, 0, 0]
    np.testing.assert_array_equal(music.mix_many([stereo, mono], end=end),
                                  stereo + np.array([padded, padded]))


@pytest.mark.parametrize("order", [1, -1])
def test_mix_many_puts_mono_in_both_channels_whichever_comes_first(order):
    """The mix is made in the shape of the sounds once they are all
    stereo. Made in the first sound's shape without that, it would hold
    a stereo sound only when one came first: the audit's mutant that
    skipped the conversion survived every test that put stereo first."""
    mono, stereo = np.ones(3), np.array([[1., 2.], [3., 4.]])
    np.testing.assert_array_equal(music.mix_many([mono, stereo][::order]),
                                  [[2, 3, 1], [4, 5, 1]])


def test_mix_many_offsets_a_stereo_sound_along_its_samples():
    stereo = np.array([[1., 1.], [2., 2.]])
    np.testing.assert_array_equal(
        music.mix_many([stereo, stereo], offset=[0, 1], sample_rate=1),
        [[1, 2, 1], [2, 4, 2]])


def test_mix_many_costs_the_length_of_the_mix_not_a_copy_per_sound():
    """Regression: each sound was padded to the length of the whole mix
    and the copies summed, so 600 notes of a tenth of a second at their
    offsets across a minute held some 12 GB and took 25 seconds.

    Four times the sounds, over four times the span, cost about 11x, and
    costs 4x now. The threshold is loose because this is wall-clock on
    shared CI.
    """
    note = np.ones(80)

    def best_of_three(count):
        sounds, offsets = [note] * count, [i * .01 for i in range(count)]
        times = []
        for _ in range(3):
            start = time.perf_counter()
            music.mix_many(sounds, offset=offsets, sample_rate=8000)
            times.append(time.perf_counter() - start)
        return min(times)

    best_of_three(10)  # warm up
    small, large = best_of_three(200), best_of_three(800)
    assert large / small < 8, (
        f"four times the sounds cost {large / small:.1f}x, which looks "
        f"quadratic rather than linear")


@pytest.mark.parametrize("end", [False, True])
def test_mix_many_places_each_sound_where_its_offset_and_end_put_it(end):
    """Aligned by its end, a sound's offset is silence before it that
    moves only the start of the mix, when it makes it the longest."""
    mixed = music.mix_many([np.ones(2), np.full(3, 2.0), np.full(1, 4.0)],
                           end=end, offset=[0, 2], sample_rate=1)
    expected = ([0, 0, 2, 2 + 1, 2 + 1 + 4] if end
                else [1 + 4, 1, 2, 2, 2])
    np.testing.assert_array_equal(mixed, expected)


def test_mix_many_takes_an_array_of_offsets_as_it_takes_a_list():
    """`if offset:` asked an array for one truth value, and numpy refused
    to give it one."""
    sounds = [np.ones(3), np.full(2, 2.0)]
    np.testing.assert_array_equal(
        music.mix_many(sounds, offset=np.array([0, 2]), sample_rate=1),
        music.mix_many(sounds, offset=[0, 2], sample_rate=1))


def test_mix_many_starts_a_sound_without_an_offset_at_zero():
    np.testing.assert_array_equal(
        music.mix_many([np.ones(2), np.ones(2), np.ones(2)], offset=[1],
                       sample_rate=1), [2, 3, 1])


@pytest.mark.parametrize("offset, message", [
    (1, "offset is a list of offsets in seconds, one for each sound; got "
        "the number 1"),
    ([0, 1, 2], "3 offsets for 2 sounds; give one for each sound, or "
                "fewer, and the rest start at zero"),
    ([0, -1], "an offset delays a sound, so it cannot be negative; offset "
              "1 is -1"),
])
def test_mix_many_refuses_offsets_it_cannot_place(offset, message):
    """Each failed in Python's words or numpy's: "'int' object is not
    iterable", "list index out of range" and "negative dimensions are not
    allowed"."""
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        music.mix_many([np.ones(3), np.ones(2)], offset=offset,
                       sample_rate=1)


# --------------------------------------------------------------------------
# mix_with_offset: how far back an offset reaches
# --------------------------------------------------------------------------

def test_an_offset_back_to_the_first_sounds_start_starts_them_together():
    first, second = np.ones(10), np.arange(4.0)
    mixed = music.mix_with_offset(first, second, number_of_samples=-10)
    np.testing.assert_array_equal(mixed, first + np.r_[second, np.zeros(6)])


@pytest.mark.parametrize("given", [{"number_of_samples": -11},
                                   {"duration": -11 / 44100}])
@pytest.mark.parametrize("first", [np.ones(10), np.ones((2, 10))])
def test_an_offset_past_the_first_sounds_start_is_refused(given, first):
    """It was read from the end of the mix, as a negative index is, and
    failed in numpy with a message about broadcasting."""
    with pytest.raises(ValueError, match=re.escape(
            "an offset of -11 samples starts the second sound before the "
            "first one, which is 10 samples long: counting back from its "
            "end, an offset reaches its start and no further")):
        music.mix_with_offset(first, np.arange(4.0), **given)
