"""Combining sonic vectors.

`mix_stereo`, `mix_many_with_offsets` (once `mix_with_offset_`) and
`resolve_stereo` had no test exercising their bodies at all, and each turned
out to have a defect.
"""

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


# --------------------------------------------------------------------------
# mix_many_with_offsets
# --------------------------------------------------------------------------

def test_mix_many_with_offsets_sums_vectors_given_without_offsets():
    out = mix_many_with_offsets(np.ones(5), np.ones(3) * 2)
    assert out.shape == (5,)
    assert np.allclose(out, [3, 3, 3, 1, 1])


def test_mix_many_with_offsets_reads_a_scalar_between_vectors_as_a_delay():
    """The arguments alternate vector, offset-in-seconds, vector..."""
    out = mix_many_with_offsets(np.ones(5), 0.5, np.ones(3) * 2)

    # Exactly: the second vector starts at sample 22,050 and is three
    # long, so the result is 22,053 -- except that it is 22,055, because
    # the offset is rounded up to a whole sample twice over. The
    # assertion used to allow two samples either way, which is enough to
    # hide the difference between those two numbers without saying which
    # one it found.
    assert out.shape[0] == 22055


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


def test_mix_many_offsets_a_stereo_sound_along_its_samples():
    stereo = np.array([[1., 1.], [2., 2.]])
    np.testing.assert_array_equal(
        music.mix_many([stereo, stereo], offset=[0, 1], sample_rate=1),
        [[1, 2, 1], [2, 4, 2]])
