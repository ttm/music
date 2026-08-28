"""Combining sonic vectors.

`mix_stereo`, `mix_with_offset_` and `resolve_stereo` had no test exercising
their bodies at all, and each turned out to have a defect.
"""

import numpy as np
import pytest

import music
from music.utils import mix_stereo, mix_with_offset_, resolve_stereo


# --------------------------------------------------------------------------
# mix_stereo
# --------------------------------------------------------------------------

def test_mix_stereo_promotes_mono_to_both_channels():
    mono = np.array([1.0, 2.0, 3.0])
    out = mix_stereo(mono)
    assert out.shape == (2, 3)
    # Mixed with itself, so each channel is doubled.
    assert np.allclose(out[0], 2 * mono)
    assert np.allclose(out[1], 2 * mono)


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


def test_mix_stereo_does_not_mistake_a_two_sample_mono_for_stereo():
    """Regression: the stereo test was `len(x) != 2`, and a mono vector of
    two samples also has len 2. Its 'channels' were then scalars, and
    len() of a scalar raises."""
    out = mix_stereo(np.array([1.0, 2.0]), np.zeros(3))
    assert out.shape == (2, 3)
    assert np.allclose(out[0][:2], [1.0, 2.0])


# --------------------------------------------------------------------------
# mix_with_offset_
# --------------------------------------------------------------------------

def test_mix_with_offset_sums_vectors_given_without_offsets():
    out = mix_with_offset_(np.ones(5), np.ones(3) * 2)
    assert out.shape == (5,)
    assert np.allclose(out, [3, 3, 3, 1, 1])


def test_mix_with_offset_reads_a_scalar_between_vectors_as_a_delay():
    """The arguments alternate vector, offset-in-seconds, vector..."""
    out = mix_with_offset_(np.ones(5), 0.5, np.ones(3) * 2)
    assert out.shape[0] == pytest.approx(0.5 * 44100 + 3, abs=2)


def test_mix_with_offset_accepts_any_sequence():
    """Regression: an exact `type(a) not in (np.ndarray, list)` check
    rejected tuples, though the parameters are array_like."""
    from_tuples = mix_with_offset_((1.0,) * 5, (2.0,) * 3)
    from_lists = mix_with_offset_([1.0] * 5, [2.0] * 3)
    assert np.allclose(from_tuples, from_lists)


def test_mix_with_offset_rejects_a_scalar_where_a_vector_belongs():
    with pytest.raises(ValueError, match="sequence of numbers"):
        mix_with_offset_(3.0)


def test_mix_with_offset_passes_a_single_vector_through():
    assert np.allclose(mix_with_offset_(np.ones(4)), np.ones(4))


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
