"""Applying a head-related transfer function, which is a convolution.

The article says the complete localization -- height, and front against
back -- is given by an HRTF, that open databases of them exist, and that
"it is possible to apply such transfer functions in a sonic signal by
convolution (see Equation ``eq:conv``)". `music.localize_hrtf` is that
application and nothing more: the package ships no impulse responses, and
where one comes from is the part that is research.

So these check the convolution, the two-channel shape, and the properties
that make it a placement rather than a filter -- that a difference between
the ears survives, and that the geometric routines are untouched by it.

References
----------
.. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
       representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)
"""

import numpy as np
import pytest

import music

SAMPLE_RATE = 44100


def _pair(left_delay=0, right_delay=8, left_gain=1.0, right_gain=0.6,
          length=32):
    """A pair of impulse responses standing in for a measured one.

    Not an HRTF. A single delayed impulse per ear carries an interaural
    time and intensity difference and nothing else, which is exactly what
    a real response carries *in addition* to the spectral shaping that
    makes it an HRTF. It is enough to check that the convolution places a
    sound, which is what this module is about.
    """
    left, right = np.zeros(length), np.zeros(length)
    left[left_delay] = left_gain
    right[right_delay] = right_gain
    return left, right


def test_applying_a_response_is_the_convolution_of_equation_conv():
    """One convolution per ear, and nothing else done to the samples."""
    sound = music.note(freq=330, duration=0.05, sample_rate=SAMPLE_RATE)
    left, right = _pair()

    placed = music.localize_hrtf(sound, left, right, sample_rate=SAMPLE_RATE)

    assert placed.shape == (2, len(sound) + len(left) - 1)
    assert np.allclose(placed[0], np.convolve(sound, left))
    assert np.allclose(placed[1], np.convolve(sound, right))


def test_responses_of_different_lengths_are_both_accepted():
    """A measured pair need not be trimmed to a common length."""
    sound = music.note(duration=0.02, sample_rate=SAMPLE_RATE)
    left = np.array([1.0, 0.5, 0.25])
    right = np.array([0.0, 0.8])

    placed = music.localize_hrtf(sound, left, right)

    assert placed.shape == (2, len(sound) + 2)
    assert np.allclose(placed[0], np.convolve(sound, left))
    # The shorter channel is padded with the silence it does not fill.
    assert np.allclose(placed[1][:len(sound) + 1], np.convolve(sound, right))
    assert placed[1][-1] == 0.0


def test_the_placement_carries_a_delay_and_a_level_between_the_ears():
    """What makes it a placement: the two channels differ, measurably."""
    sound = music.note(freq=440, duration=0.05, sample_rate=SAMPLE_RATE)
    delay = 12
    left, right = _pair(left_delay=0, right_delay=delay,
                        left_gain=1.0, right_gain=0.5)

    placed = music.localize_hrtf(sound, left, right)

    correlation = np.correlate(placed[1], placed[0], mode="full")
    measured = int(np.argmax(correlation)) - (placed.shape[1] - 1)
    assert measured == delay

    near = np.sqrt(np.mean(placed[0] ** 2))
    far = np.sqrt(np.mean(placed[1] ** 2))
    assert far / near == pytest.approx(0.5, rel=0.02)


def test_an_identical_pair_of_responses_leaves_the_sound_in_the_middle():
    sound = music.note(duration=0.02, sample_rate=SAMPLE_RATE)
    impulse = np.array([1.0, 0.2, 0.05])

    placed = music.localize_hrtf(sound, impulse, impulse)

    assert np.array_equal(placed[0], placed[1])


def test_a_unit_impulse_pair_returns_the_sound_unchanged():
    """The identity case, so the convolution is not doing anything extra."""
    sound = music.note(duration=0.02, sample_rate=SAMPLE_RATE)
    unit = np.array([1.0])

    placed = music.localize_hrtf(sound, unit, unit)

    assert placed.shape == (2, len(sound))
    assert np.array_equal(placed[0], sound)
    assert np.array_equal(placed[1], sound)


# --------------------------------------------------------------------------
# What it refuses
# --------------------------------------------------------------------------

def test_a_stereo_source_is_refused_rather_than_guessed_at():
    stereo = np.vstack([music.note(duration=0.01)] * 2)
    with pytest.raises(ValueError, match="places a mono sound"):
        music.localize_hrtf(stereo, np.array([1.0]), np.array([1.0]))


@pytest.mark.parametrize("left, right", [
    (np.array([]), np.array([1.0])),
    (np.array([1.0]), np.array([])),
    (np.zeros((2, 3)), np.array([1.0])),
])
def test_a_response_that_is_not_one_must_say_so(left, right):
    sound = music.note(duration=0.01)
    with pytest.raises(ValueError, match="one-dimensional array"):
        music.localize_hrtf(sound, left, right)


# --------------------------------------------------------------------------
# What it does not do
# --------------------------------------------------------------------------

def test_the_geometric_routines_still_cannot_tell_front_from_back():
    """This routine does not close the gap ASSESSMENT records.

    `localize` measures azimuth from the ear axis, so a source ahead and
    one behind give the same two channels -- the cone of confusion the
    article names. Adding a way to *apply* an HRTF changes nothing about
    that, and the test that pins it must go on passing, or the claim in
    ASSESSMENT.md would have quietly become false.
    """
    tone = music.note(freq=440, duration=0.05, sample_rate=SAMPLE_RATE)
    ahead = music.localize(sonic_vector=tone, x=0.0, y=1.0)
    behind = music.localize(sonic_vector=tone, x=0.0, y=-1.0)
    assert np.array_equal(ahead, behind)


def test_the_package_ships_no_impulse_responses():
    """And this test is what would notice if one were ever added.

    Shipping a measured HRTF is a licensing question and a scientific one:
    a response measured on one head is an approximation for every other.
    If a database is ever vendored, this fails and both questions get
    asked before it goes out.
    """
    import pathlib

    package = pathlib.Path(music.__file__).parent
    data = [path for path in package.rglob("*")
            if path.suffix.lower() in {".sofa", ".mat", ".hrir", ".npz",
                                       ".wav"}]
    assert not data, (
        f"{data} look like measured data shipped inside the package; "
        f"localize_hrtf takes impulse responses from its caller")
