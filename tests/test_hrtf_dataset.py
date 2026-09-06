"""Reading head-related impulse responses out of the KEMAR measurements.

The measurements are Gardner and Martin's, made on a KEMAR mannequin at
MIT. The package does not ship them, so most of these build a synthetic
dataset in the same layout: that covers the reading, the two azimuth
conventions and the nearest-angle search without a download, and it is
what runs in CI.

The ones that need the real measurements are marked, and skip when they
are absent -- the same arrangement `singing_demo.py` has. They are the
ones that check the claim the whole feature exists for: that a source in
front and one behind no longer reach the ears identically.

References
----------
.. [1] Gardner, W. G., and Martin, K. D. "HRTF measurements of a KEMAR
       dummy-head microphone." MIT Media Lab TR #280 (1994).
"""

import numpy as np
import pytest

import music
from music import hrtf

#: How many azimuths MIT measured at each elevation. They thin towards the
#: pole, from 72 at the horizon to one at the top, and the spacing is not
#: a clean divisor of 360 at every elevation -- there are 56 at -40, not
#: the 60 that a uniform six-degree step would give -- so this is the
#: measured count rather than an arithmetic model of it.
MEASURED = {-40: 56, -30: 60, -20: 72, -10: 72, 0: 72, 10: 72, 20: 72,
            30: 60, 40: 56, 50: 45, 60: 36, 70: 24, 80: 12, 90: 1}

#: A step per elevation for the synthetic dataset below. It only has to
#: be a plausible grid, not MIT's exact one.
GRID = {-40: 6, -30: 6, -20: 5, -10: 5, 0: 5, 10: 5, 20: 5, 30: 6,
        40: 6, 50: 8, 60: 10, 70: 15, 80: 30, 90: 360}


def _write_dataset(root, elevations=hrtf.ELEVATIONS, length=512):
    """A dataset in MIT's layout, with a recognisable response per file.

    Each response is a single impulse whose position encodes the angle it
    was written for, so a test can tell which file was read rather than
    only that some file was.
    """
    for elevation in elevations:
        directory = root / "full" / f"elev{elevation}"
        directory.mkdir(parents=True)
        step = GRID[elevation]
        for mit in range(0, 360, step):
            for ear, offset in (("L", 0), ("R", 1)):
                samples = np.zeros(length, dtype=">i2")
                samples[(mit // 5 + offset) % length] = 2 ** 14
                path = directory / f"{ear}{elevation}e{mit:03d}a.dat"
                path.write_bytes(samples.tobytes())
    return root


@pytest.fixture
def dataset(tmp_path):
    return _write_dataset(tmp_path / "kemar")


# --------------------------------------------------------------------------
# The two conventions
# --------------------------------------------------------------------------

@pytest.mark.parametrize("package_azimuth, mit_azimuth", [
    (0, 90),      # this package: 0 is to the right. MIT: 90 is.
    (90, 0),      # straight ahead
    (180, 270),   # to the left
    (270, 180),   # behind
])
def test_the_two_azimuth_conventions_convert_both_ways(
        package_azimuth, mit_azimuth):
    """MIT measures from the front, clockwise; this package from the ear
    axis, counter-clockwise. Forgetting it puts every source on the wrong
    side, so the conversion is checked at the four cardinal directions."""
    assert hrtf._to_mit(package_azimuth) == mit_azimuth
    assert hrtf._from_mit(mit_azimuth) == package_azimuth
    assert hrtf._from_mit(hrtf._to_mit(package_azimuth)) == package_azimuth


def test_the_conversion_is_its_own_inverse():
    for azimuth in range(0, 360, 7):
        assert hrtf._from_mit(hrtf._to_mit(azimuth)) == azimuth


# --------------------------------------------------------------------------
# Finding the nearest measurement
# --------------------------------------------------------------------------

@pytest.mark.parametrize("asked, expected", [
    (0, 0), (4, 0), (12, 10), (-14, -10), (88, 90), (-40, -40),
    # Exactly halfway goes to the lower elevation, above and below.
    (5, 0), (-5, -10), (85, 80), (-35, -40),
])
def test_an_elevation_is_rounded_to_one_that_was_measured(asked, expected):
    assert hrtf._nearest_elevation(asked) == expected


@pytest.mark.parametrize("elevation", [-46, 96, 200, -90])
def test_an_elevation_outside_the_measured_range_is_refused(elevation):
    with pytest.raises(ValueError, match="outside the measured elevations"):
        hrtf._nearest_elevation(elevation)


def test_an_azimuth_is_rounded_to_the_nearest_measured_one(dataset):
    """And around the wrap, where 358 degrees is nearer 0 than 355.

    Taking the nearest by plain subtraction sends anything just below a
    full turn to the other end of the circle.
    """
    _, at_zero, _, _ = music.hrir(0, 90, directory=dataset)
    _, wrapped, _, _ = music.hrir(0, 90 - 358, directory=dataset)
    assert at_zero == 90
    # 358 degrees the other way is two degrees short of the same place.
    assert abs(((wrapped - 90 + 180) % 360) - 180) <= 3


def test_the_direction_actually_measured_comes_back(dataset):
    """A caller must be able to tell what they got, not what they asked."""
    elevation, azimuth, _, _ = music.hrir(33, 47, directory=dataset)
    assert elevation == 30
    assert azimuth in music.available_azimuths(30, directory=dataset)


def test_the_azimuths_available_are_reported_in_this_package_s_convention(
        dataset):
    azimuths = music.available_azimuths(0, directory=dataset)
    assert len(azimuths) == 72
    assert list(azimuths) == sorted(azimuths)
    assert all(0 <= azimuth < 360 for azimuth in azimuths)
    # Every one of them can be asked for and comes back unchanged.
    for azimuth in azimuths[::12]:
        _, got, _, _ = music.hrir(0, azimuth, directory=dataset)
        assert got == azimuth


def test_an_elevation_that_was_not_measured_is_refused(dataset):
    with pytest.raises(ValueError, match="not a measured elevation"):
        music.available_azimuths(15, directory=dataset)


# --------------------------------------------------------------------------
# Reading the files
# --------------------------------------------------------------------------

def test_the_responses_are_read_as_the_big_endian_they_are(dataset):
    """Read the other way round they are noise that clips.

    The files are 16-bit, most significant byte first. Byte order is the
    kind of thing that produces a plausible-looking array rather than an
    error, so this asserts the shape of what a correct read gives: one
    impulse, at the amplitude it was written with.
    """
    _, _, left, right = music.hrir(0, 90, directory=dataset)

    assert left.shape == right.shape == (512,)
    assert left.dtype == np.float64
    assert np.count_nonzero(left) == 1
    assert left.max() == pytest.approx(2 ** 14 / 2 ** 15)
    assert np.abs(left).max() <= 1.0


def test_the_two_ears_get_different_files(dataset):
    """L and R, which is the whole point of a pair."""
    _, _, left, right = music.hrir(0, 45, directory=dataset)
    assert not np.array_equal(left, right)
    assert int(np.argmax(right)) == int(np.argmax(left)) + 1


def test_a_missing_dataset_says_how_to_get_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="setup_hrtf"):
        music.hrir(0, 90, directory=tmp_path / "nothing")


def test_a_half_finished_dataset_is_not_taken_for_a_whole_one(tmp_path):
    """An interrupted download leaves some elevations, not none."""
    partial = _write_dataset(tmp_path / "partial", elevations=(0, 10))
    assert not hrtf.is_dataset(partial)
    assert hrtf.is_dataset(_write_dataset(tmp_path / "whole"))


def test_where_the_measurements_go_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv(hrtf.ENV_VAR, str(tmp_path))
    assert music.hrtf_dir() == tmp_path
    monkeypatch.delenv(hrtf.ENV_VAR)
    assert music.hrtf_dir() == music.utils.cache_root() / "music" / "kemar"


# --------------------------------------------------------------------------
# Fetching them
# --------------------------------------------------------------------------

def test_setup_returns_the_directory_it_already_has(dataset):
    """Fetching twice does not fetch twice."""
    assert hrtf.setup_hrtf(directory=dataset) == dataset


def test_setup_unpacks_an_archive_into_place(tmp_path, dataset):
    """The whole path, from an archive to a dataset, without the network."""
    import subprocess
    import tarfile

    archive = tmp_path / "full.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(dataset / "full", arcname="full")
    subprocess.run(("gzip", "-f", str(archive)), check=True)

    target = tmp_path / "target"
    result = hrtf.setup_hrtf(directory=target,
                             url=(tmp_path / "full.tar.gz").as_uri())

    assert result == target
    assert hrtf.is_dataset(target)
    _, _, left, _ = music.hrir(0, 90, directory=target)
    assert np.count_nonzero(left) == 1


def test_an_archive_of_the_wrong_shape_is_refused(tmp_path):
    """And leaves nothing behind that would look like a dataset."""
    import tarfile

    stray = tmp_path / "stray.txt"
    stray.write_text("not impulse responses")
    archive = tmp_path / "wrong.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(stray, arcname="stray.txt")

    target = tmp_path / "target"
    with pytest.raises(OSError, match="expected layout"):
        hrtf.setup_hrtf(directory=target, url=archive.as_uri())
    assert not hrtf.is_dataset(target)


def test_setup_says_so_when_gzip_is_missing(tmp_path, monkeypatch):
    """The archive is in the compress format, which tarfile does not read."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="gzip is needed"):
        hrtf.setup_hrtf(directory=tmp_path / "target")


# --------------------------------------------------------------------------
# The measurements themselves, when they are here
# --------------------------------------------------------------------------

real = pytest.mark.skipif(
    not hrtf.is_dataset(hrtf.hrtf_dir()),
    reason="the KEMAR measurements are not installed; run music.setup_hrtf()")


@real
def test_a_source_in_front_and_one_behind_no_longer_sound_the_same():
    """The gap this feature exists to close.

    `localize` measures azimuth from the ear axis and so cannot tell them
    apart: ahead and behind give identical channels, and a test in
    test_hrtf.py asserts that it still does. A measured response does not
    have that symmetry, because an ear does not.
    """
    _, _, front_left, front_right = music.hrir(0, 90)
    _, _, back_left, back_right = music.hrir(0, 270)

    assert not np.array_equal(front_left, back_left)
    assert not np.array_equal(front_right, back_right)
    assert np.abs(front_left - back_left).max() > 0.05


@real
def test_elevation_changes_the_response():
    """The other cue the geometric routines do not carry."""
    _, _, horizon, _ = music.hrir(0, 90)
    _, _, above, _ = music.hrir(40, 90)
    assert np.abs(horizon - above).max() > 0.05


@real
def test_a_source_to_the_right_is_louder_in_the_right_ear():
    """The convention check that matters: which side is which.

    This package puts azimuth 0 to the right. If the conversion into MIT's
    convention were wrong, this is what would show it.
    """
    _, _, left, right = music.hrir(0, 0)
    assert np.sqrt(np.mean(right ** 2)) > 2 * np.sqrt(np.mean(left ** 2))

    _, _, left, right = music.hrir(0, 180)
    assert np.sqrt(np.mean(left ** 2)) > 2 * np.sqrt(np.mean(right ** 2))


@real
def test_the_measurements_are_the_grid_the_documentation_describes():
    for elevation in music.ELEVATIONS:
        azimuths = music.available_azimuths(elevation)
        assert len(azimuths) == MEASURED[elevation]
        assert len(set(azimuths)) == len(azimuths)
    _, _, left, _ = music.hrir(0, 90)
    assert left.shape == (512,)


@real
def test_a_measured_response_places_a_sound_through_localize_hrtf():
    """The two halves together: a direction, and a sound put in it."""
    tone = music.note(freq=440, duration=0.1)
    _, _, left, right = music.hrir(elevation=0, azimuth=0)

    placed = music.localize_hrtf(tone, left, right)

    assert placed.shape == (2, len(tone) + 511)
    assert np.isfinite(placed).all()
    # Placed to the right, so the right channel carries more of it.
    assert np.sqrt(np.mean(placed[1] ** 2)) > np.sqrt(np.mean(placed[0] ** 2))


def test_a_download_that_is_not_an_archive_is_reported_as_that(tmp_path):
    """gzip's own complaint, rather than a traceback from tarfile."""
    not_an_archive = tmp_path / "notes.txt"
    not_an_archive.write_text("this is not a compressed tar")

    with pytest.raises(OSError, match="could not decompress"):
        hrtf.setup_hrtf(directory=tmp_path / "target",
                        url=not_an_archive.as_uri())


def test_fetching_over_a_half_finished_directory_replaces_it(tmp_path,
                                                             dataset):
    """An interrupted fetch leaves some elevations; the next one clears them.

    Unpacking into a directory that already holds part of a dataset would
    otherwise leave whichever files the new archive did not overwrite, and
    a mixture of two datasets is worse than either.
    """
    import subprocess
    import tarfile

    archive = tmp_path / "full.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(dataset / "full", arcname="full")
    subprocess.run(("gzip", "-f", str(archive)), check=True)

    target = _write_dataset(tmp_path / "target", elevations=(0, 10))
    stray = target / "full" / "elev0" / "leftover.dat"
    stray.write_bytes(b"\x00\x00")
    assert not hrtf.is_dataset(target)

    hrtf.setup_hrtf(directory=target, url=(tmp_path / "full.tar.gz").as_uri())

    assert hrtf.is_dataset(target)
    assert not stray.exists()
