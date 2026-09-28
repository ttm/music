"""What the `hrtf` mutation audit found untested or wrong."""

import math

import pytest

from music import hrtf
from test_hrtf_dataset import _write_dataset


@pytest.fixture
def dataset(tmp_path):
    return _write_dataset(tmp_path / "kemar")


@pytest.mark.parametrize("elevation", [math.nan, math.inf, -math.inf])
def test_an_elevation_that_is_not_a_number_is_refused(dataset, elevation):
    """NaN compared false with every measurement, so the search kept the
    first, and hrir read -40 degrees."""
    with pytest.raises(ValueError, match=f"^{elevation} is outside the "
                       "measured elevations"):
        hrtf.hrir(elevation=elevation, directory=dataset)


@pytest.mark.parametrize("azimuth", [math.nan, math.inf])
def test_an_azimuth_that_is_not_a_number_is_refused(dataset, azimuth):
    """It failed in round() with a message about converting to an int."""
    with pytest.raises(ValueError, match=f"^azimuth must be a finite "
                       f"number of degrees; got {azimuth}$"):
        hrtf.hrir(azimuth=azimuth, directory=dataset)


def test_an_elevation_out_of_range_names_the_range():
    with pytest.raises(ValueError, match="^95.1 is outside the measured "
                       "elevations, which run from -40 to 90 degrees in "
                       "steps of 10$"):
        hrtf._nearest_elevation(95.1)


# --------------------------------------------------------------------------
# Which files are read. The paths are recorded rather than found, because
# on a case-insensitive filesystem "FULL" and "l" find the same files
# ---------------------------------------------------------------------------

def test_the_azimuths_come_from_the_directory_given(dataset):
    """Not the default one, which on a machine with the real measurements
    has 56 at -40 where this synthetic grid has 60."""
    assert hrtf.available_azimuths(-40, directory=dataset) == tuple(sorted(
        hrtf._from_mit(angle) for angle in range(0, 360, 6)))


def test_every_path_read_is_the_one_mit_s_layout_names(dataset,
                                                         monkeypatch):
    import pathlib
    import numpy as np

    listed, checked, read = [], [], []
    original_glob, original_is_dir = pathlib.Path.glob, pathlib.Path.is_dir
    monkeypatch.setattr(pathlib.Path, "glob", lambda self, pattern: (
        listed.append((self.relative_to(dataset), pattern))
        or original_glob(self, pattern)))
    monkeypatch.setattr(pathlib.Path, "is_dir", lambda self: (
        checked.append(self) or original_is_dir(self)))
    original_fromfile = np.fromfile
    monkeypatch.setattr(hrtf.np, "fromfile", lambda path, dtype: (
        read.append(pathlib.Path(path).relative_to(dataset))
        or original_fromfile(path, dtype=dtype)))

    hrtf.available_azimuths(-40, directory=dataset)
    hrtf.hrir(elevation=-40, azimuth=90, directory=dataset)

    assert listed == [(pathlib.Path("full/elev-40"), "L*.dat")] * 2
    assert {path.relative_to(dataset) for path in checked} == {
        pathlib.Path(f"full/elev{elevation}")
        for elevation in hrtf.ELEVATIONS}
    assert read == [pathlib.Path("full/elev-40/L-40e000a.dat"),
                    pathlib.Path("full/elev-40/R-40e000a.dat")]


def test_halfway_across_the_wrap_goes_to_the_lower_angle(dataset):
    """At -40 the synthetic grid is six degrees, so MIT's 357 is three
    from 354 and three from 0 the other way round."""
    _, azimuth, _, _ = hrtf.hrir(elevation=-40,
                                 azimuth=hrtf._from_mit(357),
                                 directory=dataset)
    assert azimuth == hrtf._from_mit(0)


# --------------------------------------------------------------------------
# setup_hrtf, without the network
# --------------------------------------------------------------------------

def _archive(tmp_path, dataset):
    import subprocess
    import tarfile

    archive = tmp_path / "full.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(dataset / "full", arcname="full")
    subprocess.run(("gzip", "-f", str(archive)), check=True)
    return tmp_path / "full.tar.gz"


def test_setup_creates_every_missing_directory_above_the_target(tmp_path,
                                                                dataset):
    target = tmp_path / "a" / "b" / "c" / "kemar"
    assert hrtf.setup_hrtf(directory=target,
                           url=_archive(tmp_path, dataset).as_uri()) == target
    assert hrtf.is_dataset(target)


def test_setup_downloads_beside_the_target_with_a_timeout(tmp_path,
                                                          dataset,
                                                          monkeypatch):
    """Beside it, so that the finished directory is moved into place on
    one filesystem rather than copied there from another."""
    import io
    import urllib.request

    payload = _archive(tmp_path, dataset).read_bytes()
    target = tmp_path / "cache" / "kemar"
    seen = []

    def fake_urlopen(url, timeout):
        seen.append((url, timeout,
                     sorted(path.name for path in target.parent.iterdir())))
        return io.BytesIO(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    hrtf.setup_hrtf(directory=target, url="https://example.invalid/k.tar.Z")
    (url, timeout, beside), = seen
    assert (url, timeout) == ("https://example.invalid/k.tar.Z", 120)
    assert len(beside) == 1 and beside[0] != "kemar"
    assert sorted(path.name for path in target.parent.iterdir()) == [
        "kemar"]


def test_a_failed_decompression_reports_what_gzip_said(tmp_path, dataset,
                                                       monkeypatch):
    """Even when gzip's message is not UTF-8."""
    import subprocess
    import types

    archive = _archive(tmp_path, dataset)
    commands = []

    def fake_run(command, stdout, stderr):
        commands.append(command)
        return types.SimpleNamespace(returncode=1,
                                     stderr=b"\xff not in compress format\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    target = tmp_path / "target"
    with pytest.raises(OSError, match="^could not decompress full.tar.Z: "
                       "� not in compress format$"):
        hrtf.setup_hrtf(directory=target, url=archive.as_uri())
    assert commands[0][:2] == ("gzip", "-dc")
    assert commands[0][2].endswith("full.tar.Z")
    assert not target.exists()


def test_an_archive_that_writes_outside_its_directory_is_refused(tmp_path):
    """The archive comes from the network, so it is unpacked with the
    'data' filter, which refuses a member that climbs out."""
    import subprocess
    import tarfile

    payload = tmp_path / "escape.txt"
    payload.write_text("outside")
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(payload, arcname="../../escaped.txt")
    subprocess.run(("gzip", "-f", str(archive)), check=True)
    target = tmp_path / "deep" / "target"
    with pytest.raises(tarfile.OutsideDestinationError):
        hrtf.setup_hrtf(directory=target,
                        url=(tmp_path / "evil.tar.gz").as_uri())
    assert not (tmp_path / "escaped.txt").exists()
    assert not (tmp_path / "deep" / "escaped.txt").exists()


def test_without_gzip_the_message_says_what_to_do(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr("shutil.which",
                        lambda name: asked.append(name) or None)
    with pytest.raises(RuntimeError, match=(
            r"^gzip is needed to unpack the KEMAR archive, which is in the "
            r"compress format that Python's tarfile does not read\. Install "
            r"gzip, or unpack the archive yourself and point "
            r"\$MUSIC_HRTF_DIR at the directory holding \"full/\"$")):
        hrtf.setup_hrtf(directory=tmp_path / "target")
    assert asked == ["gzip"]


def test_a_bare_hrir_is_straight_ahead_on_the_horizon(dataset):
    elevation, azimuth, _, _ = hrtf.hrir(directory=dataset)
    assert (elevation, azimuth) == (0, 90)
