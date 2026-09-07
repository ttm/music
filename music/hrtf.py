"""Head-related impulse responses, from the KEMAR measurements.

A head-related transfer function is the filtering a head, pinnae and torso
apply to a sound arriving from a particular direction. It is what carries
the two cues :func:`~music.localize` and :func:`~music.localize2` cannot:
the elevation of a source, and whether it is in front of the listener or
behind. Those routines model a head as two points on an axis, so a source
ahead and one behind reach the ears identically; a measured response does
not have that symmetry, because an ear does not.

The measurements are Gardner and Martin's, made on a KEMAR mannequin at
MIT: 710 directions, 512 samples each at 44.1 kHz, at a fixed distance of
1.4 metres. **This package does not ship them.** They are about 1.3 MB and
freely redistributable, and :func:`setup_hrtf` fetches them into the user's
cache directory once::

    >>> setup_hrtf()                                   # doctest: +SKIP
    >>> _, _, left, right = hrir(elevation=30, azimuth=45)  # doctest: +SKIP
    >>> placed = localize_hrtf(note(), left, right)    # doctest: +SKIP

That is the same arrangement `music.singing` uses for eCantorix, and for
the same reason: a large external resource does not belong inside
`site-packages`, which an upgrade replaces.

A response measured on a mannequin is an approximation for any particular
listener, whose own ears differ. It is much better than no HRTF at all and
it is not the listener's own.

References
----------
.. [1] Gardner, W. G., and Martin, K. D. "HRTF measurements of a KEMAR
       dummy-head microphone." MIT Media Lab Perceptual Computing
       Technical Report #280 (1994).
       https://sound.media.mit.edu/resources/KEMAR.html
.. [2] Fabbri, Renato, et al. "Musical elements in the discrete-time
       representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .utils import cache_root

__all__ = ['KEMAR_URL', 'ELEVATIONS', 'hrtf_dir', 'is_dataset',
           'setup_hrtf', 'available_azimuths', 'hrir']

#: Where the measurements come from.
KEMAR_URL = 'https://sound.media.mit.edu/resources/KEMAR/full.tar.Z'

#: Set this to use a copy of the dataset from somewhere else.
ENV_VAR = 'MUSIC_HRTF_DIR'

#: The elevations measured, in degrees, with 90 straight up. The
#: measurements thin out towards the pole: 72 azimuths at the horizon and
#: one at the top, because there is only one direction there.
ELEVATIONS: tuple[int, ...] = (-40, -30, -20, -10, 0, 10, 20, 30, 40, 50,
                               60, 70, 80, 90)

#: The responses are 16-bit, most significant byte first. Read the other
#: way round they come out as noise that clips, which is what a wrong
#: byte order looks like rather than a wrong ear.
_DTYPE = '>i2'
_FULL_SCALE = 2 ** 15


def hrtf_dir() -> Path:
    """Where the measurements are, or would go.

    Returns
    -------
    Path
        ``$MUSIC_HRTF_DIR`` if it is set, and otherwise a directory in the
        per-user cache. It need not exist yet.

    See Also
    --------
    setup_hrtf : puts them there.

    Examples
    --------
    >>> str(hrtf_dir()).endswith("kemar")
    True

    Setting ``$MUSIC_HRTF_DIR`` overrides it, which is how to use a copy
    of the measurements from somewhere else. This example does not set it,
    because a docstring that changes the environment changes it for
    whatever runs next.
    """
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser()
    return cache_root() / 'music' / 'kemar'


def is_dataset(directory: str | Path) -> bool:
    """Whether `directory` holds the measurements, rather than some of them.

    Parameters
    ----------
    directory : path
        The directory to test.

    Returns
    -------
    bool
        True when every elevation is present. A half-finished download
        leaves some of them, and reporting that as a dataset would turn a
        missing file into an error a long way from its cause.

    Examples
    --------
    >>> is_dataset("/nowhere")
    False
    """
    root = Path(directory) / 'full'
    return all((root / f'elev{elevation}').is_dir()
               for elevation in ELEVATIONS)


def setup_hrtf(directory: str | Path | None = None,
               url: str = KEMAR_URL, force: bool = False) -> Path:
    """Fetch the KEMAR measurements into the cache, once.

    Parameters
    ----------
    directory : path, optional
        Where to put them. Defaults to :func:`hrtf_dir`.
    url : string
        Where to get them.
    force : boolean
        Fetch them again even if they are already there.

    Returns
    -------
    Path
        The directory they are in.

    Raises
    ------
    RuntimeError
        If ``gzip`` is not installed. The archive is in the old ``compress``
        format, which Python's standard library does not read and ``gzip``
        does; there is no pure-Python fallback here.
    OSError
        If the download or the extraction fails. Nothing is left behind:
        the archive is unpacked beside the target and moved into place only
        once it is whole, so an interrupted fetch does not leave a
        directory that :func:`is_dataset` would accept.

    See Also
    --------
    hrir : reads one direction out of them.
    music.singing.setup_engine : the same arrangement, for the singing
                                 engine.

    Examples
    --------
    >>> setup_hrtf()                        # doctest: +SKIP
    PosixPath('.../music/kemar')

    Notes
    -----
    About 1.3 MB, fetched once. The measurements are Gardner and Martin's
    and are freely redistributable; this only saves you finding them.
    """
    # Only fetching needs any of these, and `import music` should not pay
    # for a feature most callers never reach.
    import shutil
    import subprocess
    import tarfile
    import tempfile
    import urllib.request

    target = Path(directory) if directory is not None else hrtf_dir()
    if is_dataset(target) and not force:
        return target
    if shutil.which('gzip') is None:
        raise RuntimeError(
            'gzip is needed to unpack the KEMAR archive, which is in the '
            "compress format that Python's tarfile does not read. Install "
            'gzip, or unpack the archive yourself and point '
            f'${ENV_VAR} at the directory holding "full/"')

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
        staging = Path(scratch)
        compressed = staging / 'full.tar.Z'
        with urllib.request.urlopen(url, timeout=120) as response:
            compressed.write_bytes(response.read())

        plain = staging / 'full.tar'
        with open(plain, 'wb') as out:
            result = subprocess.run(('gzip', '-dc', str(compressed)),
                                    stdout=out, stderr=subprocess.PIPE)
        if result.returncode:
            raise OSError(
                f'could not decompress {compressed.name}: '
                f'{result.stderr.decode(errors="replace").strip()}')

        unpacked = staging / 'unpacked'
        with tarfile.open(plain) as archive:
            archive.extractall(unpacked, filter='data')
        if not is_dataset(unpacked):
            raise OSError(
                f'{url} did not unpack into the expected layout; expected '
                f'a "full/" directory with one subdirectory per elevation')

        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(unpacked), str(target))
    return target


def available_azimuths(elevation: int,
                       directory: str | Path | None = None) -> tuple[int, ...]:
    """The azimuths measured at one elevation, in this package's convention.

    Parameters
    ----------
    elevation : integer
        One of :data:`ELEVATIONS`.
    directory : path, optional
        Where the measurements are. Defaults to :func:`hrtf_dir`.

    Returns
    -------
    tuple of integers
        Ascending, in degrees, measured from the ear axis as everything
        else in this package is: 0 is to the right, and the angle
        increases counter-clockwise, so 90 is straight ahead.

    Raises
    ------
    FileNotFoundError
        If the measurements are not there.
    ValueError
        If `elevation` is not one of the measured ones.

    See Also
    --------
    hrir : reads one of them.

    Examples
    --------
    >>> available_azimuths(0)[:4]           # doctest: +SKIP
    (0, 5, 10, 15)
    >>> len(available_azimuths(90))         # doctest: +SKIP
    1
    """
    root = _dataset(directory)
    if elevation not in ELEVATIONS:
        raise ValueError(
            f'{elevation} is not a measured elevation; they are '
            f'{ELEVATIONS}')
    measured = (root / 'full' / f'elev{elevation}')
    mit = sorted(int(path.name[-8:-5]) for path in measured.glob('L*.dat'))
    return tuple(sorted(_from_mit(angle) for angle in mit))


def hrir(elevation: float = 0, azimuth: float = 90,
         directory: str | Path | None = None
         ) -> tuple[int, int, NDArray[np.float64], NDArray[np.float64]]:
    """The impulse responses measured nearest to one direction.

    Parameters
    ----------
    elevation : scalar
        Degrees above the horizontal, from -40 to 90. Rounded to the
        nearest measured elevation, which are ten degrees apart.
    azimuth : scalar
        Degrees, measured from the ear axis as everything else in this
        package is: 0 is to the right, 90 straight ahead, 180 to the left
        and 270 behind. Rounded to the nearest measured azimuth, which are
        between five and thirty degrees apart depending on the elevation.
    directory : path, optional
        Where the measurements are. Defaults to :func:`hrtf_dir`.

    Returns
    -------
    elevation, azimuth : integers
        The direction actually measured, which is what the responses are
        for. Read them rather than assuming you got what you asked for.
    left, right : ndarray
        512 samples each, at 44.1 kHz, scaled to [-1, 1).

    Raises
    ------
    FileNotFoundError
        If the measurements are not there. Run :func:`setup_hrtf`.
    ValueError
        If the elevation is outside the measured range.

    See Also
    --------
    localize_hrtf : convolves a sound with the pair.
    setup_hrtf : fetches the measurements.
    available_azimuths : what was measured at an elevation.

    Examples
    --------
    >>> elevation, azimuth, left, right = hrir(0, 90)  # doctest: +SKIP
    >>> left.shape                                     # doctest: +SKIP
    (512,)

    Notes
    -----
    The measurements are at a fixed distance of 1.4 metres, so they carry
    no distance cue: scale the result yourself for anything nearer or
    further.
    """
    root = _dataset(directory)
    nearest_elevation = _nearest_elevation(elevation)
    measured = root / 'full' / f'elev{nearest_elevation}'

    wanted = _to_mit(azimuth)
    angles = sorted(int(path.name[-8:-5]) for path in measured.glob('L*.dat'))
    # The measurements wrap, so 358 degrees is nearer to 0 than to 355.
    nearest = min(angles, key=lambda angle: min(abs(angle - wanted),
                                                360 - abs(angle - wanted)))

    responses = []
    for ear in ('L', 'R'):
        path = measured / f'{ear}{nearest_elevation}e{nearest:03d}a.dat'
        responses.append(
            np.fromfile(path, dtype=_DTYPE).astype(np.float64) / _FULL_SCALE)
    left, right = responses
    return nearest_elevation, _from_mit(nearest), left, right


# ---------------------------------------------------------------------
# The two conventions, and where the files are
# ---------------------------------------------------------------------

def _to_mit(azimuth: float) -> int:
    """This package's azimuth as MIT's.

    This package measures from the ear axis and counter-clockwise, with 0
    to the right. MIT measures from straight ahead and clockwise. So the
    two run in opposite directions from origins ninety degrees apart, and
    a routine that forgets it puts every source on the wrong side.
    """
    return int(round(360 - azimuth + 90)) % 360


def _from_mit(azimuth: int) -> int:
    """MIT's azimuth as this package's. The conversion is its own inverse."""
    return int(round(360 - azimuth + 90)) % 360


def _nearest_elevation(elevation: float) -> int:
    """The measured elevation nearest `elevation`, or an error.

    Anything within half a step of a measurement is rounded to it, and
    anything further out is refused rather than silently pulled to the
    nearest end. Exactly halfway goes to the lower elevation, both above
    and below the horizon: 5 rounds to 0 and -5 to -10. The MASS reference
    rounds through ``numpy.round``, whose ties go to even, so it sends
    both of those to 0; the two differ only on that knife edge and this
    rule is at least the same in both directions.
    """
    nearest = min(ELEVATIONS,
                  key=lambda measured: abs(measured - elevation))
    if abs(nearest - elevation) > 5:
        raise ValueError(
            f'{elevation} is outside the measured elevations, which run '
            f'from {ELEVATIONS[0]} to {ELEVATIONS[-1]} degrees in steps '
            f'of 10')
    return nearest


def _dataset(directory: str | Path | None) -> Path:
    root = Path(directory) if directory is not None else hrtf_dir()
    if not is_dataset(root):
        raise FileNotFoundError(
            f'the KEMAR measurements are not in {root}. Run '
            f'music.setup_hrtf() to fetch them, or set ${ENV_VAR} to a '
            f'directory holding "full/"')
    return root
