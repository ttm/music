"""Where the eCantorix engine lives, and what it needs to run.

The engine used to be cloned into the installed package directory. That fails
on a read-only install, in a container, and for any second user of a shared
site-packages, and a ``pip`` upgrade discards it. It is now kept in the
user's cache directory instead.
"""

import os
import shutil
import sys
from pathlib import Path

#: Set this to put the engine somewhere specific.
ENV_VAR = "MUSIC_ECANTORIX_DIR"

#: External programs eCantorix shells out to. It is Perl driving espeak
#: through a Makefile, so none of these can be pip-installed.
SYSTEM_REQUIREMENTS = ("git", "make", "perl", "espeak")

_LEGACY_DIR = Path(__file__).resolve().parent / "ecantorix"

#: eCantorix drives its own build from this file, so its presence is what
#: distinguishes a usable clone from an empty or half-finished directory.
ENGINE_MARKER = "Makefile"


def is_engine(directory) -> bool:
    """Whether `directory` holds a usable eCantorix clone.

    Parameters
    ----------
    directory : path-like
        The directory to check.

    Returns
    -------
    bool
        True when the directory exists and contains the engine's Makefile.
    """
    directory = Path(directory)
    return directory.is_dir() and (directory / ENGINE_MARKER).is_file()


def _cache_root() -> Path:
    """The platform's per-user cache directory."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home())
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")


def engine_dir() -> Path:
    """Return the directory the eCantorix engine lives in.

    Resolved fresh on each call, in order:

    1. ``$MUSIC_ECANTORIX_DIR``, if set.
    2. A *usable* clone inside the package, so an installation set up by an
       older version keeps working. A half-finished directory there is
       ignored rather than preferred over a good one in the cache.
    3. The per-user cache directory, which is where new clones go.

    Returns
    -------
    Path
        The engine directory. It need not exist yet.

    See Also
    --------
    music.singing.setup_engine : clones the engine into this directory.
    """
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser()
    if is_engine(_LEGACY_DIR):
        return _LEGACY_DIR
    return _cache_root() / "music" / "ecantorix"


def cache_dir() -> Path:
    """Return eCantorix's own scratch directory, inside the engine."""
    return engine_dir() / "cache"


def missing_requirements() -> list[str]:
    """Return the names of the external programs that are not installed.

    Returns
    -------
    list of str
        A subset of SYSTEM_REQUIREMENTS, empty when everything is present.

    Examples
    --------
    >>> missing = missing_requirements()
    >>> isinstance(missing, list)
    True
    """
    return [name for name in SYSTEM_REQUIREMENTS if shutil.which(name) is None]


def require_system_dependencies() -> None:
    """Raise RuntimeError naming whatever external program is missing.

    Raises
    ------
    RuntimeError
        If any of SYSTEM_REQUIREMENTS is not on PATH.
    """
    missing = missing_requirements()
    if missing:
        raise RuntimeError(
            "the singing engine needs these programs, which are not "
            f"installed: {', '.join(missing)}. On Debian or Ubuntu: "
            f"sudo apt install {' '.join(missing)}. On macOS with Homebrew: "
            f"brew install {' '.join(missing)}."
        )
