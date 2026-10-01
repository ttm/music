"""Where the eCantorix engine lives, and what it needs to run.

The engine used to be cloned into the installed package directory. That fails
on a read-only install, in a container, and for any second user of a shared
site-packages, and a ``pip`` upgrade discards it. It is now kept in the
user's cache directory instead.
"""

import os
import shutil
import subprocess
from pathlib import Path

from ..utils import cache_root

#: Set this to put the engine somewhere specific.
ENV_VAR = "MUSIC_ECANTORIX_DIR"

#: External programs eCantorix shells out to. It is Perl driving espeak
#: through a Makefile, so none of these can be pip-installed. Its Makefile
#: turns the score into MIDI with abc2midi, and the script shapes each
#: syllable with sox; this list used to leave both out, so the check
#: passed and the render failed inside make.
SYSTEM_REQUIREMENTS = ("git", "make", "perl", "espeak", "abc2midi", "sox")

#: The Perl modules the script loads that Perl does not ship with.
PERL_MODULES = ("MIDI", "Math::FFT", "URI::Escape", "Digest::SHA")

#: The package that provides a program, where it is not named after it,
#: for the install command the error suggests.
_PACKAGES = {"abc2midi": "abcmidi"}

_LEGACY_DIR = Path(__file__).resolve().parent / "ecantorix"

#: Said whenever eCantorix lacks something: sing() needs none of it.
_DEFAULT_NEEDS_NONE = ("sing's default backend, psola, needs none of "
                       "these: only espeak-ng and pip install "
                       "'music[singing]'.")

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


#: Kept as a name here because this module has always had one; the
#: decision it makes is shared with `music.hrtf` and lives in `utils`.
_cache_root = cache_root


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


def missing_perl_modules() -> list[str]:
    """Return the Perl modules the ``perl`` on PATH cannot load.

    Returns
    -------
    list of str
        A subset of PERL_MODULES, empty when every one loads, or when
        there is no ``perl`` to ask: :func:`missing_requirements` reports
        that.
    """
    if shutil.which("perl") is None:
        return []
    return [module for module in PERL_MODULES
            if subprocess.run(["perl", f"-M{module}", "-e1"],
                              capture_output=True).returncode]


def require_system_dependencies() -> None:
    """Raise RuntimeError naming whatever the engine needs and lacks.

    Raises
    ------
    RuntimeError
        If any of SYSTEM_REQUIREMENTS is not on PATH, or the ``perl`` on
        PATH cannot load one of PERL_MODULES.
    """
    missing = missing_requirements()
    if missing:
        packages = ' '.join(_PACKAGES.get(name, name) for name in missing)
        raise RuntimeError(
            "the eCantorix engine needs these programs, which are not "
            f"installed: {', '.join(missing)}. On Debian or Ubuntu: "
            f"sudo apt install {packages}. On macOS with Homebrew: "
            f"brew install {packages}. {_DEFAULT_NEEDS_NONE}"
        )
    modules = missing_perl_modules()
    if modules:
        raise RuntimeError(
            "the eCantorix engine's Perl script needs these modules, which "
            f"the perl on PATH cannot load: {', '.join(modules)}. Install "
            f"them with: cpan {' '.join(modules)}. {_DEFAULT_NEEDS_NONE}"
        )
