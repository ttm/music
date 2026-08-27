"""Download and configure the eCantorix engine."""

import subprocess

from .paths import engine_dir, is_engine, require_system_dependencies
from .perform import sing

REPO_URLS = {
    "http": "https://github.com/ttm/ecantorix",
    "ssh": "git@github.com:ttm/ecantorix.git",
}


def get_engine():
    """Return the path to the local eCantorix engine.

    Returns
    -------
    str
        The engine directory.

    Raises
    ------
    RuntimeError
        If the engine has not been cloned yet.

    See Also
    --------
    setup_engine : clones the engine.
    """
    directory = engine_dir()
    if not is_engine(directory):
        raise RuntimeError(
            f"no usable eCantorix engine at {directory}. "
            "Run 'setup_engine()' to install it."
        )
    return str(directory)


def setup_engine(method="http"):
    """Clone the eCantorix repository into the user's cache directory.

    The engine is a Perl program driving espeak through a Makefile, so it is
    cloned rather than installed from PyPI, and it needs git, make, perl and
    espeak on the system. Set ``$MUSIC_ECANTORIX_DIR`` to choose a different
    location.

    Parameters
    ----------
    method : {'http', 'ssh'}
        How to reach GitHub.

    Returns
    -------
    str
        The engine directory, whether it was just cloned or already present.

    Raises
    ------
    ValueError
        If `method` is not 'http' or 'ssh'.
    RuntimeError
        If a required external program is missing, or the clone fails.
    """
    directory = engine_dir()
    if is_engine(directory):
        return str(directory)

    if method not in REPO_URLS:
        raise ValueError(
            f"method not understood: {method!r}; "
            f"expected one of {sorted(REPO_URLS)}"
        )
    require_system_dependencies()

    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", REPO_URLS[method], str(directory)], check=True
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to clone repository: {exc}") from exc
    return str(directory)


def make_test_song():
    """Render a short sung phrase, to check the engine works.

    Returns
    -------
    ndarray
        The PCM samples of the sung phrase.
    """
    whole = 1
    half = .5
    quarter = .25
    text = "hey ma bro, why fly while dive?"
    # One note and one duration per syllable of `text`; zip would silently
    # drop a surplus note.
    notes = 7, 0, 5, 7, 11, 12, 7
    durs = half, half, quarter, quarter, whole, quarter, half
    return sing(text, notes, durs)
