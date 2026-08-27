"""Where the singing engine is looked for, and what it needs.

The engine itself is a Perl program driving espeak, cloned at runtime, so it
is not available in CI. These cover everything around it: path resolution,
the system-dependency check, and the errors raised when it is absent.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import music.singing.bootstrap as bootstrap
import music.singing.paths as paths
import music.singing.perform as perform


def make_engine(directory):
    """Create a directory that `is_engine` accepts."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / paths.ENGINE_MARKER).write_text("all:\n\ttrue\n")
    return directory


@pytest.fixture
def no_legacy_clone(monkeypatch, tmp_path):
    """Pretend no engine was ever cloned into the package directory."""
    monkeypatch.setattr(paths, "_LEGACY_DIR", tmp_path / "never-created")
    monkeypatch.delenv(paths.ENV_VAR, raising=False)


# --------------------------------------------------------------------------
# Where the engine lives
# --------------------------------------------------------------------------

def test_engine_dir_honours_the_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "elsewhere"))
    assert paths.engine_dir() == tmp_path / "elsewhere"


def test_engine_dir_defaults_into_a_user_cache_directory(no_legacy_clone):
    """Regression: the engine used to be cloned into the installed package,
    which breaks on read-only installs, in containers, and for any second
    user of a shared site-packages."""
    directory = paths.engine_dir()
    package_root = Path(music_package_root())

    assert package_root not in directory.parents
    assert directory.name == "ecantorix"
    assert "music" in directory.parts


def music_package_root():
    import music
    return Path(music.__file__).resolve().parent


def test_engine_dir_prefers_an_existing_in_package_clone(monkeypatch,
                                                         tmp_path):
    """An installation set up by an older version must keep working."""
    legacy = make_engine(tmp_path / "ecantorix")
    monkeypatch.setattr(paths, "_LEGACY_DIR", legacy)
    monkeypatch.delenv(paths.ENV_VAR, raising=False)

    assert paths.engine_dir() == legacy


def test_engine_dir_ignores_a_half_finished_in_package_clone(monkeypatch,
                                                             tmp_path):
    """A directory is not an engine. An interrupted clone in the package
    must not shadow a good one in the cache."""
    partial = tmp_path / "ecantorix"
    partial.mkdir()  # no Makefile
    monkeypatch.setattr(paths, "_LEGACY_DIR", partial)
    monkeypatch.delenv(paths.ENV_VAR, raising=False)

    assert paths.engine_dir() != partial


def test_cache_dir_sits_inside_the_engine(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "engine"))
    assert paths.cache_dir() == tmp_path / "engine" / "cache"


# --------------------------------------------------------------------------
# System dependencies
# --------------------------------------------------------------------------

def test_missing_requirements_reports_absent_programs():
    with patch.object(paths.shutil, "which", return_value=None):
        assert paths.missing_requirements() == list(paths.SYSTEM_REQUIREMENTS)

    with patch.object(paths.shutil, "which", return_value="/usr/bin/thing"):
        assert paths.missing_requirements() == []


def test_require_system_dependencies_names_what_is_missing():
    """The engine cannot be pip-installed, so the error has to say what to
    install and how."""
    with patch.object(paths, "missing_requirements", return_value=["espeak"]):
        with pytest.raises(RuntimeError, match="espeak") as excinfo:
            paths.require_system_dependencies()
    assert "apt install" in str(excinfo.value)


def test_require_system_dependencies_is_quiet_when_satisfied():
    with patch.object(paths, "missing_requirements", return_value=[]):
        paths.require_system_dependencies()


# --------------------------------------------------------------------------
# setup_engine / get_engine
# --------------------------------------------------------------------------

def test_setup_engine_clones_when_absent(monkeypatch, tmp_path):
    target = tmp_path / "engine"
    monkeypatch.setenv(paths.ENV_VAR, str(target))

    with patch.object(paths, "missing_requirements", return_value=[]), \
         patch.object(bootstrap.subprocess, "run") as run:
        returned = bootstrap.setup_engine()

    run.assert_called_once()
    argv = run.call_args[0][0]
    assert argv[:2] == ["git", "clone"]
    assert argv[2] == bootstrap.REPO_URLS["http"]
    assert argv[3] == str(target)
    assert returned == str(target)


def test_setup_engine_returns_the_path_when_already_present(monkeypatch,
                                                            tmp_path):
    """It used to return None in this case, so the return value could not be
    relied on."""
    target = make_engine(tmp_path / "engine")
    monkeypatch.setenv(paths.ENV_VAR, str(target))

    with patch.object(bootstrap.subprocess, "run") as run:
        assert bootstrap.setup_engine() == str(target)
    run.assert_not_called()


def test_setup_engine_uses_ssh_when_asked(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "engine"))
    with patch.object(paths, "missing_requirements", return_value=[]), \
         patch.object(bootstrap.subprocess, "run") as run:
        bootstrap.setup_engine(method="ssh")
    assert run.call_args[0][0][2] == bootstrap.REPO_URLS["ssh"]


def test_setup_engine_rejects_an_unknown_method(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "engine"))
    with pytest.raises(ValueError, match="method not understood"):
        bootstrap.setup_engine(method="carrier-pigeon")


def test_setup_engine_reports_a_missing_program_before_cloning(monkeypatch,
                                                               tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "engine"))
    with patch.object(paths, "missing_requirements", return_value=["perl"]), \
         patch.object(bootstrap.subprocess, "run") as run:
        with pytest.raises(RuntimeError, match="perl"):
            bootstrap.setup_engine()
    run.assert_not_called()


def test_setup_engine_wraps_a_failed_clone(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "engine"))
    failure = subprocess.CalledProcessError(1, ["git"])
    with patch.object(paths, "missing_requirements", return_value=[]), \
         patch.object(bootstrap.subprocess, "run", side_effect=failure):
        with pytest.raises(RuntimeError, match="Failed to clone"):
            bootstrap.setup_engine()


def test_get_engine_explains_how_to_install(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "absent"))
    with pytest.raises(RuntimeError, match="setup_engine"):
        bootstrap.get_engine()


def test_get_engine_returns_the_directory_once_present(monkeypatch, tmp_path):
    target = make_engine(tmp_path / "engine")
    monkeypatch.setenv(paths.ENV_VAR, str(target))
    assert bootstrap.get_engine() == str(target)


# --------------------------------------------------------------------------
# sing
# --------------------------------------------------------------------------

def test_sing_says_the_engine_is_missing_rather_than_failing_obscurely(
        monkeypatch, tmp_path):
    """It used to reach `cp` and surface a CalledProcessError about a path
    the caller had never heard of."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "absent"))
    with pytest.raises(RuntimeError, match="setup_engine"):
        perform.sing()


def test_sing_wraps_a_failed_make(monkeypatch, tmp_path):
    engine = make_engine(tmp_path / "engine")
    monkeypatch.setenv(paths.ENV_VAR, str(engine))

    failure = subprocess.CalledProcessError(1, ["make"])
    with patch.object(paths, "missing_requirements", return_value=[]), \
         patch.object(perform, "write_abc"), \
         patch.object(perform.subprocess, "run", side_effect=failure):
        with pytest.raises(RuntimeError, match="Failed to build singing"):
            perform.sing()


def test_sing_names_a_directory_that_is_not_an_engine(monkeypatch, tmp_path):
    """A bare directory used to reach the Makefile copy and fail there."""
    engine = tmp_path / "engine"
    engine.mkdir()  # deliberately no Makefile
    monkeypatch.setenv(paths.ENV_VAR, str(engine))

    with pytest.raises(RuntimeError, match="has no Makefile"):
        perform.sing()


def test_sing_rejects_an_unknown_effect(monkeypatch, tmp_path):
    engine = make_engine(tmp_path / "engine")
    monkeypatch.setenv(paths.ENV_VAR, str(engine))

    with patch.object(paths, "missing_requirements", return_value=[]), \
         patch.object(perform, "write_abc"):
        with pytest.raises(ValueError, match="effect not understood"):
            perform.sing(effect="reverse-cathedral")


def test_make_test_song_gives_one_note_and_duration_per_syllable():
    """Regression: it had eight notes for seven syllables, and zip silently
    dropped the surplus."""
    import inspect
    import re

    source = inspect.getsource(bootstrap.make_test_song)
    text = re.search(r'text = "([^"]+)"', source).group(1)
    notes = re.search(r"notes = (.+)", source).group(1).split(",")
    durs = re.search(r"durs = (.+)", source).group(1).split(",")

    assert len(text.split()) == len(notes) == len(durs)
