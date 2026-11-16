"""The package leaves logging to the program that uses it.

`mix_with_offset` and `read_audio` each said what they were doing through
``logging.debug``, the module's own function, which speaks for the root
logger -- and which gives the root logger a handler when it has none. So
the first mix, or the first file read, configured logging for the whole
program. A script that called ``logging.basicConfig`` afterwards found it
did nothing, since it acts only on a root logger without handlers: the
level and the format it asked for were dropped, and its own INFO lines
with them.
"""

import ast
import contextlib
import logging
from pathlib import Path

import music

#: The functions of the ``logging`` module that speak for the root logger,
#: and configure it when nothing has.
ROOT_FUNCTIONS = {"debug", "info", "warning", "warn", "error", "exception",
                  "critical", "fatal", "log", "basicConfig"}


@contextlib.contextmanager
def unconfigured_root():
    """The root logger as a program that has not configured it finds it:
    with no handlers, and at its own level again afterwards.

    Entered inside a test rather than as a fixture: pytest gives the root
    logger handlers of its own as a test's call begins, after its
    fixtures are set up, and a handler added to those would not show.
    """
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    root.handlers.clear()
    try:
        yield root
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


def test_using_the_package_gives_the_root_logger_no_handler(tmp_path):
    path = str(tmp_path / "tone.wav")
    tone = music.note(440, 0.05)
    with unconfigured_root() as root:
        music.write_wav_mono(tone, path)
        music.read_audio(path)
        music.mix_with_offset(tone, tone, duration=0.01)
        music.mix_many_with_offsets(tone, 0.01, tone)
        music.cross_fade(tone, tone, duration_ms=10)
        assert root.handlers == []


def test_a_program_can_configure_logging_after_it_has_mixed():
    """As a script does that makes a sound before it sets up its log."""
    tone = music.note(440, 0.05)
    with unconfigured_root() as root:
        music.mix_with_offset(tone, tone, duration=0.01)
        logging.basicConfig(level=logging.INFO)
        assert root.level == logging.INFO


def test_no_module_speaks_for_the_root_logger():
    """A library that wants to log asks for a logger of its own name,
    ``logging.getLogger(__name__)``, which configures nothing."""
    package = Path(music.__file__).parent
    found = []
    for path in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "logging"
                    and node.func.attr in ROOT_FUNCTIONS):
                found.append(f"{path.relative_to(package)}:{node.lineno} "
                             f"logging.{node.func.attr}")
    assert found == []
