"""A warning names the line that called the package, whatever the route.

Python shows a warning with the line it names, and shows it once for each
line. The package's warnings counted frames to find that line, a count
that is right for one route in only: `pan_transitions` handing a sound
to `convert_to_stereo`, `PlainChanges` calling `initialize_hunts`, and
`read_wav` calling `read_audio` each named a line inside the package, so
the warning was shown once, there, and hidden from every caller after.
"""

import dis
from pathlib import Path

import numpy as np
import pytest

import music
from music.structures import PlainChanges


def _at_48k(tmp_path):
    """A file at a rate other than the default, which reading warns of."""
    path = str(tmp_path / "48k.wav")
    music.write_wav_mono(music.note(duration=0.01), path, sample_rate=48000)
    return path


#: Each route to each of the package's warnings, called from this file.
ROUTES = {
    "convert_to_stereo": lambda tmp_path: music.convert_to_stereo(
        np.ones((3, 8))),
    "pan_transitions": lambda tmp_path: music.pan_transitions(
        sonic_vector=np.ones((3, 44100 * 6))),
    "resolve_stereo": lambda tmp_path: music.resolve_stereo(
        music.loud, {"sonic_vector": np.ones((3, 100))}),
    "read_audio": lambda tmp_path: music.read_audio(_at_48k(tmp_path)),
    "read_wav": lambda tmp_path: music.read_wav(_at_48k(tmp_path)),
    "localize2": lambda tmp_path: music.localize2(
        sonic_vector=music.note(440, 0.05), method="brute"),
    "PlainChanges, hunts": lambda tmp_path: PlainChanges(4, hunts=1),
    "PlainChanges, nhunts": lambda tmp_path: PlainChanges(5, 4),
    "initialize_hunts": lambda tmp_path: PlainChanges().initialize_hunts(
        5, 4),
    "adsr, an old name": lambda tmp_path: music.adsr(attack_duration=10),
    "mix2": lambda tmp_path: music.mix2([music.note(duration=0.01)]),
}


@pytest.mark.parametrize("route", ROUTES)
def test_a_warning_names_the_line_that_called_the_package(route, tmp_path,
                                                          recwarn):
    ROUTES[route](tmp_path)
    lines = {line for _, line in dis.findlinestarts(ROUTES[route].__code__)}
    assert recwarn.list
    assert [(warning.filename, warning.lineno in lines)
            for warning in recwarn] == [(__file__, True)] * len(recwarn)


def test_the_package_warns_only_through_its_helper():
    """A warning given any other way would count frames again."""
    package = Path(music.__file__).parent
    assert [str(path.relative_to(package))
            for path in sorted(package.rglob("*.py"))
            if "warnings.warn(" in path.read_text()] == ["_warn.py"]
