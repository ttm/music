"""What `import music` is allowed to cost.

Importing the package should not import a computer algebra system. The
permutation and change-ringing structures need sympy, and importing
`sympy.combinatorics` runs `sympy/__init__.py` first, which pulls in
`sympy.polys` and the rest -- about half of `import music`, paid by every
caller including the majority who only synthesize sound.

`music/__init__.py` defers those exports behind PEP 562 `__getattr__`.
That is easy to undo by accident: one `from .structures import ...`
added at the top of any eagerly imported module puts the cost straight
back, with nothing failing. These tests are the thing that fails.
"""

import subprocess
import sys

import pytest

import music


def run(code):
    """Run `code` in a clean interpreter and return its stdout."""
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_importing_music_does_not_import_sympy():
    """The regression this guards is silent: everything still works, only
    slower, so only a test notices."""
    assert run("import sys, music; print('sympy' in sys.modules)") == "False"


def test_touching_a_structure_imports_sympy():
    """The other half of the bargain: deferred, not removed."""
    assert run("import sys, music; music.Peals; "
               "print('sympy' in sys.modules)") == "True"


@pytest.mark.parametrize("name", sorted(music._LAZY_STRUCTURES))
def test_every_deferred_name_resolves(name):
    assert getattr(music, name) is not None


def test_from_import_reaches_the_deferred_names():
    """`from music import Peals` goes through __getattr__ too, and a
    reader would not expect it to behave differently from an attribute."""
    assert run("from music import Peals, dist; print(Peals.__name__)") \
        == "Peals"


def test_a_name_that_does_not_exist_still_raises_attribute_error():
    """A typo must fail as a missing attribute, not as an import error
    from somewhere the caller has never heard of."""
    with pytest.raises(AttributeError, match="no attribute 'nonesuch'"):
        music.nonesuch


def test_dir_still_lists_the_deferred_names():
    """Otherwise tab completion quietly loses half the structures API."""
    listed = dir(music)
    assert music._LAZY_STRUCTURES <= set(listed)
    assert "note" in listed
    assert listed == sorted(listed)
