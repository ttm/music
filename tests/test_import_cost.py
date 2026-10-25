"""What `import music` is allowed to cost, and what it needs installed.

Importing the package should not import a computer algebra system. The
permutation and change-ringing structures are built from sympy's
permutation groups, and importing `sympy.combinatorics` runs
`sympy/__init__.py` first, which pulls in `sympy.polys` and the rest --
about half of `import music`, paid by every caller including the majority
who only synthesize sound.

So sympy is the `structures` extra, and the structures import it when one
is built rather than when their modules are imported. That is easy to
undo by accident: one `from sympy ...` at the top of a structures module
puts the cost straight back, and breaks `import music` wherever the extra
is not installed, with nothing failing here but these tests.
"""

import builtins
import subprocess
import sys

import pytest

import music
from music.structures import _sympy

#: The names the structures export, which have to resolve with or without
#: sympy installed.
STRUCTURES = ('dist', 'GenericPeal', 'InterestingPermutations', 'Peals',
              'PlainChanges', 'print_peal', 'transpose_permutation')

#: Python, started with sympy unimportable: what `pip install music`
#: without the extra gives.
WITHOUT_SYMPY = "import sys; sys.modules['sympy'] = None; "


def run(code):
    """Run `code` in a clean interpreter and return its stdout."""
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_importing_music_does_not_import_sympy():
    """The regression this guards is silent: everything still works, only
    slower, so only a test notices."""
    assert run("import sys, music; print('sympy' in sys.modules)") == "False"


def test_reaching_a_structure_does_not_import_sympy_either():
    """The names, the module that holds them, and `from music import`:
    none of them is a structure being built."""
    assert run("import sys, music; from music import Peals, dist; "
               "music.structures.peals.PlainChanges; "
               "print('sympy' in sys.modules)") == "False"


def test_building_a_structure_is_what_imports_sympy():
    """The other half of the bargain: deferred, not removed."""
    assert run("import sys, music; music.PlainChanges(4); "
               "print('sympy' in sys.modules)") == "True"


@pytest.mark.parametrize("name", STRUCTURES)
def test_every_structure_is_exported(name):
    assert name in music.__all__
    assert getattr(music, name) is not None


def test_without_sympy_every_name_still_imports():
    """`from music import *` is how the docstrings are written to be read,
    so an install without the extra must still be able to do it."""
    assert run(WITHOUT_SYMPY + "from music import *; "
               "print(Peals.__name__, PlainChanges.__name__)") \
        == "Peals PlainChanges"


def test_without_sympy_building_a_structure_says_what_to_install():
    assert "music[structures]" in run(
        WITHOUT_SYMPY + "import music\n"
        "try:\n    music.PlainChanges(4)\n"
        "except ImportError as exc:\n    print(exc)")


def test_the_install_hint_is_given_in_this_process_too(monkeypatch):
    """The subprocess tests above run where coverage cannot see them."""
    for name in [name for name in sys.modules
                 if name.startswith('sympy.')]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, 'sympy', None)
    with pytest.raises(ImportError, match=r"music\[structures\]"):
        _sympy.combinatorics()


def test_a_sympy_that_is_there_and_broken_is_not_reported_as_missing(
        monkeypatch):
    """Installing sympy would not fix a sympy missing one of its own
    dependencies, so that failure is raised as it is."""
    real_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name.startswith('sympy'):
            raise ModuleNotFoundError("No module named 'mpmath'",
                                      name='mpmath')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', broken)
    with pytest.raises(ModuleNotFoundError, match='mpmath'):
        _sympy.combinatorics()


def test_print_peal_says_what_to_install_without_its_colours(monkeypatch):
    monkeypatch.setitem(sys.modules, 'termcolor', None)
    with pytest.raises(ImportError, match=r"music\[structures\]"):
        music.print_peal([[0, 1]])


def test_printing_a_peal_leaves_standard_output_alone():
    """colorama's `init()` used to run on import and replace
    `sys.stdout` for the whole process with a wrapper that strips colour
    from anything not sent to a terminal."""
    assert run("import sys, music; before = sys.stdout; "
               "music.print_peal([[0, 1]]); "
               "print(sys.stdout is before)").splitlines()[-1] == "True"


def test_a_name_that_does_not_exist_still_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'nonesuch'"):
        music.nonesuch


def test_dir_lists_the_structures():
    """Otherwise tab completion quietly loses half the structures API."""
    listed = dir(music)
    assert set(STRUCTURES) <= set(listed)
    assert 'structures' in listed


def test_the_structures_submodule_resolves_as_an_attribute():
    """`music.structures.peals.PlainChanges` is how three of the examples
    reach it."""
    assert run("import music; print(music.structures.__name__)") \
        == "music.structures"
