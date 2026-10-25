"""Names on their way out: they still work, and they say so.

The package had no way to retire a name. `mix2`, `mix_with_offset_` and
`read_wav` were second names kept bound with nothing telling their
callers to move. Each old spelling now works as it did and warns, naming
what replaced it; CONTRIBUTING.md says when one may go.
"""

import inspect
import warnings

import pytest

import music
from music._deprecation import renamed_arguments
from test_public_api import ZERO_ARG_EXPORTS, _callable_with_defaults


def test_renamed_arguments_maps_only_what_was_given():
    with pytest.warns(DeprecationWarning):
        assert renamed_arguments("f", {"a": 1}, a="b", c="d") == {"b": 1}
    assert renamed_arguments("f", {}, a="b") == {}
    with pytest.raises(TypeError,
                       match=r"f\(\) got an unexpected keyword argument 'x'"):
        renamed_arguments("f", {"x": 1}, a="b")


@pytest.mark.parametrize("old,new", [("mix2", "mix_many"),
                                     ("mix_with_offset_",
                                      "mix_many_with_offsets"),
                                     ("read_wav", "read_audio")])
def test_an_alias_takes_what_its_replacement_takes(old, new):
    alias, replacement = getattr(music, old), getattr(music, new)
    assert inspect.signature(alias) == inspect.signature(replacement)
    assert alias.__name__ == old
    assert inspect.getdoc(alias).startswith(
        f"Deprecated: another name for :func:`music.{new}`")


def test_an_alias_warns_where_it_is_called():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        music.mix2([music.note(duration=0.01)])
    assert [warning.filename for warning in caught] == [__file__]
    assert issubclass(caught[0].category, DeprecationWarning)


@pytest.mark.parametrize("name", ZERO_ARG_EXPORTS)
def test_no_routine_uses_a_retired_name_itself(name):
    """A routine warning on its own defaults is the package calling one
    of its old names, which would make every caller see the warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _callable_with_defaults(name)()
