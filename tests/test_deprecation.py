"""Names on their way out: they still work, and they say so.

The package had no way to retire a name. `mix2`, `mix_with_offset_` and
`read_wav` were second names kept bound with nothing telling their
callers to move, and the parameters in milliseconds -- among parameters
in seconds -- said nothing of their unit, so `cross_fade(a, b,
duration=0.5)` crossfaded over half a millisecond. Each old spelling now
works as it did and warns, naming what replaced it; CONTRIBUTING.md says
when one may go.
"""

import inspect
import warnings

import numpy as np
import pytest

import music
from music._deprecation import renamed_arguments
from test_public_api import ZERO_ARG_EXPORTS, _callable_with_defaults

#: Each routine whose parameters were renamed, with the arguments it is
#: called with besides them, and each old name with the value to give.
RENAMED = {
    "adsr": ({}, {"attack_duration": 30, "decay_duration": 40,
                  "release_duration": 60, "to_zero": 3}),
    "adsr_stereo": ({}, {"attack_duration": 30, "decay_duration": 40,
                         "release_duration": 60, "to_zero": 3}),
    "cross_fade": ({"sonic_vector_1": np.ones(4410),
                    "sonic_vector_2": np.ones(4410) * 2},
                   {"duration": 50}),
}

NEW_NAMES = {"attack_duration": "attack_ms", "decay_duration": "decay_ms",
             "release_duration": "release_ms", "to_zero": "to_zero_ms",
             "duration": "duration_ms", "fades": "fades_ms"}


def _own(arguments):
    """The arguments with their arrays copied, so that two calls compared
    cannot see what the other did to them."""
    return {name: np.array(value) for name, value in arguments.items()}


@pytest.mark.parametrize("routine,old", [
    (routine, old) for routine, (_, olds) in RENAMED.items() for old in olds])
def test_an_old_parameter_name_still_works_and_warns(routine, old):
    function = getattr(music, routine)
    fixed, olds = RENAMED[routine]
    new = NEW_NAMES[old]
    with pytest.warns(DeprecationWarning,
                      match=rf"{routine}\(\)'s '{old}' is now '{new}'"):
        given_old = function(**_own(fixed), **{old: olds[old]})
    np.testing.assert_array_equal(given_old,
                                  function(**_own(fixed), **{new: olds[old]}))


@pytest.mark.parametrize("writer", ["write_wav_mono", "write_wav_stereo",
                                    "write_audio"])
def test_the_writers_still_take_fades(tmp_path, writer):
    sound = music.note(duration=0.2)
    if writer == "write_wav_stereo":
        sound = np.vstack((sound, sound))
    old, new = tmp_path / "old.wav", tmp_path / "new.wav"
    with pytest.warns(DeprecationWarning, match="'fades' is now 'fades_ms'"):
        getattr(music, writer)(sound, str(old), fades=(30, 60))
    getattr(music, writer)(sound, str(new), fades_ms=(30, 60))
    np.testing.assert_array_equal(music.read_audio(str(old)),
                                  music.read_audio(str(new)))


def test_a_millisecond_name_is_what_the_old_one_meant():
    """Half a second, asked for in the unit the name now states."""
    first, second = music.note(220, 1), music.note(330, 1)
    joined = music.cross_fade(first, second, duration_ms=500)
    assert len(joined) == 2 * 44100 - 22050


@pytest.mark.parametrize("routine", ["adsr", "adsr_stereo", "cross_fade",
                                     "write_wav_mono", "write_wav_stereo",
                                     "write_audio"])
def test_a_keyword_that_is_neither_name_is_still_a_type_error(routine):
    """Accepting the old names must not mean accepting any name."""
    required = RENAMED.get(routine, ({},))[0]
    with pytest.raises(TypeError,
                       match=rf"{routine}\(\) got an unexpected keyword "
                             r"argument 'atack_ms'"):
        getattr(music, routine)(**required, atack_ms=5)


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


@pytest.mark.parametrize("name", ZERO_ARG_EXPORTS)
def test_no_routine_uses_a_retired_name_itself(name):
    """A routine warning on its own defaults is the package calling one
    of its old names, which would make every caller see the warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _callable_with_defaults(name)()
