"""Retiring a name without breaking the code that uses it.

The package had no way to remove anything. `mix2`, `mix_with_offset_` and
`read_wav` are second names for routines that have better ones, kept bound
so existing code keeps working; nothing told that code it should move, so
the second names could only ever accumulate.

So a name is now retired in two steps. First it keeps working, under its
old spelling, and warns with a `DeprecationWarning` naming what replaced
it; then, no sooner than the next major release, it is removed.
CONTRIBUTING.md states the policy. `DeprecationWarning` rather than
`FutureWarning`, because the audience is the programmer, not the person
running their program: Python shows it for code run as a script and in
test runs, and hides it from the users of an application built on this.

Nothing here is a decorator. The mutation audit cannot reach a decorated
function, and the routines whose parameters are renamed are audited ones,
so they take their old names through ``**renamed`` and ask
:func:`renamed_arguments` what was given.
"""

from __future__ import annotations

import functools
import inspect
import warnings
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def alias(function: F, old_name: str) -> F:
    """`function` under a name it no longer goes by, warning when called.

    The alias calls straight through, so it takes and returns exactly
    what `function` does; :func:`inspect.signature` and the reference
    documentation see `function`'s parameters, under a docstring that
    opens by saying what to call instead.
    """
    new = f"music.{function.__name__}"

    @functools.wraps(function)
    def deprecated(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(f"music.{old_name} is deprecated; call {new}, which "
                      f"it is another name for", DeprecationWarning,
                      stacklevel=2)
        return function(*args, **kwargs)

    deprecated.__name__ = deprecated.__qualname__ = old_name
    deprecated.__doc__ = (
        f"Deprecated: another name for :func:`{new}`, which to call "
        f"instead.\n\nIt warns when called and will be removed in a major "
        f"release. What follows is :func:`{new}`'s own documentation.\n\n"
        + inspect.cleandoc(function.__doc__ or ""))
    return deprecated  # type: ignore[return-value]


def renamed_arguments(routine: str, given: dict[str, Any],
                      **renames: str) -> dict[str, Any]:
    """The arguments given under parameter names that have since changed.

    Parameters
    ----------
    routine : str
        The routine's name, for the messages.
    given : dict
        Its ``**renamed``: whatever keywords its signature did not take.
    **renames : str
        Each old parameter name, mapped to the name that replaced it.

    Returns
    -------
    dict
        The values of the old names that were given, keyed by the new
        names. Each one given warns, once a call.

    Raises
    ------
    TypeError
        If a keyword is neither a parameter nor an old name for one, in
        the words Python uses for any other unexpected keyword argument.
    """
    values: dict[str, Any] = {}
    for old, value in given.items():
        if old not in renames:
            raise TypeError(
                f"{routine}() got an unexpected keyword argument {old!r}")
        new = renames[old]
        warnings.warn(f"{routine}()'s {old!r} is now {new!r}, which says "
                      f"its unit; {old!r} still works but is deprecated",
                      DeprecationWarning, stacklevel=3)
        values[new] = value
    return values
