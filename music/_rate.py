"""The sample rate a routine uses when it is not given one.

Every routine here that renders, shapes or writes sound takes a
``sample_rate``, and each used to default it to 44,100 Hz on its own. A
caller working at any other rate had to pass it to every call, and a call
that did not ran at 44.1 kHz with no error: a note a little flat, an
envelope a little short. The package did it to itself three times -- the
file writers' fades, `cross_fade` and `reverb` each ran at 44.1 kHz
whatever rate they were handed -- and each was found by measuring, long
after it shipped.

So the default lives here, once. Every ``sample_rate`` now defaults to
None, which means the rate `get_sample_rate` returns, and
`set_sample_rate` changes it: for the rest of a script, or, used in a
``with`` statement, for one block.

The setting is a `contextvars.ContextVar`, so it belongs to the thread or
the asyncio task that made it. A new thread starts at 44,100 Hz rather
than inheriting another's, which is what keeps two threads rendering at
two rates from changing each other's sound.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from numbers import Integral
from typing import Any

#: The rate every routine used before the default could be changed, and
#: the one it starts at.
DEFAULT_SAMPLE_RATE = 44100

_rate: ContextVar[int] = ContextVar("music_sample_rate",
                                    default=DEFAULT_SAMPLE_RATE)


def get_sample_rate() -> int:
    """The sample rate routines use when they are not given one.

    Returns
    -------
    int
        Samples per second: 44,100, unless :func:`set_sample_rate` has
        changed it in this thread or task.

    See Also
    --------
    set_sample_rate : change it, for a script or for one block.

    Examples
    --------
    >>> get_sample_rate()
    44100
    """
    return _rate.get()


class _RateSetting:
    """What :func:`set_sample_rate` returns: a setting already in effect,
    which a ``with`` statement undoes when its block ends."""

    def __init__(self, rate: int, token: Token[int]) -> None:
        self.rate = rate
        self._token = token

    def __enter__(self) -> int:
        return self.rate

    def __exit__(self, *exc_info: Any) -> None:
        _rate.reset(self._token)


def set_sample_rate(rate: int) -> _RateSetting:
    """Set the sample rate routines use when they are not given one.

    Every routine that takes a ``sample_rate`` defaults it to None, which
    means this rate. Called on its own line, the setting lasts for the
    rest of the thread or asyncio task; as the subject of a ``with``
    statement, it lasts for the block, and the rate before it comes back
    when the block ends, however it ends. Either way it applies
    immediately, so there is no form of the call that sets nothing.

    A ``sample_rate`` given to a routine always wins over this one, and
    an object that renders later -- a :class:`Sequencer`, a
    :class:`StimulationSession`, an oscillator -- keeps the rate that was
    in effect when it was made.

    Parameters
    ----------
    rate : int
        Samples per second, a positive whole number.

    Returns
    -------
    context manager
        The setting, which a ``with`` statement undoes on exit; ``as``
        binds the rate. Use it in one ``with`` statement at most.

    Raises
    ------
    ValueError
        If ``rate`` is not a positive whole number. A fractional rate,
        or a boolean, is refused rather than truncated: nothing that
        writes a file could keep it.

    See Also
    --------
    get_sample_rate : the rate in effect.

    Examples
    --------
    >>> with set_sample_rate(48000):
    ...     len(note(duration=1))
    48000
    >>> len(note(duration=1))
    44100
    """
    if (isinstance(rate, bool) or not isinstance(rate, Integral)
            or rate <= 0):
        raise ValueError(
            f"the sample rate must be a positive whole number of samples "
            f"per second; got {rate!r}")
    rate = int(rate)
    return _RateSetting(rate, _rate.set(rate))


def resolve_rate(sample_rate: Any) -> Any:
    """`sample_rate`, or the default rate when it is None.

    The first line of every routine that takes a ``sample_rate``. Only
    None is resolved: whatever else a caller gives is the routine's to
    use or refuse, exactly as before the default could be changed.
    """
    return _rate.get() if sample_rate is None else sample_rate
