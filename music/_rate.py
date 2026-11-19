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
the asyncio task that made it, and a setting made in one never reaches
another already running: two threads rendering at two rates cannot
change each other's sound. A new thread starts at 44,100 Hz, unless the
interpreter starts threads with a copy of the context of the thread that
starts them -- Python 3.14's free-threaded build does, and any 3.14 run
with ``-X thread_inherit_context=1`` -- and then at that thread's rate. A
new asyncio task always starts at the rate of the task that made it.
"""

from __future__ import annotations

import math
from contextvars import ContextVar, Token
from numbers import Integral, Real
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
    samples = whole(rate)
    if samples is None or samples <= 0:
        raise ValueError(
            f"the sample rate must be a positive whole number of samples "
            f"per second; got {rate!r}")
    return _RateSetting(samples, _rate.set(samples))


def whole(value: Any) -> int | None:
    """`value` as an ``int`` if it is a whole number, and None if not.

    A rate of samples, or a count of them, is a whole number, and one of
    NumPy's integers is that without being an ``int``. The band-limited
    oscillators and `render_oversampled` asked ``isinstance(value, int)``,
    so a rate read out of an array, or a count that came out of one, was
    refused in words that were true of it: 48,000 "must be 8000 to 192000
    Hz". A boolean is an ``int`` to Python and counts nothing, so it is
    not whole here. Nor is a float, 48000.0 included: nothing that writes
    a file could be handed one.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value)


def resolve_rate(sample_rate: Any) -> Any:
    """`sample_rate`, or the default rate when it is None.

    The first line of every routine that takes a ``sample_rate``. Only
    None is resolved: whatever else a caller gives is the routine's to
    use or refuse, exactly as before the default could be changed.
    """
    return _rate.get() if sample_rate is None else sample_rate


def sample_count(duration: Any, sample_rate: Any) -> int:
    """How many samples `duration` seconds last at `sample_rate`.

    The floor of their product, as the article's ``eq:dur`` counts a
    duration, but of the product the caller wrote rather than of its
    binary rounding. 0.35 s at 44.1 kHz is 15,435 samples, and ``0.35 *
    44100`` is 15434.999999999998, so ``int()`` of it -- which every
    routine here took -- made the note a sample short. So did one
    duration in fourteen written to two decimals, and a line of them
    drifted behind the sequencer, which places notes by rounding. The
    product is rounded to a millionth of a sample before it is floored:
    that removes the binary error and nothing a caller could mean, since
    15,435.5 samples is still 15,435.

    A negative duration floors too, away from zero, which is how
    :func:`music.mix_with_offset` has always counted an offset back from
    the end of a sound.

    NaN and infinity are no number of samples, and are refused here for
    every routine that counts a duration: they reached ``math.floor``,
    which said "cannot convert float NaN to integer", naming neither the
    duration nor the rate, and raised ``OverflowError`` for infinity,
    which a caller catching ``ValueError`` for a bad argument missed.
    """
    product = duration * sample_rate
    if not math.isfinite(product):
        raise ValueError(
            f"a duration is a finite number of seconds, at a finite rate; "
            f"got {duration!r} s at {sample_rate!r} Hz, which is no number "
            f"of samples")
    return math.floor(round(product, 6))


def whole_count(value: Any, name: str = "number_of_samples") -> int:
    """`value` as an ``int`` count of samples, refused by `name` if not.

    A count is a whole number: an ``int``, one of NumPy's integers, or a
    float that is whole, which arithmetic on seconds gives as often as
    not. That last is where a count parts from :func:`whole`, which
    refuses 48000.0 as a rate, since a file's header holds no float. A
    fraction, NaN, infinity and a boolean are no count, and are refused.

    A negative count is returned as it is, as :func:`sample_count`
    returns a negative duration's: what a routine makes of either is the
    routine's to say.
    """
    if not isinstance(value, bool) and isinstance(value, Real):
        # An integer first, which no float need hold: 10**400 is a count.
        if isinstance(value, Integral):
            return int(value)
        if float(value).is_integer():
            return int(float(value))
    raise ValueError(
        f"{name} is a count of samples, a whole number; got {value!r}")


def render_length(duration: Any, sample_rate: Any,
                  number_of_samples: Any = None) -> int:
    """How many samples a routine renders: given, or as long as `duration`.

    `number_of_samples` when it is given, by :func:`whole_count`, and the
    samples `duration` lasts at `sample_rate` when it is None, by
    :func:`sample_count`. Every routine that takes both reads them here.

    Each used to read a count its own way, so one that was not whole was
    five things: `note` and the envelopes rounded 5.5 up to six samples,
    `silence` and the stimulation routines down to five, `adsr`, `noise`
    and the sequence routines refused even 5.0 in numpy's words, and
    `render_oversampled` refused it in its own. A note and its envelope,
    asked for one count, could differ by a sample.
    """
    if number_of_samples is None:
        return sample_count(duration, sample_rate)
    return whole_count(number_of_samples)
