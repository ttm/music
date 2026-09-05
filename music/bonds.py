"""Bonds between a note's characteristics, as ``eq:vinculos`` describes them.

The article's equation is a schema rather than a formula. It says the
vibrato rate, the tremolo rate, the vibrato depth and the tremolo depth may
each be given as a function of the note's frequency --
``f_vbr = f_tr = func_a(f)``, ``nu = func_b(f)``, ``V_dB = func_c(f)`` --
and then says of those functions only that they "are arbitrary and
dependent on musical intentions". No test can settle a claim like that, and
no routine can implement one function it does not name. What can be built
is the place where such a construction goes, which is this.

    >>> language = Bonds(vibrato_freq=proportional(1 / 40),
    ...                  max_pitch_dev=inversely_proportional(500))
    >>> sound = language.render([220, 440, 330], duration=0.5)

A bond is any callable of the frequency, or a constant. What each note then
sounds like is decided once, for the piece, rather than note by note --
which is what the article means by using this to build a musical language.
The piece *Bonds* is its own worked example of that.

References
----------
.. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
       representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .core.synths.envelopes import tremolo
from .core.synths.notes import note, note_with_vibrato
from .utils import horizontal_stack

__all__ = ['Bonds', 'proportional', 'inversely_proportional', 'stepped']

#: The characteristics a bond may set, and which routine reads each.
BINDABLE = ('vibrato_freq', 'max_pitch_dev', 'tremolo_freq', 'max_db_dev')


def proportional(factor: float = 1 / 100, offset: float = 0.0) \
        -> Callable[[float], float]:
    """A bond that rises with the frequency: ``factor * f + offset``.

    The article's first example, "a vibrato frequency proportional to note
    pitch".

    Parameters
    ----------
    factor : scalar
        What to multiply the frequency by.
    offset : scalar
        Added afterwards.

    Returns
    -------
    callable
        A function of one frequency, for :class:`Bonds`.

    Examples
    --------
    >>> rate = proportional(1 / 100)
    >>> rate(220), rate(440)
    (2.2, 4.4)
    """
    def bond(freq: float) -> float:
        return factor * freq + offset
    return bond


def inversely_proportional(numerator: float = 1000.0,
                           offset: float = 0.0) -> Callable[[float], float]:
    """A bond that falls with the frequency: ``numerator / f + offset``.

    The article's second example, "a tremolo depth inversely proportional
    to pitch".

    Parameters
    ----------
    numerator : scalar
        Divided by the frequency.
    offset : scalar
        Added afterwards.

    Returns
    -------
    callable
        A function of one frequency, for :class:`Bonds`.

    Raises
    ------
    ValueError
        When called with a frequency of zero.

    Examples
    --------
    >>> depth = inversely_proportional(1000)
    >>> depth(200), depth(500)
    (5.0, 2.0)
    """
    def bond(freq: float) -> float:
        if freq == 0:
            raise ValueError('an inversely proportional bond has no value '
                             'at a frequency of zero')
        return numerator / freq + offset
    return bond


def stepped(thresholds: Sequence[tuple[float, float]],
            otherwise: float = 0.0) -> Callable[[float], float]:
    """A bond that changes in steps rather than continuously.

    Takes the value of the first threshold the frequency is below, which
    is how a register-dependent decision is usually written: one thing
    below middle C, another above it.

    Parameters
    ----------
    thresholds : sequence of (scalar, scalar)
        ``(below, value)`` pairs, tried in the order given.
    otherwise : scalar
        The value for a frequency above every threshold.

    Returns
    -------
    callable
        A function of one frequency, for :class:`Bonds`.

    Examples
    --------
    >>> register = stepped([(262, 3.0), (523, 6.0)], otherwise=12.0)
    >>> register(220), register(440), register(880)
    (3.0, 6.0, 12.0)
    """
    def bond(freq: float) -> float:
        for below, value in thresholds:
            if freq < below:
                return value
        return otherwise
    return bond


class Bonds:
    """A set of relations tying a note's characteristics to its frequency.

    Each keyword is one of :data:`BINDABLE`, and its value is either a
    callable of the frequency -- the ``func_a``, ``func_b`` and ``func_c``
    of ``eq:vinculos`` -- or a constant, for a characteristic that does not
    vary. Anything left unbound keeps the synthesis routine's own default.

    Parameters
    ----------
    **relations
        The bonds, by the name of what they set.

    Raises
    ------
    ValueError
        If a keyword is not one of :data:`BINDABLE`.

    See Also
    --------
    proportional : a bond rising with the frequency.
    inversely_proportional : a bond falling with it.
    stepped : a bond that changes by register.
    note_with_vibrato : what a bound vibrato is applied through.
    tremolo : what a bound tremolo is applied through.

    Examples
    --------
    >>> language = Bonds(vibrato_freq=proportional(1 / 50),
    ...                  max_pitch_dev=2.0)
    >>> language.characteristics(440)['vibrato_freq']
    8.8
    >>> language.characteristics(440)['max_pitch_dev']
    2.0
    >>> sound = language.render([220, 440], duration=0.2)
    >>> len(sound)
    17640

    Notes
    -----
    The article gives no functions, and neither does this: what makes a set
    of bonds musical is a decision about a piece, not a fact about
    synthesis. What it fixes is that the decision is made once and applies
    to every note, which is the difference between a bond and a parameter.
    """

    def __init__(self, **relations) -> None:
        unknown = set(relations) - set(BINDABLE)
        if unknown:
            raise ValueError(
                f'{sorted(unknown)} cannot be bound; the characteristics '
                f'eq:vinculos ties to the frequency are {list(BINDABLE)}')
        self.relations = relations

    def __repr__(self) -> str:
        bound = ', '.join(sorted(self.relations))
        return f'Bonds({bound})' if bound else 'Bonds()'

    def characteristics(self, freq: float) -> dict[str, float]:
        """What every bond says about a note at `freq`.

        Parameters
        ----------
        freq : scalar
            The note's frequency in Hertz.

        Returns
        -------
        dict
            One entry per bond, with the value it gives at that frequency.
            A callable bond is called; a constant is returned as it is.

        Examples
        --------
        >>> Bonds(max_db_dev=proportional(1 / 20)).characteristics(200)
        {'max_db_dev': 10.0}
        """
        return {name: (bond(freq) if callable(bond) else bond)
                for name, bond in self.relations.items()}

    def note(self, freq: float = 220.0, duration: float = 2.0,
             sample_rate: int = 44100) -> NDArray[np.float64]:
        """One note, with every bond applied to it.

        A bound vibrato is rendered through
        :func:`~music.note_with_vibrato`, and a bound tremolo is a
        :func:`~music.tremolo` envelope over it. With neither bound this is
        :func:`~music.note`.

        Parameters
        ----------
        freq : scalar
            The note's frequency in Hertz, which every bond is a function
            of.
        duration : scalar
            Its duration in seconds.
        sample_rate : integer
            The sample rate.

        Returns
        -------
        ndarray
            The note.

        Examples
        --------
        >>> plain = Bonds().note(freq=440, duration=0.1)
        >>> np.array_equal(plain, note(freq=440, duration=0.1))
        True
        """
        values = self.characteristics(freq)
        vibrato = {key: values[key] for key in
                   ('vibrato_freq', 'max_pitch_dev') if key in values}
        tremolo_values = {key: values[key] for key in
                          ('tremolo_freq', 'max_db_dev') if key in values}

        if vibrato:
            sound = note_with_vibrato(
                freq=freq, duration=duration,
                vibrato_freq=vibrato.get('vibrato_freq', 4),
                max_pitch_dev=vibrato.get('max_pitch_dev', 2),
                sample_rate=sample_rate)
        else:
            sound = note(freq=freq, duration=duration,
                         sample_rate=sample_rate)

        if tremolo_values:
            sound = tremolo(
                duration=duration, sonic_vector=sound,
                tremolo_freq=tremolo_values.get('tremolo_freq', 2),
                max_db_dev=tremolo_values.get('max_db_dev', 10),
                sample_rate=sample_rate)
        return np.asarray(sound, dtype=np.float64)

    def render(self, freqs: Sequence[float],
               duration: float | Sequence[float] = 2.0,
               sample_rate: int = 44100) -> NDArray[np.float64]:
        """A sequence of notes, each bound to its own frequency.

        Parameters
        ----------
        freqs : sequence of scalars
            The frequencies, in order.
        duration : scalar or sequence of scalars
            One duration for every note, or one per note.
        sample_rate : integer
            The sample rate.

        Returns
        -------
        ndarray
            The notes, end to end.

        Raises
        ------
        ValueError
            If `freqs` is empty, or if a sequence of durations is not as
            long as it.

        Examples
        --------
        >>> language = Bonds(vibrato_freq=proportional(1 / 40))
        >>> len(language.render([220, 440, 330], duration=0.1))
        13230
        >>> len(language.render([220, 440], duration=[0.1, 0.3]))
        17640
        """
        freqs = list(freqs)
        if not freqs:
            raise ValueError('render needs at least one frequency')
        if isinstance(duration, (int, float)):
            durations = [float(duration)] * len(freqs)
        else:
            durations = [float(d) for d in duration]
            if len(durations) != len(freqs):
                raise ValueError(
                    f'{len(durations)} durations for {len(freqs)} '
                    f'frequencies; give one each, or one for all of them')
        return horizontal_stack(*[
            self.note(freq=freq, duration=each, sample_rate=sample_rate)
            for freq, each in zip(freqs, durations)])
