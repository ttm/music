"""Provides primary tables for waveform lookup.

This module contains the `PrimaryTables` class, which allows the creation of
sine, triangle, square, and saw wave periods with a given number of samples.
It also provides a method to visualize these waveform tables.

Examples
--------
To create and visualize waveform tables:

>>> from music import PrimaryTables
>>> PrimaryTables.__module__  # confirm correct package name
'music.tables'
>>> primary_tables = PrimaryTables()
>>> primary_tables.draw_tables()  # doctest: +SKIP

Classes in this module:

* ``PrimaryTables`` -- provides primary tables for waveform lookup.
"""
from .utils import waveform_table


class PrimaryTables:
    """Provides primary tables for waveform lookup.

    This class creates sine, triangle, square, and saw wave periods
    with a given number of samples.

    Parameters
    ----------
    size : int, optional
        The number of samples for each waveform table, by default 2048.

    Attributes
    ----------
    sine : ndarray
        The sine wave table.
    triangle : ndarray
        The triangle wave table.
    square : ndarray
        The square wave table.
    saw : ndarray
        The sawtooth wave table.
    size : int
        The number of samples for each waveform table.

    Examples
    --------
    >>> primary_tables = PrimaryTables()
    >>> primary_tables.size
    2048
    >>> primary_tables.draw_tables()  # doctest: +SKIP
    """
    def __init__(self, size=2048):
        """Initialize the PrimaryTables class.

        Parameters
        ----------
        size : int, optional
            The number of samples for each waveform table, by default 2048.
        """
        self.make_tables(size)

    def make_tables(self, size):
        """Create waveform tables.

        Parameters
        ----------
        size : int
            The number of samples for each waveform table. It becomes
            :attr:`size`, which used to keep the size the object was made
            with, so the axes :meth:`draw_tables` drew fitted neither.
        """
        self.size = size
        self.sine = waveform_table("sine", size)
        self.saw = waveform_table("sawtooth", size)
        self.square = waveform_table("square", size)
        self.triangle = waveform_table("triangle", size)

    def draw_tables(self):
        """Draw the four waveform tables on one pair of axes, each named.

        Raises
        ------
        ImportError
            If matplotlib is not installed. Install it with
            ``pip install 'music[plot]'``.

        Notes
        -----
        matplotlib is imported here rather than at the top of the module.
        It is the only thing in the package that needs it, and importing
        it eagerly cost about 40% of ``import music``'s total time --
        1237 ms against 743 ms -- and pulled eight further packages into
        every installation, for a function that only exists to look at
        the tables.

        The examples are marked ``+SKIP``: ``show()`` blocks on a window,
        so a doctest run would hang rather than fail.

        Examples
        --------
        >>> PrimaryTables(size=64).draw_tables()  # doctest: +SKIP

        """
        try:
            from matplotlib import pyplot
        except ImportError:
            raise ImportError(
                "draw_tables needs matplotlib, which is not installed. "
                "Install it with: pip install 'music[plot]'") from None

        # pyplot rather than pylab, which matplotlib discourages; and a
        # legend, without which four lines of one style were four
        # unnamed lines.
        pyplot.plot(self.sine, "-o", label="sine")
        pyplot.plot(self.saw, "-o", label="sawtooth")
        pyplot.plot(self.square, "-o", label="square")
        pyplot.plot(self.triangle, "-o", label="triangle")
        pyplot.xlim(-self.size * 0.1, self.size * 1.1)
        pyplot.ylim(-1.1, 1.1)
        pyplot.legend()
        pyplot.show()
