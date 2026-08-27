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
>>> primary_tables.draw_tables()

Classes in this module:

* ``PrimaryTables`` -- provides primary tables for waveform lookup.
"""
import pylab as p

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
    >>> primary_tables.draw_tables()  # Draw the waveform tables
    """
    def __init__(self, size=2048):
        """Initialize the PrimaryTables class.

        Parameters
        ----------
        size : int, optional
            The number of samples for each waveform table, by default 2048.
        """
        self.triangle = None
        self.square = None
        self.saw = None
        self.sine = None
        self.size = size
        self.make_tables(size)

    def make_tables(self, size):
        """Create waveform tables.

        Parameters
        ----------
        size : int
            The number of samples for each waveform table.
        """
        self.sine = waveform_table("sine", size)
        self.saw = waveform_table("sawtooth", size)
        self.square = waveform_table("square", size)
        self.triangle = waveform_table("triangle", size)

    def draw_tables(self):
        """Draw waveform tables."""
        p.plot(self.sine, "-o")
        p.plot(self.saw, "-o")
        p.plot(self.square, "-o")
        p.plot(self.triangle, "-o")
        p.xlim(-self.size * 0.1, self.size * 1.1)
        p.ylim(-1.1, 1.1)
        p.show()
