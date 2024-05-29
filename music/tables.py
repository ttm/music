"""Provides primary tables for waveform lookup.

This module contains the `PrimaryTables` class, which allows the creation of
sine, triangle, square, and saw wave periods with a given number of samples.
It also provides a method to visualize these waveform tables.

Example:
    To create and visualize waveform tables:

    >>> from musisc import PrimaryTables
    >>> primary_tables = PrimaryTables()
    >>> primary_tables.draw_tables()

Classes:
    - PrimaryTables: Provides primary tables for waveform lookup.
"""
import numpy as np
import pylab as p


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
        self.sine = np.sin(np.linspace(0, 2 * np.pi, size, endpoint=False))
        self.saw = np.linspace(-1, 1, size)
        self.square = np.hstack((np.ones(size // 2) * -1, np.ones(size // 2)))
        foo = np.linspace(-1, 1, size // 2, endpoint=False)
        self.triangle = np.hstack((foo, foo * -1))

    def draw_tables(self):
        """Draw waveform tables."""
        p.plot(self.sine, "-o")
        p.plot(self.saw, "-o")
        p.plot(self.square, "-o")
        p.plot(self.triangle, "-o")
        p.xlim(-self.size * 0.1, self.size * 1.1)
        p.ylim(-1.1, 1.1)
        p.show()
