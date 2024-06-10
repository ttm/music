"""
Provide primary tables for lookup, including sine, triangle, square, and saw
wave periods.
"""

import numpy as np
import pylab as plt


class Basic:
    """
    Provides primary tables for lookup, including sine, triangle, square, and
    saw wave periods.
    """

    def __init__(self, size=2048):
        """
        Initializes a Basic object.

        Parameters:
            size (int, optional): The size of the tables. Defaults to 2048.
        """
        self.size = size
        self.make_tables(size)

    def make_tables(self, size):
        """
        Creates sine, triangle, square, and saw wave periods.

        Parameters:
            size (int): The size of the tables.
        """
        self.sine = np.sin(np.linspace(0, 2 * np.pi, size, endpoint=False))
        self.saw = np.linspace(-1, 1, size)
        self.square = np.hstack((np.ones(size // 2) * -1, np.ones(size // 2)))
        foo = np.linspace(-1, 1, size // 2, endpoint=False)
        self.triangle = np.hstack((foo, foo * -1))

    def draw_tables(self):
        """
        Plots the sine, triangle, square, and saw wave periods.
        """
        plt.plot(self.sine, "-o")
        plt.plot(self.saw, "-o")
        plt.plot(self.square, "-o")
        plt.plot(self.triangle, "-o")
        plt.xlim(-self.size * 0.1, self.size * 1.1)
        plt.ylim(-1.1, 1.1)
        plt.show()
