"""_summary_
"""
import numpy as np
import pylab as p


class Basic:
    """Provide primary tables for lookup

    create sine, triangle, square and saw wave periods
    with size samples.
    """
    def __init__(self, size=2048):
        self.triangle = None
        self.square = None
        self.saw = None
        self.sine = None
        self.size = size
        self.make_tables(size)

    def make_tables(self, size):
        """_summary_

        Parameters
        ----------
        size : _type_
            _description_
        """
        self.sine = np.sin(np.linspace(0, 2 * np.pi, size, endpoint=False))
        self.saw = np.linspace(-1, 1, size)
        self.square = np.hstack((np.ones(size // 2) * -1, np.ones(size // 2)))
        foo = np.linspace(-1, 1, size // 2, endpoint=False)
        self.triangle = np.hstack((foo, foo * -1))

    def draw_tables(self):
        """_summary_
        """
        p.plot(self.sine, "-o")
        p.plot(self.saw, "-o")
        p.plot(self.square, "-o")
        p.plot(self.triangle, "-o")
        p.xlim(-self.size*0.1, self.size*1.1)
        p.ylim(-1.1, 1.1)
        p.show()
