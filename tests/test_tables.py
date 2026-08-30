"""The primary waveform lookup tables."""

import numpy as np
import pytest

from music.tables import PrimaryTables
from music.utils import WAVEFORMS, waveform_table


def test_primary_tables_shapes():
    pt = PrimaryTables(size=16)
    assert pt.sine.shape == (16,)
    assert pt.triangle.shape == (16,)
    assert pt.square.shape == (16,)
    assert pt.saw.shape == (16,)
    # sine first element is 0 and last is close to -step
    assert np.isclose(pt.sine[0], 0.0)


def test_primary_tables_records_its_size():
    assert PrimaryTables(size=64).size == 64


@pytest.mark.parametrize("size", [4, 15, 16, 2048, 2049])
def test_primary_tables_holds_exactly_the_size_requested(size):
    """Regression: built by halves, the square and triangle came back one
    sample short whenever size was odd."""
    pt = PrimaryTables(size=size)
    for name in ("sine", "saw", "square", "triangle"):
        assert len(getattr(pt, name)) == size, name


@pytest.mark.parametrize("kind, attribute", list(zip(
    WAVEFORMS, ("sine", "saw", "square", "triangle"))))
def test_primary_tables_delegates_to_the_one_generator(kind, attribute):
    """There used to be three copies of these four definitions."""
    pt = PrimaryTables(size=256)
    assert np.array_equal(getattr(pt, attribute), waveform_table(kind, 256))


def test_draw_tables_plots_all_four(monkeypatch):
    """draw_tables is a convenience for looking at the tables; check it
    reaches the plotting calls rather than opening a window.

    pylab is substituted in ``sys.modules`` rather than patched as a
    module attribute, because the import happens inside the function --
    matplotlib is no longer imported when ``music`` is.
    """
    import sys
    import types

    plotted, shown = [], []
    fake = types.ModuleType("pylab")
    fake.plot = lambda data, *a, **k: plotted.append(len(data))
    fake.xlim = lambda *a, **k: None
    fake.ylim = lambda *a, **k: None
    fake.show = lambda: shown.append(True)
    monkeypatch.setitem(sys.modules, "pylab", fake)

    PrimaryTables(size=32).draw_tables()

    assert plotted == [32, 32, 32, 32]
    assert shown == [True]


def test_draw_tables_says_what_to_install_when_matplotlib_is_absent(
        monkeypatch):
    """matplotlib is an extra now, so the one function that needs it has
    to say so rather than raise a bare ImportError from an import line."""
    import sys

    # None in sys.modules makes the import statement raise ImportError,
    # which is how the absent-matplotlib install behaves.
    monkeypatch.setitem(sys.modules, "pylab", None)

    with pytest.raises(ImportError, match=r"pip install 'music\[plot\]'"):
        PrimaryTables(size=8).draw_tables()
