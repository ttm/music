"""What the `tables` mutation audit found untested or wrong."""

import sys
import types

import numpy as np

import music
from music import PrimaryTables


def _drawn(monkeypatch, tables):
    """Every call draw_tables makes, in order, to a stand-in pylab."""
    calls = []
    fake = types.ModuleType("pylab")
    for name in ("plot", "xlim", "ylim", "show"):
        setattr(fake, name,
                lambda *args, _name=name, **kwargs:
                calls.append((_name, args, kwargs)))
    monkeypatch.setitem(sys.modules, "pylab", fake)
    tables.draw_tables()
    return calls


def test_the_tables_are_drawn_in_order_on_axes_that_fit_them(monkeypatch):
    tables = PrimaryTables(size=40)
    calls = _drawn(monkeypatch, tables)
    names = [name for name, _, _ in calls]
    assert names == ["plot"] * 4 + ["xlim", "ylim", "show"]
    for (_, args, _), table in zip(calls, (tables.sine, tables.saw,
                                          tables.square, tables.triangle)):
        assert args[0] is table and args[1:] == ("-o",)
    assert calls[4][1] == (-4.0, 44.0)
    assert calls[5][1] == (-1.1, 1.1)
    assert calls[6][1:] == ((), {})


def test_remaking_the_tables_remakes_their_size(monkeypatch):
    """The size stayed what the object was made with, so the axes were
    drawn for tables of another length."""
    tables = PrimaryTables(size=8)
    tables.make_tables(20)
    assert tables.size == 20
    for kind, table in (("sine", tables.sine), ("sawtooth", tables.saw),
                        ("square", tables.square),
                        ("triangle", tables.triangle)):
        np.testing.assert_array_equal(table, music.waveform_table(kind, 20))
    assert _drawn(monkeypatch, tables)[4][1] == (-2.0, 22.0)


def test_bare_tables_have_2048_samples():
    tables = PrimaryTables()
    assert tables.size == 2048 and len(tables.sine) == 2048


def test_without_matplotlib_the_message_says_what_to_install(monkeypatch):
    import pytest
    monkeypatch.setitem(sys.modules, "pylab", None)
    with pytest.raises(ImportError, match=(
            r"^draw_tables needs matplotlib, which is not installed\. "
            r"Install it with: pip install 'music\[plot\]'$")):
        PrimaryTables(size=8).draw_tables()
