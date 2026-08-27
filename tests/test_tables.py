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
