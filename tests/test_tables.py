import importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    path = HERE / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

tables = load_module('tables', 'music/tables.py')


def test_primary_tables_shapes():
    pt = tables.PrimaryTables(size=16)
    assert pt.sine.shape == (16,)
    assert pt.triangle.shape == (16,)
    assert pt.square.shape == (16,)
    assert pt.saw.shape == (16,)
    # sine first element is 0 and last is close to -step
    assert np.isclose(pt.sine[0], 0.0)

