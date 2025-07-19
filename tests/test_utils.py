import importlib.util
from pathlib import Path
import numpy as np
import pytest
import warnings

HERE = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    path = HERE / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

utils = load_module('utils', 'music/utils.py')
functions = load_module('functions', 'music/core/functions.py')

def test_db_amp_conversion():
    values = np.array([-12, -6, 0, 6, 12])
    amps = utils.db_to_amp(values)
    back = utils.amp_to_db(amps)
    assert np.allclose(back, values)


def test_hz_midi_conversion():
    freqs = np.array([220.0, 440.0, 880.0])
    midis = utils.hz_to_midi(freqs)
    back = utils.midi_to_hz(midis)
    assert np.allclose(back, freqs)


def test_horizontal_stack_and_convert_to_stereo():
    m1 = np.arange(4)
    m2 = np.arange(4) + 4
    stereo = np.vstack((np.arange(4), np.arange(4) + 10))
    stacked = utils.horizontal_stack(m1, stereo, m2)
    assert stacked.shape == (2, 12)
    conv = utils.convert_to_stereo(m1)
    assert conv.shape == (2, 4)
    assert np.allclose(conv[0], m1)
    multi = np.vstack((m1, m1 + 10, m1 + 20))
    conv_multi = utils.convert_to_stereo(multi)
    expected = np.vstack((multi[0] + multi[2], multi[1] + multi[2]))
    assert np.allclose(conv_multi, expected)


def test_mix_and_normalize():
    a = np.ones(5)
    b = np.arange(3)
    mixed = utils.mix(a, b)
    expected = a.copy()
    expected[:3] += b
    assert np.allclose(mixed, expected)
    norm = functions.normalize_mono(mixed)
    assert np.max(norm) <= 1 and np.min(norm) >= -1


def test_mix2_basic():
    a = np.array([1, 1, 1])
    b = np.array([1, 2])
    mixed = utils.mix2([a, b])
    assert np.allclose(mixed, np.array([2, 3, 1]))


def test_mix2_offset_and_end():
    a = np.array([1, 1])
    b = np.array([1, 1, 1])
    out = utils.mix2([a, b], end=True)
    assert np.allclose(out, np.array([1, 2, 2]))

    out_offset = utils.mix2([a, b], offset=[0, 1], sample_rate=1)
    assert np.allclose(out_offset, np.array([1, 2, 1, 1]))


def test_hz_to_midi_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        utils.hz_to_midi(np.array([0.0, 440.0]))
