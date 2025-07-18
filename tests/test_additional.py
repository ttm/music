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

utils = load_module('utils', 'music/utils.py')
permutations = load_module('permutations', 'music/structures/permutations.py')


def test_midi_interval_and_pitch_to_freq():
    assert np.isclose(utils.midi_to_hz_interval(12), 2.0)
    assert np.isclose(utils.midi_to_hz_interval(-12), 0.5)

    freqs = utils.pitch_to_freq(start_freq=220, semitones=[0, 12, 7])
    expected = [220 * 2 ** (i / 12) for i in [0, 12, 7]]
    assert np.allclose(freqs, expected)


def test_rhythm_to_durations_equivalence():
    durations = [4, 2, 2]
    result_time = utils.rhythm_to_durations(durations=durations, duration=0.25)
    freqs = [4, 8, 8]
    result_freq = utils.rhythm_to_durations(freqs=freqs, duration=4)
    assert np.allclose(result_time, result_freq)

    nested = utils.rhythm_to_durations(durations=[4, [2, 1, 1], 2], duration=0.5)
    assert np.allclose(nested, [2.0, 0.5, 0.5, 1.0])


def test_mix_with_offset_positive_and_negative():
    s1 = np.array([1, 1, 1, 1])
    s2 = np.array([1, 2])
    mixed_pos = utils.mix_with_offset(s1, s2, number_of_samples=2)
    assert np.allclose(mixed_pos, [1, 1, 2, 3])

    mixed_neg = utils.mix_with_offset(s1, s2, number_of_samples=-1)
    assert np.allclose(mixed_neg, [1, 1, 1, 2, 2])


def test_permutation_helpers():
    from sympy.combinatorics import Permutation

    swap = Permutation(0, 3, size=4)
    assert permutations.dist(swap) == 1

    perm = Permutation([2, 0, 1])
    transposed = permutations.transpose_permutation(perm, 1)
    assert transposed == Permutation(1, 2, 3)
    assert permutations.transpose_permutation(perm, 0) == perm
