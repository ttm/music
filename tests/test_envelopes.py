import numpy as np
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import music


# Test amplitude modulation envelope

def test_am_envelope_range_and_application():
    ns = 1000
    env = music.am(number_of_samples=ns, fm=50, max_amplitude=0.3, sonic_vector=None)
    assert env.min() >= 1 - 0.3 - 1e-6
    assert env.max() <= 1 + 0.3 + 1e-6

    wave = np.ones_like(env)
    modulated = music.am(number_of_samples=ns, fm=50, max_amplitude=0.3, sonic_vector=wave)
    assert np.max(modulated) > 1
    assert np.min(modulated) < 1


# Test tremolo envelope

def test_tremolo_envelope_range_and_application():
    ns = 1000
    db_dev = 6
    env = music.tremolo(number_of_samples=ns, tremolo_freq=100, max_db_dev=db_dev, sonic_vector=None)
    min_val = 10 ** (-db_dev / 20)
    max_val = 10 ** (db_dev / 20)
    assert env.min() >= min_val - 1e-6
    assert env.max() <= max_val + 1e-6

    wave = np.ones_like(env)
    modulated = music.tremolo(number_of_samples=ns, tremolo_freq=100, max_db_dev=db_dev, sonic_vector=wave)
    assert np.max(modulated) > 1
    assert np.min(modulated) < 1

