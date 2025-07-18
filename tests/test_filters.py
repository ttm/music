import numpy as np
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from music.core.filters import adsr, fade, cross_fade, reverb
from music.core.filters.localization import localize


def test_adsr_envelope_basic():
    env = adsr(
        envelope_duration=0.1,
        attack_duration=10,
        decay_duration=20,
        sustain_level=-6,
        release_duration=10,
        transition="exp",
        sample_rate=1000,
    )
    sustain_amp = 10 ** (-6 / 20)
    assert len(env) == 100
    assert env[0] < 1e-3
    assert np.isclose(env[9], 1.0, atol=1e-6)
    assert np.allclose(env[30:90], sustain_amp)
    assert env[-1] < 1e-4


def test_fade_and_cross_fade():
    fade_out = fade(number_of_samples=5, fade_out=True, method="linear")
    fade_in = fade(number_of_samples=5, fade_out=False, method="linear")
    assert np.allclose(fade_out, np.linspace(1, 0, 5))
    assert np.allclose(fade_in, np.linspace(0, 1, 5))

    s1 = np.ones(441)
    s2 = np.ones(441) * 2
    mixed = cross_fade(s1.copy(), s2.copy(), duration=5, sample_rate=44100)
    assert len(mixed) == 661
    assert mixed[0] == 1.0
    assert np.isclose(mixed[-1], 2.0, atol=1e-6)


def test_reverb_minimal_operation():
    np.random.seed(0)
    ir = reverb(
        duration=0.02,
        first_phase_duration=0.01,
        decay=-1,
        noise_type="white",
        sample_rate=100,
    )
    assert len(ir) == 2
    assert ir[0] == 1.0

    out = reverb(
        duration=0.02,
        first_phase_duration=0.01,
        decay=-1,
        noise_type="white",
        sonic_vector=np.ones(5),
        sample_rate=100,
    )
    assert out.shape == (6,)

def test_localize_basic():
    sv = np.ones(5)
    out = localize(sonic_vector=sv, x=0.1, y=0.1, sample_rate=10)
    assert out.shape[0] == 2
    assert out.shape[1] >= 5
    assert not np.allclose(out[0], out[1])

