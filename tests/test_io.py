import numpy as np
import sys
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import music


def test_invalid_bit_depth_mono():
    with pytest.raises(ValueError):
        music.write_wav_mono(np.zeros(10), filename="tmp.wav", bit_depth=24)


def test_invalid_bit_depth_stereo():
    with pytest.raises(ValueError):
        music.write_wav_stereo(np.zeros((2, 10)), filename="tmp.wav", bit_depth=24)
