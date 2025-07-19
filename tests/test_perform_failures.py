import sys
from pathlib import Path
from unittest.mock import patch, mock_open
import subprocess
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import music.singing.perform as perform


def test_sing_raises_when_copy_fails():
    with patch.object(perform, 'write_abc'), \
         patch.object(perform.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['cp'])), \
         patch.object(perform.wavfile, 'read'), \
         patch('builtins.open', mock_open()):
        with pytest.raises(RuntimeError):
            perform.sing()


def test_sing_raises_when_make_fails():
    def side_effect(args, check):
        if args[0] == 'cp':
            return None
        raise subprocess.CalledProcessError(1, args)

    with patch.object(perform, 'write_abc'), \
         patch.object(perform.subprocess, 'run', side_effect=side_effect), \
         patch.object(perform.wavfile, 'read'), \
         patch('builtins.open', mock_open()):
        with pytest.raises(RuntimeError):
            perform.sing()
