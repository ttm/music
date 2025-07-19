import sys
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import music.singing.bootstrap as bootstrap


def test_setup_engine_invokes_git_clone_when_missing():
    with patch.object(bootstrap.os.path, 'exists', return_value=False):
        with patch.object(bootstrap.subprocess, 'run') as mock_run:
            bootstrap.setup_engine(method='http')
            mock_run.assert_called_once_with(
                ['git', 'clone', 'https://github.com/ttm/ecantorix', bootstrap.ECANTORIXDIR],
                check=True
            )


def test_setup_engine_skips_when_dir_exists():
    with patch.object(bootstrap.os.path, 'exists', return_value=True):
        with patch.object(bootstrap.subprocess, 'run') as mock_run:
            bootstrap.setup_engine(method='http')
            mock_run.assert_not_called()
