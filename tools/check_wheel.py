#!/usr/bin/env python3
"""Install a supplied wheel outside the checkout and smoke its public API.

    python tools/check_wheel.py --wheel dist/music-VERSION-py3-none-any.whl

Dependencies come from the current Python environment. Installing the wheel
uses no index and does not install dependencies or change that environment.
The temporary installation, working directory and rendered files are removed
after the check, including on failure.
"""
from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent.parent
SMOKE = Path(__file__).with_name('wheel_smoke.py')


class WheelCheckError(RuntimeError):
    """An invalid artifact or a failed isolated check."""


def wheel_version(wheel: Path) -> str:
    """Read the package identity from the artifact, not the checkout."""
    if not wheel.is_file() or wheel.suffix != '.whl':
        raise WheelCheckError(f'not a wheel file: {wheel}')
    try:
        with zipfile.ZipFile(wheel) as archive:
            names = [name for name in archive.namelist()
                     if name.endswith('.dist-info/METADATA')
                     and name.count('/') == 1]
            if len(names) != 1:
                raise WheelCheckError(
                    'expected one dist-info/METADATA file in the wheel')
            metadata = BytesParser(policy=default).parsebytes(
                archive.read(names[0]))
    except (OSError, zipfile.BadZipFile) as error:
        raise WheelCheckError(f'cannot read wheel: {error}') from error
    name = re.sub(r'[-_.]+', '-', metadata.get('Name', '')).lower()
    version = metadata.get('Version', '').strip()
    if name != 'music' or not version:
        raise WheelCheckError(
            f'expected music with a version in wheel metadata; '
            f'got name={name!r}, version={version!r}')
    return version


def _run(command: list[str], cwd: Path, environment: dict[str, str],
         stage: str) -> str:
    try:
        result = subprocess.run(command, cwd=cwd, env=environment,
                                capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as error:
        raise WheelCheckError(
            f'{stage} timed out after 120 seconds') from error
    if result.returncode:
        output = '\n'.join(part.strip() for part in
                           (result.stdout, result.stderr) if part.strip())
        raise WheelCheckError(f'{stage} failed:\n{output}')
    return result.stdout


def check_wheel(wheel: Path) -> str:
    """Install and check one wheel; return the smoke process's report."""
    wheel = wheel.resolve()
    version = wheel_version(wheel)
    environment = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP',
                 'PYTHONUSERBASE'):
        environment.pop(name, None)
    with tempfile.TemporaryDirectory(prefix='music-wheel-') as temporary:
        scratch = Path(temporary).resolve()
        if scratch.is_relative_to(ROOT):
            raise WheelCheckError(
                'the temporary directory must be outside the checkout; '
                'set TMPDIR to an external temporary directory')
        target = scratch / 'installed'
        work = scratch / 'work'
        work.mkdir()
        _run([sys.executable, '-I', '-m', 'pip', '--isolated', 'install',
              '--disable-pip-version-check', '--no-index', '--no-deps',
              '--no-cache-dir', '--no-compile', '--target', str(target),
              str(wheel)], work, environment, 'wheel installation')
        return _run([sys.executable, '-I', str(SMOKE), '--target',
                     str(target), '--version', version], work, environment,
                    'installed-wheel smoke check')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True,
                        help='the built wheel to install and check')
    args = parser.parse_args(argv)
    try:
        report = check_wheel(args.wheel)
    except (WheelCheckError, OSError) as error:
        print(f'wheel check failed: {error}', file=sys.stderr)
        return 1
    print(report.strip())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
