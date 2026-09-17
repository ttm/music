"""A wheel check must not quietly test the editable checkout instead."""
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from tools.check_wheel import (ROOT, SMOKE, WheelCheckError, check_wheel,
                               main, wheel_version)


def _wheel(tmp_path, package, name='music', version='9.9.9'):
    """A local wheel fixture that pip can install without a build backend."""
    directory = f'{name}-{version}.dist-info'
    contents = {
        f'{directory}/METADATA': (
            f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n'),
        f'{directory}/WHEEL': (
            'Wheel-Version: 1.0\nGenerator: music-wheel-test\n'
            'Root-Is-Purelib: true\nTag: py3-none-any\n'),
        **package,
    }
    record = f'{directory}/RECORD'
    contents[record] = ''.join(f'{path},,\n' for path in [*contents, record])
    path = tmp_path / f'{name}-{version}-py3-none-any.whl'
    with zipfile.ZipFile(path, 'w') as archive:
        for filename, data in contents.items():
            archive.writestr(filename, data)
    return path


def test_wheel_smoke_uses_the_installed_artifact_and_its_own_version(
        tmp_path, monkeypatch):
    """A fake release version distinguishes the artifact from the checkout."""
    package = {path.relative_to(ROOT).as_posix(): path.read_bytes()
               for path in (ROOT / 'music').rglob('*.py')}
    package['music/py.typed'] = b''
    wheel = _wheel(tmp_path, package)
    monkeypatch.setenv('PYTHONPATH', str(ROOT))

    report = check_wheel(wheel)

    assert 'installed music 9.9.9 from ' in report
    assert 'wheel smoke passed' in report
    installed = Path(report.splitlines()[0].split(' from ', 1)[1])
    assert not installed.is_relative_to(ROOT)
    assert not installed.exists(), 'the temporary install must be cleaned up'


def test_a_wheel_without_music_cannot_borrow_the_editable_checkout(
        tmp_path, monkeypatch):
    wheel = _wheel(tmp_path, {})
    monkeypatch.setenv('PYTHONPATH', str(ROOT))

    with pytest.raises(WheelCheckError, match='wheel has no music package'):
        check_wheel(wheel)


def test_a_wheel_cannot_report_an_unrelated_imported_version(tmp_path):
    wheel = _wheel(tmp_path, {
        'music/__init__.py': '__version__ = "0.0.0"\n',
        'music/py.typed': '',
    })
    with pytest.raises(WheelCheckError, match='imported version 0.0.0'):
        check_wheel(wheel)


def test_a_wheel_must_carry_the_type_marker(tmp_path):
    wheel = _wheel(tmp_path, {'music/__init__.py': ''})
    with pytest.raises(WheelCheckError, match='missing music/py.typed'):
        check_wheel(wheel)


def test_a_partial_wheel_cannot_borrow_a_submodule_from_elsewhere(tmp_path):
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    (foreign / 'fallback.py').write_text('value = 42\n')
    wheel = _wheel(tmp_path, {
        'music/__init__.py': (
            f'__path__.append({str(foreign)!r})\n'
            'from . import fallback\n__version__ = "9.9.9"\n'),
        'music/py.typed': '',
    })
    with pytest.raises(WheelCheckError,
                       match='music.fallback came from outside the wheel'):
        check_wheel(wheel)


def test_distribution_metadata_cannot_come_from_another_install(tmp_path):
    """Matching metadata elsewhere cannot validate an incomplete wheel."""
    target = tmp_path / 'installed'
    package = target / 'music'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('__version__ = "9.9.9"\n')
    (package / 'py.typed').touch()
    foreign = tmp_path / 'foreign'
    info = foreign / 'music-9.9.9.dist-info'
    info.mkdir(parents=True)
    (info / 'METADATA').write_text(
        'Metadata-Version: 2.1\nName: music\nVersion: 9.9.9\n')
    (info / 'RECORD').write_text('music-9.9.9.dist-info/METADATA,,\n')
    bootstrap = (
        'import runpy,sys; sys.path.insert(0,sys.argv[1]); '
        'sys.argv=sys.argv[2:]; '
        'runpy.run_path(sys.argv[0],run_name="__main__")')

    result = subprocess.run(
        [sys.executable, '-I', '-c', bootstrap, str(foreign), str(SMOKE),
         '--target', str(target), '--version', '9.9.9'],
        cwd=tmp_path, capture_output=True, text=True)

    assert result.returncode != 0
    assert 'metadata came from outside the wheel' in result.stderr


def test_a_wheel_for_another_project_is_refused(tmp_path):
    wheel = _wheel(tmp_path, {}, name='unrelated')
    with pytest.raises(WheelCheckError, match='expected music'):
        wheel_version(wheel)


def test_missing_wheel_fails_without_using_the_installed_package(
        tmp_path, capsys):
    assert main(['--wheel', str(tmp_path / 'missing.whl')]) == 1
    assert 'not a wheel file' in capsys.readouterr().err
