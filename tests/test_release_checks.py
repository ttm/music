"""Publication must stop when a newly automated release check fails."""

from pathlib import Path
import subprocess

import pytest

from tools import release


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(release, 'ROOT', tmp_path)
    (tmp_path / 'pyproject.toml').write_text("version = '1.8.1'\n")
    (tmp_path / 'CITATION.cff').write_text('version: 1.8.1\n')
    (tmp_path / 'CHANGELOG.md').write_text('## [1.8.1] - 2026-09-16\n')
    monkeypatch.setattr(release, 'check_repository_state', lambda _: None)
    monkeypatch.setattr(release, 'check_not_on_pypi', lambda _: None)
    return tmp_path


def forbid(*args, **kwargs):
    pytest.fail('this operation must not be reached')


@pytest.mark.parametrize('failed_check', [
    'run_examples.py', 'article_coverage.py', 'mass_reconcile.py',
    'verify_subjects.py',
])
def test_a_failed_check_stops_before_build_or_publication(
        failed_check, checkout, monkeypatch):
    reference = checkout / 'functions.py'
    reference.touch()

    def run(*command):
        if any(Path(part).name == failed_check for part in command):
            raise release.ReleaseError(f'{failed_check}: verification failed')
        return ''

    monkeypatch.setattr(release, 'run', run)
    monkeypatch.setattr(release, 'build', forbid)
    monkeypatch.setattr(release, 'publish', forbid)
    with pytest.raises(release.ReleaseError, match=failed_check):
        release.main(['publish', '--mass', str(reference)])


def test_an_invalid_selected_reference_prevents_publication(
        checkout, monkeypatch):
    monkeypatch.setattr(release, 'run', forbid)
    monkeypatch.setattr(release, 'build', forbid)
    monkeypatch.setattr(release, 'publish', forbid)
    with pytest.raises(release.ReleaseError, match='MASS'):
        release.main(['publish', '--mass', str(checkout / 'absent')])


def test_external_checks_share_the_reference_and_show_exceptions(
        checkout, monkeypatch, capsys):
    reference = checkout / 'functions.py'
    reference.touch()
    commands = []

    def run(*command):
        commands.append(command)
        if any(part.endswith('verify_subjects.py') for part in command):
            return '17 confirmed; 2 documented EuroSciVoc exceptions'
        return 'check summary'

    monkeypatch.setattr(release, 'run', run)
    release.run_gate(str(reference))
    external = {Path(command[1]).name: command[2:] for command in commands
                if command[1].endswith(('.py',))}
    for script in ('article_coverage.py', 'mass_reconcile.py'):
        options = external[script]
        assert options[options.index('--mass') + 1] == str(reference.resolve())
    assert '--strict' in external['article_coverage.py']
    assert '--strict' in external['verify_subjects.py']
    assert '2 documented EuroSciVoc exceptions' in capsys.readouterr().out


def test_verify_runs_checks_without_requiring_an_unpublished_version(
        checkout, monkeypatch, capsys):
    monkeypatch.setattr(release, 'check_repository_state', forbid)
    monkeypatch.setattr(release, 'check_not_on_pypi', forbid)
    monkeypatch.setattr(release, 'publish', forbid)
    seen = []
    monkeypatch.setattr(release, 'run_gate', lambda mass: seen.append(mass))
    monkeypatch.setattr(release, 'build', lambda: seen.append('built'))

    assert release.main(['verify', '--mass', 'selected-checkout']) == 0
    assert seen == ['selected-checkout', 'built']
    assert 'nothing was published' in capsys.readouterr().out


@pytest.mark.parametrize('skip_gate', [False, True])
def test_a_failed_installed_wheel_check_prevents_publication(
        checkout, monkeypatch, skip_gate, capsys):
    monkeypatch.setattr(release, 'run_gate', lambda _: None)
    monkeypatch.setattr(release, 'publish', forbid)
    wheel_checked = []

    def run(*command):
        if command[1:3] == ('-m', 'build'):
            dist = checkout / 'dist'
            dist.mkdir()
            (dist / 'music-1.8.1-py3-none-any.whl').touch()
        if any(part.endswith('check_wheel.py') for part in command):
            wheel_checked.append(command[-1])
            raise release.ReleaseError('installed wheel is broken')
        return ''

    monkeypatch.setattr(release, 'run', run)
    args = ['publish'] + (['--skip-gate'] if skip_gate else [])
    with pytest.raises(release.ReleaseError, match='wheel is broken'):
        release.main(args)
    assert wheel_checked == [
        str(checkout / 'dist' / 'music-1.8.1-py3-none-any.whl')]
    if skip_gate:
        assert 'SKIPPED' in capsys.readouterr().out


def test_failed_commands_preserve_the_report_alongside_warnings(monkeypatch):
    def failed(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0], 1, stdout='wrong scientific result',
            stderr='unrelated warning')

    monkeypatch.setattr(release.subprocess, 'run', failed)
    with pytest.raises(release.ReleaseError) as error:
        release.run('check')
    assert 'wrong scientific result' in str(error.value)
    assert 'unrelated warning' in str(error.value)


def test_an_entry_without_a_headline_stops_before_the_gate(checkout,
                                                            monkeypatch):
    """The Zenodo summary is written after publication; its input is not."""
    (checkout / 'CHANGELOG.md').write_text(
        '## [1.8.1] - 2026-09-16\n\n### Fixed\n\n'
        '- **Headed.** Explained.\n- Not headed.\n')
    monkeypatch.setattr(release, 'run_gate', forbid)
    monkeypatch.setattr(release, 'build', forbid)
    with pytest.raises(release.ReleaseError, match="'Not headed.'"):
        release.main(['verify'])
