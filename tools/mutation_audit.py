#!/usr/bin/env python3
"""Run the bounded normalization/export/session mutation audit in a copy.

Install mutmut==3.7.0 into the development environment, then run:
    python tools/mutation_audit.py --revision HEAD

Only committed files at the requested revision are used. The scratch tree,
per-mutant results and surviving diffs are retained outside the checkout.
See MUTATION_AUDIT.md for the scope and the dataclass adapter's limits.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
SOURCES = (
    'music/core/functions.py',
    'music/core/io.py',
    'music/stimulation/session.py',
)
TESTS = (
    'tests/test_normalization.py',
    'tests/test_io_paths.py',
    'tests/test_io.py',
    'tests/test_audio_formats.py',
    'tests/test_fidelity.py',
    'tests/test_stimulation_session.py',
    'tests/test_mass_reconciliation.py',
    'tests/test_artifacts.py::test_the_quantizer_clips_rather_than_wraps',
    'tests/test_degenerate.py::test_writing_nothing_says_there_is_nothing',
)


def expose_dataclasses(tree):
    """Apply dataclass after each class so mutmut visits its methods.

    mutmut 3.7.0 skips decorated classes. This changes only the scratch
    copy; both classes still become dataclasses before anything uses them.
    Property-decorated methods remain outside mutmut's scope.
    """
    path = tree / SOURCES[2]
    source = path.read_text()
    for name in ('StimulusPhase', 'StimulationSession'):
        marker = f'@dataclass\nclass {name}:'
        if source.count(marker) != 1:
            raise SystemExit(f'dataclass adapter no longer matches {name}')
        source = source.replace(marker, f'class {name}:', 1)
    marker = '\n\ndef _ramp_shape('
    if source.count(marker) != 1:
        raise SystemExit('dataclass adapter no longer matches _ramp_shape')
    source = source.replace(
        marker, '\n\nStimulusPhase = dataclass(StimulusPhase)' + marker, 1)
    source += '\n\nStimulationSession = dataclass(StimulationSession)\n'
    path.write_text(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--max-children', type=int, default=2)
    args = parser.parse_args()
    version = importlib.metadata.version('mutmut')
    if version != '3.7.0':
        raise SystemExit(f'expected mutmut 3.7.0, found {version}')
    revision = subprocess.check_output(
        ['git', 'rev-parse', '--verify', args.revision + '^{commit}'],
        cwd=ROOT, text=True).strip()
    archived = subprocess.check_output(
        ['git', 'archive', revision], cwd=ROOT)
    tree = Path(tempfile.mkdtemp(prefix='music-mutation-'))
    print(f'Auditing {revision}\nScratch tree: {tree}', flush=True)
    with tarfile.open(fileobj=io.BytesIO(archived)) as archive:
        archive.extractall(tree, filter='data')
    expose_dataclasses(tree)
    config = {
        'source_paths': ['music/'],
        'only_mutate': list(SOURCES),
        'also_copy': ['conftest.py', 'pytest.ini', 'tests/', 'tools/',
                      'RECONCILIATION.md'],
        'pytest_add_cli_args': ['-o', 'addopts=', '-p', 'no:cacheprovider'],
        'pytest_add_cli_args_test_selection': list(TESTS),
        'max_stack_depth': -1,
        'use_setproctitle': False,
    }
    with (tree / 'pyproject.toml').open('a') as stream:
        stream.write('\n[tool.mutmut]\n')
        for key, value in config.items():
            stream.write(f'{key} = {json.dumps(value)}\n')
    start = time.monotonic()
    subprocess.run(
        [sys.executable, '-m', 'mutmut', 'run', '--max-children',
         str(args.max_children)], cwd=tree, check=True)
    elapsed = time.monotonic() - start
    subprocess.run([sys.executable, '-m', 'mutmut', 'export-cicd-stats'],
                   cwd=tree, check=True)
    stats = json.loads(
        (tree / 'mutants/mutmut-cicd-stats.json').read_text())
    results = {}
    for source in SOURCES:
        metadata = tree / 'mutants' / (source + '.meta')
        results.update(json.loads(metadata.read_text())['exit_code_by_key'])
    report = {
        'revision': revision,
        'python': sys.version,
        'mutmut': version,
        'elapsed_seconds': round(elapsed, 2),
        'configuration': config,
        'dataclass_adapter': True,
        'stats': stats,
        'exit_code_by_mutant': results,
    }
    (tree / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    # These APIs are private, hence the version pin above.
    from mutmut.configuration import Config
    from mutmut.__main__ import get_diff_for_mutant

    os.chdir(tree)
    Config.ensure_loaded()
    with (tree / 'survivors.patch').open('w') as stream:
        for name, code in sorted(results.items()):
            if code == 0:
                stream.write(f'# {name}\n')
                stream.write(get_diff_for_mutant(name) + '\n')
    print(json.dumps(stats, indent=2))
    print(f'{elapsed:.1f} seconds; report and survivor diffs in {tree}')
    incomplete = sum(value for key, value in stats.items()
                     if key not in ('killed', 'survived', 'total'))
    return 1 if incomplete else 0


if __name__ == '__main__':
    raise SystemExit(main())
