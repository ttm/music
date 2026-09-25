#!/usr/bin/env python3
"""Run one bounded mutation audit of this package in a scratch copy.

Install mutmut==3.7.0 into the development environment, then run:
    python tools/mutation_audit.py --area envelopes --revision HEAD

Each area names the files to mutate and the tests to judge them with; see
``AREAS`` below and ``--list-areas``. An area is deliberately small, because
the cost of a run is the test selection multiplied by the mutants, and
because the output worth reading is the surviving mutants rather than a
score. The default area is the one audited first, so the command recorded
in MUTATION_AUDIT.md keeps reproducing that run.

Only committed files at the requested revision are used. The scratch tree,
per-mutant results and surviving diffs are retained outside the checkout.
See MUTATION_AUDIT.md for each area's scope and the dataclass adapter's
limits.
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

#: Files the audit copies into the scratch tree besides the package and its
#: tests, because some selected test reads them: ``RECONCILIATION.md`` holds
#: the reference comparison, ``docs/`` the tutorial and API listing, and
#: ``README.md`` the figures checked against the code.
SUPPORT = ['conftest.py', 'pytest.ini', 'tests/', 'tools/',
           'RECONCILIATION.md', 'docs/', 'README.md']

#: The bounded areas this runner knows how to audit. ``sources`` are mutated;
#: ``tests`` judge the mutants and must cover every line and branch of the
#: sources between them, or mutmut reports mutants no test reaches. Add an
#: area rather than widening one: see MUTATION_AUDIT.md.
AREAS = {
    # Audited 2026-09-16 with the 1.8.1 correctness patch.
    'export': {
        'sources': (
            'music/core/functions.py',
            'music/core/io.py',
            'music/stimulation/session.py',
        ),
        'tests': (
            'tests/test_normalization.py',
            'tests/test_io_paths.py',
            'tests/test_io.py',
            'tests/test_audio_formats.py',
            'tests/test_fidelity.py',
            'tests/test_stimulation_session.py',
            'tests/test_mass_reconciliation.py',
            'tests/test_artifacts.py::test_the_quantizer_clips_rather_than_wraps',
            'tests/test_degenerate.py::test_writing_nothing_says_there_is_nothing',
        ),
        'dataclass_adapter': 'music/stimulation/session.py',
    },
    # Oscillator timing: the notes, their vibratos and their glissandi.
    'oscillators': {
        'sources': (
            'music/core/synths/notes.py',
        ),
        'tests': (
            'tests/test_degenerate.py',
            'tests/test_properties.py',
            'tests/test_fidelity.py',
            'tests/test_artifacts.py',
            'tests/test_branches.py',
            'tests/test_public_api.py',
            'tests/test_article.py',
            'tests/test_stimulation.py',
            'tests/test_remaining_paths.py',
            'tests/test_audio_formats.py',
            'tests/test_bonds.py',
            'tests/test_hrtf.py',
            'tests/test_notes_extra.py',
            'tests/test_synths.py',
            'tests/test_seq_localization_pitch.py',
            'tests/test_seq_localization_spatial.py',
            'tests/test_seq_localization_edges.py',
            'tests/test_vibratos_glissandos_audit.py',
            'tests/test_doppler_audit.py',
            'tests/test_glissando_trill_audit.py',
            'tests/test_utils.py',
            'tests/test_mass_reconciliation.py',
            'tests/test_theory_properties.py',
            'tests/test_localize_linear.py',
            'tests/test_sequencer.py',
            'tests/test_spectral.py',
            'tests/test_filter_design.py',
            'tests/test_io_paths.py',
            'tests/test_legacy.py',
            'tests/test_stimulation_session.py',
            'tests/test_theory.py',
            'tests/test_filters.py',
            'tests/test_hrtf_dataset.py',
            'tests/test_normalization.py',
            'tests/test_tutorial.py',
        ),
        'dataclass_adapter': None,
    },
    # The note-level amplitude envelopes: ADSR, fades and tremolo/AM.
    'envelopes': {
        'sources': (
            'music/core/synths/envelopes.py',
            'music/core/filters/adsr.py',
            'music/core/filters/fade.py',
        ),
        'tests': (
            'tests/test_degenerate.py',
            'tests/test_public_api.py',
            'tests/test_io_paths.py',
            'tests/test_properties.py',
            'tests/test_fidelity.py',
            'tests/test_branches.py',
            'tests/test_artifacts.py',
            'tests/test_remaining_paths.py',
            'tests/test_article.py',
            'tests/test_mass_reconciliation.py',
            'tests/test_legacy.py',
            'tests/test_filters.py',
            'tests/test_bonds.py',
            'tests/test_envelopes.py',
            'tests/test_notes_extra.py',
            'tests/test_mixing.py',
            'tests/test_audio_formats.py',
            'tests/test_theory_properties.py',
            'tests/test_tutorial.py',
        ),
        'dataclass_adapter': None,
    },
    # The sensory-stimulation generators: beats, pulses, modulations and
    # motion. Selected from `pytest --cov-context=test`.
    'stimuli': {
        'sources': (
            'music/stimulation/stimuli.py',
        ),
        'tests': (
            'tests/test_stimulation.py',
            'tests/test_stimuli_audit.py',
            'tests/test_degenerate.py',
            'tests/test_properties.py',
            'tests/test_public_api.py',
            'tests/test_artifacts.py',
            'tests/test_stimulation_session.py',
        ),
        'dataclass_adapter': None,
    },
    # Interaural time and intensity cues: fixed, per-frequency, moving
    # and convolved. Selected from `pytest --cov-context=test`.
    'localization': {
        'sources': (
            'music/core/filters/localization.py',
        ),
        'tests': (
            'tests/test_localize_linear.py',
            'tests/test_localization_audit.py',
            'tests/test_hrtf.py',
            'tests/test_hrtf_dataset.py',
            'tests/test_fidelity.py',
            'tests/test_degenerate.py',
            'tests/test_branches.py',
            'tests/test_remaining_paths.py',
            'tests/test_article.py',
            'tests/test_mass_reconciliation.py',
            'tests/test_audio_formats.py',
            'tests/test_filters.py',
            'tests/test_properties.py',
            'tests/test_public_api.py',
            'tests/test_sequencer.py',
            'tests/test_stimulation.py',
            'tests/test_stimuli_audit.py',
            'tests/test_tutorial.py',
        ),
        'dataclass_adapter': None,
    },
    # Shared utilities: conversions, mixing, waveform tables, profiles and
    # rhythmic durations. Selected from pytest's per-test coverage contexts.
    'utils': {
        'sources': (
            'music/utils.py',
        ),
        'tests': (
            'tests/test_branches.py',
            'tests/test_utils.py',
            'tests/test_mixing.py',
            'tests/test_remaining_paths.py',
            'tests/test_artifacts.py',
            'tests/test_degenerate.py',
            'tests/test_fidelity.py',
            'tests/test_envelopes.py',
            'tests/test_additional.py',
        ),
        'dataclass_adapter': None,
    },
}


def expose_dataclasses(tree, relative_path):
    """Apply dataclass after each class so mutmut visits its methods.

    mutmut 3.7.0 skips decorated classes. This changes only the scratch
    copy; both classes still become dataclasses before anything uses them.
    Property-decorated methods remain outside mutmut's scope.

    Only an area whose sources define decorated classes needs this, which
    today is ``export`` alone; the envelope modules are plain functions.
    """
    path = tree / relative_path
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
    parser.add_argument('--area', default='export', choices=sorted(AREAS),
                        help='which bounded area to audit (default: export)')
    parser.add_argument('--list-areas', action='store_true',
                        help='print each area with its sources, and stop')
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--max-children', type=int, default=2)
    args = parser.parse_args()
    if args.list_areas:
        for name, area in sorted(AREAS.items()):
            print(name)
            for source in area['sources']:
                print(f'    {source}')
        return 0
    area = AREAS[args.area]
    sources, tests = area['sources'], area['tests']
    version = importlib.metadata.version('mutmut')
    if version != '3.7.0':
        raise SystemExit(f'expected mutmut 3.7.0, found {version}')
    revision = subprocess.check_output(
        ['git', 'rev-parse', '--verify', args.revision + '^{commit}'],
        cwd=ROOT, text=True).strip()
    archived = subprocess.check_output(
        ['git', 'archive', revision], cwd=ROOT)
    tree = Path(tempfile.mkdtemp(prefix='music-mutation-'))
    print(f'Auditing {args.area} at {revision}\nScratch tree: {tree}',
          flush=True)
    with tarfile.open(fileobj=io.BytesIO(archived)) as archive:
        archive.extractall(tree, filter='data')
    adapted = area['dataclass_adapter']
    if adapted:
        expose_dataclasses(tree, adapted)
    config = {
        'source_paths': ['music/'],
        'only_mutate': list(sources),
        'also_copy': list(SUPPORT),
        'pytest_add_cli_args': ['-o', 'addopts=', '-p', 'no:cacheprovider'],
        'pytest_add_cli_args_test_selection': list(tests),
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
    for source in sources:
        metadata = tree / 'mutants' / (source + '.meta')
        results.update(json.loads(metadata.read_text())['exit_code_by_key'])
    report = {
        'area': args.area,
        'revision': revision,
        'python': sys.version,
        'mutmut': version,
        'elapsed_seconds': round(elapsed, 2),
        'configuration': config,
        'dataclass_adapter': adapted,
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
    # A mutant that hangs has been detected: no suite that finishes
    # accepts it. `trill` accumulates samples in a while loop, so four of
    # its mutants run forever rather than returning a wrong answer, and
    # counting those as an incomplete run would fail every audit of that
    # file. The categories below are the ones that really leave a mutant
    # without a verdict.
    undecided = sum(value for key, value in stats.items()
                    if key not in ('killed', 'survived', 'total', 'timeout'))
    if stats.get('timeout'):
        print(f"{stats['timeout']} mutant(s) timed out; a mutant that hangs "
              f"is detected, not surviving")
    return 1 if undecided else 0


if __name__ == '__main__':
    raise SystemExit(main())
