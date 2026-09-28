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

By default only committed files at the requested revision are used; the
optional working-tree overlay copies the selected area sources and tests into
that snapshot. The scratch tree, per-mutant results and surviving diffs are
retained outside the checkout. See MUTATION_AUDIT.md for each area's scope
and the dataclass adapter's limits.
"""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
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
            'tests/test_transition_methods.py',
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
            'tests/test_transition_methods.py',
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
            'tests/test_transition_methods.py',
        ),
        'dataclass_adapter': None,
    },
    # Filter design and application, loudness transitions, reverberation
    # and time stretching. Selected from full-suite per-test coverage.
    'filters': {
        'sources': (
            'music/core/filters/design.py',
            'music/core/filters/impulse_response.py',
            'music/core/filters/loud.py',
            'music/core/filters/reverb.py',
            'music/core/filters/stretches.py',
        ),
        'tests': (
            'tests/test_filter_design.py',
            'tests/test_filters_response.py',
            'tests/test_filters.py',
            'tests/test_filters_audit.py',
            'tests/test_artifacts.py::test_a_one_pole_design_is_stable_at_any_cutoff[0.49-high_pass]',
            'tests/test_artifacts.py::test_a_two_pole_design_is_stable_anywhere_in_its_grid[0.4-0.25-band_reject]',
            'tests/test_branches.py::test_stretches_rejects_a_non_positive_duration',
            'tests/test_article.py::test_iir_is_the_difference_equation_diferencas_writes',
            'tests/test_article.py::test_the_reverberation_is_the_two_periods_equations_p1rev_and_p2rev',
            'tests/test_article.py::test_the_reverberation_decays_by_exactly_the_curve_it_documents',
            'tests/test_degenerate.py::test_a_parameter_at_zero_does_not_quietly_add_a_bias[band_pass-bandwidth]',
            'tests/test_degenerate.py::test_a_parameter_at_zero_does_not_quietly_add_a_bias[high_pass-cutoff]',
            'tests/test_degenerate.py::test_a_parameter_at_zero_works_or_is_refused_clearly[band_reject-centre]',
            'tests/test_degenerate.py::test_a_parameter_at_zero_works_or_is_refused_clearly[reverb-duration]',
            'tests/test_degenerate.py::test_a_parameter_at_zero_works_or_is_refused_clearly[low_pass-cutoff]',
            'tests/test_fidelity.py::test_stretching_nothing_gives_nothing[shape0]',
            'tests/test_fidelity.py::test_loud_is_the_decibel_curve_it_documents',
            'tests/test_fidelity.py::test_loud_with_no_deviation_is_unity_gain',
            'tests/test_fidelity.py::test_reverb_decays_by_the_decibels_it_was_given',
            'tests/test_fidelity.py::test_stretches_gives_each_repeat_the_duration_it_asked_for',
            'tests/test_fidelity.py::test_every_stretch_lasts_exactly_its_duration',
            'tests/test_fidelity.py::test_stretches_squeezes_rather_than_truncates',
            'tests/test_mass_reconciliation.py::test_the_exact_cases_reproduce_the_reference',
            'tests/test_mass_reconciliation.py::test_the_package_runs_where_the_reference_cannot[FIR]',
            'tests/test_public_api.py::test_louds_pads_the_envelope_to_the_signal',
            'tests/test_public_api.py::test_sonic_vector_accepts_any_array_like[reverb]',
            'tests/test_public_api.py::test_stretches_resamples_to_the_requested_durations',
            'tests/test_tutorial.py::test_every_tutorial_block_runs',
            'tests/test_filters_audit.py::test_louds_pads_a_short_signal_with_silence',
            'tests/test_transition_methods.py',
        ),
        'dataclass_adapter': None,
    },
    # Scales, modes, chords, intervals and the harmonic series of the MASS
    # companion paper. Selected from full-suite per-test coverage.
    'theory': {
        'sources': (
            'music/theory/chords.py',
            'music/theory/intervals.py',
            'music/theory/scales.py',
        ),
        'tests': (
            'tests/test_theory.py',
            'tests/test_theory_properties.py',
            'tests/test_theory_audit.py',
            'tests/test_degenerate.py::test_a_parameter_at_zero_does_not_quietly_add_a_bias[harmonic_series-partials]',
            'tests/test_degenerate.py::test_a_parameter_at_zero_does_not_quietly_add_a_bias[mode_by_rotation-kappa]',
            'tests/test_degenerate.py::test_a_parameter_at_zero_works_or_is_refused_clearly[harmonic_series-partials]',
            'tests/test_degenerate.py::test_a_parameter_at_zero_works_or_is_refused_clearly[mode_by_rotation-kappa]',
            'tests/test_public_api.py::test_export_runs_with_its_documented_defaults[chord]',
            'tests/test_public_api.py::test_export_runs_with_its_documented_defaults[harmonic_series]',
            'tests/test_public_api.py::test_export_runs_with_its_documented_defaults[mode_by_rotation]',
            'tests/test_public_api.py::test_export_runs_with_its_documented_defaults[scale]',
        ),
        'dataclass_adapter': None,
    },
    # Permutation families and the change-ringing peals built from them.
    # Selected from full-suite per-test coverage.
    'structures': {
        'sources': (
            'music/structures/permutations.py',
            'music/structures/peals/base.py',
            'music/structures/peals/peals.py',
            'music/structures/peals/plain_changes.py',
        ),
        'tests': (
            'tests/test_structures.py',
            'tests/test_peals_named.py',
            'tests/test_structures_audit.py',
            'tests/test_additional.py::test_permutation_helpers',
            'tests/test_article.py::test_the_permutation_structures_satisfy_the_axioms_of_equation_groups',
            'tests/test_article.py::test_the_rotations_are_a_group_in_their_own_right',
            'tests/test_branches.py::test_a_generic_peal_acts_all_on_a_domain_it_is_given',
            'tests/test_branches.py::test_a_peal_acts_all_of_them_on_a_domain_it_is_given',
            'tests/test_branches.py::test_a_peal_acts_on_the_domain_and_peal_it_is_given',
            'tests/test_branches.py::test_even_odd_agrees_with_sympy_for_every_permutation',
            'tests/test_branches.py::test_generic_peal_needs_nelements_for_a_default_domain',
            'tests/test_branches.py::test_interesting_permutations_of_a_pair',
            'tests/test_branches.py::test_plain_changes_act_all_records_every_peal',
            'tests/test_branches.py::test_plain_changes_acts_on_a_given_domain',
            'tests/test_branches.py::test_plain_changes_acts_on_its_own_domain_by_default',
            'tests/test_legacy.py::test_set_size_and_set_perms_record_what_they_are_given',
            'tests/test_legacy.py::test_stay_accepts_a_numpy_domain',
            'tests/test_legacy.py::test_stay_falls_back_to_the_grid_when_no_domain_is_set',
            'tests/test_legacy.py::test_stay_permutes_the_domain',
            'tests/test_remaining_paths.py::test_perform_peal_builds_its_own_hunts_when_given_none',
            'tests/test_theory_properties.py::test_a_peal_becomes_a_melody_of_the_right_length',
            'tests/test_theory_properties.py::test_each_change_swaps_one_adjacent_pair',
            'tests/test_theory_properties.py::test_interesting_permutations_are_permutations',
            'tests/test_theory_properties.py::test_plain_changes_ring_every_row_once',
            'tests/test_theory_properties.py::test_transposing_a_permutation_shifts_what_it_moves',
            'tests/test_tutorial.py::test_every_tutorial_block_runs',
        ),
        'dataclass_adapter': None,
    },    # Coloured, Gaussian-band and silent noise. Selected from full-suite
    # per-test coverage.
    'noises': {
        'sources': (
            'music/core/synths/noises.py',
        ),
        'tests': (
            'tests/test_noises_audit.py',
            'tests/test_article.py',
            'tests/test_stimulation.py::test_full_depth_takes_the_noise_envelope_to_silence',
            'tests/test_stimulation.py::test_modulated_noise_puts_the_envelope_at_the_modulation_rate',
            'tests/test_stimulation.py::test_noise_colour_changes_the_spectral_tilt',
            'tests/test_stimulation.py::test_spatial_motion_can_move_a_sound_it_did_not_synthesize',
            'tests/test_stimulation.py::test_unmodulated_noise_is_the_bare_noise_bed',
            'tests/test_stimulation.py::test_zero_depth_leaves_the_noise_bed_alone',
            'tests/test_degenerate.py::test_a_gaussian_noise_refuses_a_band_with_nothing_in_it',
            'tests/test_degenerate.py::test_a_parameter_at_zero_does_not_quietly_add_a_bias',
            'tests/test_degenerate.py::test_a_parameter_at_zero_works_or_is_refused_clearly',
            'tests/test_degenerate.py::test_a_zero_duration_renders_nothing',
            'tests/test_degenerate.py::test_a_zero_duration_renders_nothing_down_every_branch',
            'tests/test_properties.py::test_a_noise_spectrum_describes_a_real_signal',
            'tests/test_properties.py::test_a_render_is_as_long_as_the_duration_and_rate_it_was_given',
            'tests/test_properties.py::test_a_whole_short_piece_renders_at_either_rate',
            'tests/test_properties.py::test_noise_renders_at_any_rate_including_an_odd_number_of_samples',
            'tests/test_properties.py::test_the_energy_in_the_samples_is_the_energy_in_the_spectrum',
            'tests/test_stimuli_audit.py::test_a_bare_call_renders_the_defaults_it_declares',
            'tests/test_stimuli_audit.py::test_a_one_sample_stimulus_is_one_sample',
            'tests/test_stimuli_audit.py::test_the_noise_and_the_orbit_honour_number_of_samples',
            'tests/test_stimuli_audit.py::test_the_noise_envelope_is_the_same_envelope_on_the_same_bed',
            'tests/test_stimuli_audit.py::test_unmodulated_noise_is_the_band_it_was_asked_for_at_its_rate',
            'tests/test_filters_audit.py::test_filter_defaults_keep_their_documented_sample_spans',
            'tests/test_filters_audit.py::test_reverb_applies_the_response_by_convolution',
            'tests/test_filters_audit.py::test_reverb_defaults_to_a_point_one_five_second_first_phase',
            'tests/test_filters_audit.py::test_reverb_with_no_first_phase_leaves_the_noise_random_stream_alone',
            'tests/test_fidelity.py::test_noise_colour_has_its_documented_spectral_slope',
            'tests/test_fidelity.py::test_numeric_noise_type_is_taken_as_decibels_per_octave',
            'tests/test_fidelity.py::test_reverb_decays_by_the_decibels_it_was_given',
            'tests/test_filters.py::test_a_one_sample_reverb_is_the_direct_sound',
            'tests/test_filters.py::test_reverb_minimal_operation',
            'tests/test_filters.py::test_the_reverb_tail_spans_the_band_at_any_rate',
            'tests/test_public_api.py::test_export_runs_with_its_documented_defaults',
            'tests/test_public_api.py::test_noise_accepts_a_numeric_gain_per_octave',
            'tests/test_public_api.py::test_sonic_vector_accepts_any_array_like',
            'tests/test_synths.py::test_gaussian_noise_takes_a_fractional_duration',
            'tests/test_synths.py::test_noise_and_silence_generation',
            'tests/test_synths.py::test_noise_no_warnings',
            'tests/test_artifacts.py::test_what_the_click_measure_can_and_cannot_see',
            'tests/test_branches.py::test_noise_rejects_an_unknown_colour',
            'tests/test_mass_reconciliation.py::test_the_package_runs_where_the_reference_cannot',
            'tests/test_tutorial.py::test_every_tutorial_block_runs',
        ),
        'dataclass_adapter': None,
    },    # The note sequencer: scheduling, per-note rendering and mixing.
    # Selected from full-suite per-test coverage.
    'sequencer': {
        'sources': (
            'music/sequencer.py',
        ),
        'tests': (
            'tests/test_sequencer_audit.py',
            'tests/test_remaining_paths.py::test_sequencer_applies_an_adsr_envelope',
            'tests/test_remaining_paths.py::test_sequencer_mixes_mono_and_stereo_notes_together',
            'tests/test_remaining_paths.py::test_sequencer_mixes_stereo_then_mono',
            'tests/test_remaining_paths.py::test_sequencer_places_a_note_in_space_and_writes_stereo',
            'tests/test_remaining_paths.py::test_sequencer_renders_a_vibrato_note',
            'tests/test_sequencer.py::test_a_note_cannot_start_before_the_sequence',
            'tests/test_sequencer.py::test_sequencer_basic_mono',
            'tests/test_sequencer.py::test_sequencer_stereo_spatial',
            'tests/test_sequencer.py::test_sequencer_write',
            'tests/test_theory_properties.py::test_a_sequencer_puts_its_notes_where_it_was_told',
            'tests/test_tutorial.py::test_every_tutorial_block_runs',
        ),
        'dataclass_adapter': 'music/sequencer.py',
    },
}


def expose_dataclasses(tree, relative_path):
    """Apply dataclass after each class so mutmut visits its methods.

    mutmut 3.7.0 skips decorated classes. This changes only the scratch
    copy: each class decorated with a bare ``@dataclass`` loses the
    decorator and is followed by ``Name = dataclass(Name)``, so it is
    still a dataclass before anything uses it. Property-decorated methods
    remain outside mutmut's scope.

    Only an area whose sources define decorated classes needs this:
    ``export`` for the session, and ``sequencer``.
    """
    path = tree / relative_path
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    classes = [node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef)
               and any(isinstance(decorator, ast.Name)
                       and decorator.id == 'dataclass'
                       for decorator in node.decorator_list)]
    if not classes:
        raise SystemExit(f'dataclass adapter found no dataclass in '
                         f'{relative_path}')
    # From the end, so earlier line numbers stay put.
    for node in reversed(classes):
        lines.insert(node.end_lineno,
                     f'\n\n{node.name} = dataclass({node.name})\n')
        decorator = next(decorator for decorator in node.decorator_list
                         if isinstance(decorator, ast.Name)
                         and decorator.id == 'dataclass')
        if lines[decorator.lineno - 1].strip() != '@dataclass':
            raise SystemExit(f'dataclass adapter no longer matches '
                             f'{node.name}')
        del lines[decorator.lineno - 1]
    path.write_text(''.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--area', default='export', choices=sorted(AREAS),
                        help='which bounded area to audit (default: export)')
    parser.add_argument('--list-areas', action='store_true',
                        help='print each area with its sources, and stop')
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--max-children', type=int, default=2)
    parser.add_argument(
        '--overlay-working-tree', action='store_true',
        help=('copy this area’s source and selected test files over the '
              'archive'))
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
    overlay = []
    if args.overlay_working_tree:
        paths = set(sources)
        paths.update(test.split('::', 1)[0] for test in tests)
        for relative in sorted(paths):
            source = ROOT / relative
            destination = tree / relative
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            elif source.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            else:
                raise SystemExit(
                    f'working-tree overlay path does not exist: {relative}')
            overlay.append(relative)
        print(f'Overlaid {len(overlay)} working-tree paths', flush=True)
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
        'working_tree_overlay': overlay,
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
