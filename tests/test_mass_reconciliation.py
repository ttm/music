"""The package, routine by routine, against the MASS reference implementation.

The package's central claim is fidelity to a published framework.  These tests
check it: for each routine in the reference's `src/aux/functions.py`, either
the package reproduces it sample for sample, or it diverges for a reason the
register states.

The reference is GPL-3 and this package is MIT, so no reference source lives
here.  `tools/mass_reconcile.py` runs both implementations against a MASS
checkout and records what the reference produced in
`tests/fixtures/mass_reference.npz` — digests for the routines that must agree
exactly, and the samples themselves where the two differ.  These tests read
that fixture, so they need no checkout of their own.

    python tools/mass_reconcile.py --write-fixture   # after changing a case

References
----------
.. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
       representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)
"""

import numpy as np
import pytest

from music.utils import (WAVEFORM_SAWTOOTH, WAVEFORM_SINE, WAVEFORM_SQUARE,
                         WAVEFORM_TRIANGULAR)
from tools.mass_reconcile import (DIVERGENT, EXACT, REFERENCE_BROKEN, FIXTURE,
                                  build_cases, digest)


@pytest.fixture(scope='module')
def recorded():
    """What the reference produced, as recorded by tools/mass_reconcile.py."""
    if not FIXTURE.is_file():
        pytest.skip(f'{FIXTURE} is missing; run tools/mass_reconcile.py '
                    '--write-fixture against a MASS checkout')
    with np.load(FIXTURE, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope='module')
def cases(recorded):
    """The register's cases, with the package driven by the reference's tables.

    The two tables the package deliberately corrected are read back from the
    fixture, so the synthesis routines are compared on the reference's own
    constants and a difference in a table can never be mistaken for a
    difference in an algorithm.  The two that agree exactly are taken from the
    package, which `test_the_tables_that_agree_are_identical` establishes
    before anything here relies on it.
    """
    return build_cases({
        'Tr': recorded['Tr.samples'],
        'Sa': recorded['Sa.samples'],
        'S': np.asarray(WAVEFORM_SINE),
        'Q': np.asarray(WAVEFORM_SQUARE),
    })


def _ids(expect):
    def select(case_list):
        return [c for c in case_list if c.expect == expect]
    return select


def _case_ids(case_list):
    return [f'{c.mass}-{c.music}' for c in case_list]


# --------------------------------------------------------------------------
# The two constants everything else is measured on
# --------------------------------------------------------------------------

@pytest.mark.parametrize('name, table',
                         [('S', WAVEFORM_SINE), ('Q', WAVEFORM_SQUARE)])
def test_the_tables_that_agree_are_identical(recorded, name, table):
    """Sine and square are the reference's, byte for byte."""
    assert digest(np.asarray(table)) == str(recorded[f'{name}.digest'])


def test_the_triangle_reaches_full_amplitude_where_the_reference_does_not(
        recorded):
    """The reference's triangle duplicates its peak and never reaches 1."""
    reference = recorded['Tr.samples']
    # Each half is a ramp of 8192 points, so the reference's peak falls one
    # step of that ramp short of full amplitude, and lands on two samples.
    assert reference.max() == pytest.approx(1 - 2 / 8192)
    assert reference[8191] == reference[8192]        # the duplicated peak
    assert WAVEFORM_TRIANGULAR.max() == 1.0
    assert WAVEFORM_TRIANGULAR.argmax() == 8192      # the midpoint, once
    assert np.max(np.abs(reference - WAVEFORM_TRIANGULAR)) <= 2 / 8192


def test_the_sawtooth_tiles_where_the_reference_does_not(recorded):
    """The reference includes the endpoint, so its table does not tile."""
    reference = recorded['Sa.samples']
    assert np.diff(reference)[0] == pytest.approx(2 / 16383)
    assert abs(reference[0] - reference[-1]) == pytest.approx(2.0)

    step = 2 / 16384
    assert np.allclose(np.diff(WAVEFORM_SAWTOOTH), step)
    assert abs(WAVEFORM_SAWTOOTH[0] - WAVEFORM_SAWTOOTH[-1]) == \
        pytest.approx(step * (len(WAVEFORM_SAWTOOTH) - 1))


# --------------------------------------------------------------------------
# The register
# --------------------------------------------------------------------------

def test_every_reference_routine_is_accounted_for(cases):
    """Each of the reference's routines is mapped, or deliberately not."""
    mapped = {c.mass for c in cases}
    # The reference's 36 defs, minus the private convolve helper and the
    # duplicate second definition of T_ that shadows the first.
    assert len(mapped) == 35
    for case in cases:
        assert case.expect in (EXACT, DIVERGENT, REFERENCE_BROKEN)
        if case.expect != EXACT:
            assert case.reason, f'{case.mass} diverges without a reason'


def test_the_exact_cases_reproduce_the_reference_sample_for_sample(
        cases, recorded, subtests=None):
    """The claim, for the routines that make it without qualification."""
    checked = []
    for case in cases:
        if case.expect != EXACT:
            continue
        produced = np.asarray(case.package(), dtype=float)
        expected_shape = tuple(recorded[f'{case.mass}.shape'])
        assert produced.shape == expected_shape, (
            f'{case.music} no longer has the shape {case.mass} produced')
        assert digest(produced) == str(recorded[f'{case.mass}.digest']), (
            f'{case.music} no longer reproduces {case.mass} sample for '
            f'sample; run tools/mass_reconcile.py against a MASS checkout '
            f'to see where')
        checked.append(case.mass)
    assert len(checked) == 26


@pytest.mark.parametrize('mass_name', ['trill', 'loc_', 'D_', 'Tr', 'Sa'])
def test_the_divergences_stay_the_size_their_reason_accounts_for(
        cases, recorded, mass_name):
    """A stated divergence is bounded, and it is still there.

    Growing past the bound means the reason no longer explains it.  Shrinking
    to nothing means the correction the reason describes has been undone.
    """
    case = next(c for c in cases if c.mass == mass_name)
    assert case.expect == DIVERGENT
    reference = recorded[f'{case.mass}.samples']
    produced = np.asarray(case.package(), dtype=float)

    assert produced.shape == reference.shape
    delta = float(np.max(np.abs(produced - reference)))
    assert delta > 0, (
        f'{case.music} now matches {case.mass} exactly, which undoes the '
        f'correction the register records: {case.reason}')
    assert delta <= case.bound, (
        f'{case.music} differs from {case.mass} by {delta:.3e}, more than the '
        f'{case.bound:.3g} its reason accounts for: {case.reason}')


@pytest.mark.parametrize('mass_name', ['noises', 'loc2', 'FIR', 'R'])
def test_the_package_runs_where_the_reference_cannot(cases, mass_name):
    """Four reference routines raise for every input.  These do not."""
    case = next(c for c in cases if c.mass == mass_name)
    assert case.expect == REFERENCE_BROKEN
    produced = np.asarray(case.package(), dtype=float)
    assert produced.size > 0
    assert np.isfinite(produced).all()
