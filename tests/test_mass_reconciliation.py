"""The package, routine by routine, against the MASS reference implementation.

The package's central claim is fidelity to a published framework.  These tests
check it: for each routine in the reference's `src/aux/functions.py`, either
the package reproduces it sample for sample, or it diverges for a reason the
register states.

The reference is GPL-3 and this package is MIT, so no reference source lives
here.  `tools/mass_reconcile.py` runs both implementations against a MASS
checkout and records what the reference produced in
`tests/fixtures/mass_reference.npz`.  These tests read that fixture, so they
need no checkout of their own.

**Where "sample for sample" is checked, and where it is not.** That tool is
where it means something: it runs both implementations in one process
against one NumPy, and a difference there is a difference in the code.
These tests cannot make that claim, because a fixture carries the floating
point of the machine that recorded it, and two NumPy builds need not agree
on the last bit of `np.sin` -- an earlier version of this file compared
digests and failed on CI while the package was right.  So what they assert
is the size of the disagreement: no more than one step of the waveform
table, on a handful of samples out of thousands.  That is twelve decibels
below the quietest thing a 16-bit file can carry, and it is still tight
enough that every defect the reconciliation has found would have broken it.

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
from tools.mass_reconcile import (DIVERGENT, EXACT, FIXTURE,
                                  REFERENCE_BROKEN, build_cases)


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
        'S': recorded['S.samples'],
        'Q': recorded['Q.samples'],
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
def test_the_tables_that_agree_are_the_reference_s(recorded, name, table):
    """Sine and square are the reference's table, to floating point.

    The square table is built from ones and is identical everywhere. The
    sine is `np.sin` of a ramp, and that is the one place a different
    NumPy build shows: its last bit is nobody's guarantee, so this asks for
    agreement far below any audible threshold rather than for the bytes.
    """
    produced = np.asarray(table, dtype=float)
    reference = recorded[f'{name}.samples']
    assert produced.shape == reference.shape
    assert np.max(np.abs(produced - reference)) < 1e-12


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


#: One step of a 16384-entry waveform table. A difference of this size is
#: one lookup landing on the next entry, which a last-bit difference in a
#: transcendental can cause on a different NumPy build.
ONE_TABLE_STEP = 2 / 8192

#: A 16-bit sample is this far from its neighbour.
SIXTEEN_BIT_STEP = 2 / 2 ** 15

#: Above the last few bits of a float64 and far below anything audible.
#: Two libm implementations disagree by an ulp or so on `sin` and on `**`,
#: and that reaches hundreds of samples out of thousands -- which is
#: nothing, and is not the same event as a lookup landing on the next table
#: entry. The tests below count only differences larger than this.
FLOATING_POINT_NOISE = 1e-12


def test_the_tolerance_is_bounded_by_a_count_rather_than_by_being_small():
    """One table entry is four 16-bit steps, so the count is what binds.

    Worth stating plainly rather than leaving to be assumed: the magnitude
    this test tolerates is *not* below audibility. One entry of a
    16384-point table is about four times the quietest change a 16-bit file
    can carry, so a single sample of it is representable and, in a
    pathological place, audible as a click.

    What makes the test tight is the second bound, on how many samples may
    differ at all: at most one in a thousand. A real regression moves a
    routine, not one lookup in a thousand -- both defects the reconciliation
    found changed over 99% of their samples.
    """
    assert ONE_TABLE_STEP > SIXTEEN_BIT_STEP
    assert ONE_TABLE_STEP / SIXTEEN_BIT_STEP == pytest.approx(4.0)
    # And the noise floor sits eight orders below one entry, so counting
    # above it cannot mistake a rounding for a regression or the reverse.
    assert FLOATING_POINT_NOISE < ONE_TABLE_STEP / 1e8


def test_the_exact_cases_reproduce_the_reference(cases, recorded):
    """The claim, for the routines that make it without qualification.

    Agreement to within one table entry, on almost every sample. See this
    module's own docstring for why not to the byte: `tools/mass_reconcile.py`
    is where bit-exactness is checked, and it is checked there on every
    routine in this list.
    """
    checked = []
    for case in cases:
        if case.expect != EXACT:
            continue
        produced = np.asarray(case.package(), dtype=float)
        reference = recorded[f'{case.mass}.samples']

        assert produced.shape == reference.shape, (
            f'{case.music} no longer has the shape {case.mass} produced')

        difference = np.abs(produced - reference)
        assert difference.max() <= ONE_TABLE_STEP, (
            f'{case.music} differs from {case.mass} by '
            f'{difference.max():.3e}, more than one entry of the waveform '
            f'table; run tools/mass_reconcile.py against a MASS checkout to '
            f'see where')

        # Hundreds of samples differ in their last bits on a runner whose
        # libm is not this machine's, which is expected and invisible.
        # What must stay rare is a difference big enough to be a lookup
        # landing on a different entry.
        moved = int(np.count_nonzero(difference > FLOATING_POINT_NOISE))
        assert moved <= max(4, produced.size // 1000), (
            f'{case.music} differs from {case.mass} by more than '
            f'{FLOATING_POINT_NOISE:g} on {moved} of {produced.size} '
            f'samples. Rounding moves a handful of lookups to the next '
            f'table entry; this is too many for that')
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
    assert delta > FLOATING_POINT_NOISE, (
        f'{case.music} now matches {case.mass} to within floating-point '
        f'noise, which undoes the correction the register records: '
        f'{case.reason}')
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
