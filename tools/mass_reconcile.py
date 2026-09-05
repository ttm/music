"""Reconcile the package, routine by routine, with the MASS reference.

The package's central claim is fidelity to the MASS framework.  Until this
existed the claim rested on the docstrings.  This runs both implementations on
matched arguments and reports, for each of the reference's 36 routines, either
sample-exact agreement or a divergence with a stated reason.

    python tools/mass_reconcile.py                  # print the register
    python tools/mass_reconcile.py --write-fixture  # refresh the test fixture
    python tools/mass_reconcile.py --register FILE  # write the markdown table

The reference is GPL-3 and this package is MIT, so it is never vendored.  The
fixture holds only the numbers the reference produced, which is why the test
suite can check the register without a MASS checkout present.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import music  # noqa: E402
from music.utils import (WAVEFORM_SAWTOOTH, WAVEFORM_SINE,  # noqa: E402
                         WAVEFORM_SQUARE, WAVEFORM_TRIANGULAR)
from tools.mass_reference import load  # noqa: E402

FIXTURE = (Path(__file__).resolve().parent.parent / 'tests' / 'fixtures'
           / 'mass_reference.npz')

#: Outcomes a routine can have.
#: Markers delimiting the generated table inside a document that carries it.
BEGIN = '<!-- register:begin -->'
END = '<!-- register:end -->'

EXACT = 'exact'
DIVERGENT = 'divergent'
REFERENCE_BROKEN = 'reference-broken'


@dataclass
class Case:
    """One reference routine and the package routine that answers it."""

    mass: str
    music: str
    #: Called with the reference namespace; returns the reference's output.
    reference: Callable
    #: Called with no arguments; returns the package's output.
    package: Callable
    #: What this comparison is expected to show.  Anything other than EXACT
    #: needs a reason, and the tool fails when reality disagrees.
    expect: str = EXACT
    #: Why the two differ, or why the reference cannot run.
    reason: str = ''
    #: A note about how the comparison had to be set up.
    notes: str = ''
    seed: int | None = None
    #: Largest difference the divergence accounts for.  None is unbounded.
    bound: float | None = None
    result: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.expect != EXACT and not self.reason:
            raise ValueError(f'{self.mass}: a divergence needs a reason')


D = 0.1          # seconds; long enough to exercise the maths, small on disk
FS = 44100
#: Sequence arguments mirroring the shapes the reference's own defaults use,
#: with the durations cut down so the fixture stays small.
SEQ_F = (220, 440, 330)
SEQ_D = ((0.01, 0.012), (0.008, 0.015, 0.01),
         (0.006, 0.01, 0.012, 0.004, 0.004))
SEQ_FV = ((2, 6, 1), (0.5, 15, 2, 6, 3))
SEQ_NU = ((2, 1, 5), (4, 3, 7, 10, 3))
SEQ_ALPHA = ((1, 1), (1, 1, 1), (1, 1, 1, 1, 1))
SEQ_D_LOC = SEQ_D + ((0.012, 0.016, 0.006),)
SEQ_ALPHA_LOC = SEQ_ALPHA + ((1, 1, 1),)


def lists(seq):
    """Deep-copy a nested sequence into lists.

    Several reference routines assign into the sequences they are handed.
    """
    if isinstance(seq, (tuple, list)):
        return [lists(item) for item in seq]
    return seq


def build_cases(ns: dict) -> list[Case]:
    """The mapping from the reference's names to the package's."""
    Tr, S, Q, Sa = ns['Tr'], ns['S'], ns['Q'], ns['Sa']
    mono = music.note(freq=330, duration=D, waveform_table=WAVEFORM_SINE)

    # Both sides are handed the reference's own tables, so that a difference
    # in the synthesis is never confused with a difference in a constant.
    # The tables are compared as cases in their own right, at the end.
    def seq_tables():
        return ((Tr, Tr), (S, Tr, S), (S,) * 5)

    return [
        # ---- normalisation ------------------------------------------------
        Case('__n', 'normalize_mono',
             lambda: ns['__n'](mono * 0.3, True),
             lambda: music.normalize_mono(mono * 0.3, True)),
        Case('__ns', 'normalize_stereo',
             lambda: ns['__ns'](np.vstack([mono, mono * 0.5]), True, False),
             lambda: music.normalize_stereo(
                 np.vstack([mono, mono * 0.5]), True, False)),
        # ---- synthesis ----------------------------------------------------
        Case('N', 'note',
             lambda: ns['N'](f=220, d=D, tab=Tr, fs=FS),
             lambda: music.note(freq=220, duration=D,
                                waveform_table=Tr, sample_rate=FS)),
        Case('N_', 'note_with_phase',
             lambda: ns['N_'](f=220, d=D, phase=1.0, tab=Tr, fs=FS),
             lambda: music.note_with_phase(freq=220, duration=D, phase=1.0,
                                           waveform_table=Tr, sample_rate=FS)),
        Case('V', 'note_with_vibrato',
             lambda: ns['V'](f=220, d=D, fv=6, nu=1, tab=Tr, tabv=S, fs=FS),
             lambda: music.note_with_vibrato(freq=220, duration=D,
                                             vibrato_freq=6, max_pitch_dev=1,
                                             waveform_table=Tr,
                                             vibrato_waveform_table=S,
                                             sample_rate=FS)),
        Case('FM', 'note_with_fm',
             lambda: ns['FM'](f=220, d=D, fm=100, mu=2, tab=Tr, tabm=S, fs=FS),
             lambda: music.note_with_fm(freq=220, duration=D, fm=100,
                                        max_fm_deviation=2,
                                        waveform_table=Tr,
                                        fm_waveform_table=S,
                                        sample_rate=FS)),
        Case('P', 'note_with_glissando',
             lambda: ns['P'](f1=220, f2=440, d=D, tab=Tr, method='exp', fs=FS),
             lambda: music.note_with_glissando(start_freq=220, end_freq=440,
                                               duration=D,
                                               waveform_table=Tr,
                                               method='exp', sample_rate=FS)),
        Case('PV', 'note_with_glissando_vibrato',
             lambda: ns['PV'](f1=220, f2=440, d=D, fv=6, nu=1, tab=Tr,
                              tabv=S, fs=FS),
             lambda: music.note_with_glissando_vibrato(
                 start_freq=220, end_freq=440, duration=D, vibrato_freq=6,
                 max_pitch_dev=1, waveform_table=Tr,
                 vibrato_waveform_table=S, sample_rate=FS)),
        Case('VV', 'note_with_two_vibratos',
             lambda: ns['VV'](f=220, d=D, fv1=2, fv2=6, nu1=2, nu2=4,
                              tab=Tr, tabv1=S, tabv2=S, fs=FS),
             lambda: music.note_with_two_vibratos(
                 freq=220, duration=D, vibrato_freq=2,
                 secondary_vibrato_freq=6, nu1=2, nu2=4, waveform_table=Tr,
                 vibrato_waveform_table=S,
                 sec_vibrato_waveform_table=S, sample_rate=FS)),
        Case('PVV', 'note_with_two_vibratos_glissando',
             lambda: ns['PVV'](f1=220, f2=440, d=D, fv1=2, fv2=6, nu1=2,
                               nu2=0.5, tab=Tr, tabv1=S, tabv2=S, fs=FS),
             lambda: music.note_with_two_vibratos_glissando(
                 start_freq=220, end_freq=440, duration=D, vibrato_freq=2,
                 secondary_vibrato_freq=6, max_pitch_dev=2,
                 secondary_max_pitch_dev=0.5,
                 waveform_table=Tr, tabv1=S,
                 tabv2=S, sample_rate=FS)),
        Case('PV_', 'note_with_vibratos_glissandos',
             lambda: ns['PV_'](f=list(SEQ_F), d=lists(SEQ_D), fv=lists(SEQ_FV),
                               nu=lists(SEQ_NU), alpha=lists(SEQ_ALPHA),
                               tab=lists(seq_tables()), fs=FS),
             lambda: music.note_with_vibratos_glissandos(
                 freqs=SEQ_F, durations=SEQ_D, vibratos_freqs=SEQ_FV,
                 vibratos_max_pitch_devs=SEQ_NU, alpha=SEQ_ALPHA,
                 waveform_tables=seq_tables(), sample_rate=FS)),
        Case('trill', 'trill',
             lambda: ns['trill'](f=[220, 440], ft=17, d=D, fs=FS),
             lambda: music.trill(freqs=[220, 440], notes_per_second=17,
                                 duration=D, sample_rate=FS),
             expect=DIVERGENT, bound=2.5e-4,
             reason='trill takes no waveform table, so it synthesizes through '
                    "the package's corrected triangular table while the "
                    'reference uses its own'),
        Case('noises', 'noise',
             lambda: ns['noises'](ntype='brown', d=D, fs=FS),
             lambda: music.noise(noise_type='brown', duration=D,
                                 sample_rate=FS),
             seed=0,
             expect=REFERENCE_BROKEN,
             reason='the reference indexes coefs[Lambda/2] with a true- '
                    'division float, which has raised IndexError since Python '
                    '3; under Python 2 it ran, but into a real-valued '
                    'coefficient array that discarded the imaginary part of '
                    'every randomized phase, leaving a spectrum with no phase '
                    'randomization at all'),
        # ---- envelopes ----------------------------------------------------
        Case('T', 'tremolo',
             lambda: ns['T'](d=D, fa=6, dB=10, taba=S, fs=FS),
             lambda: music.tremolo(duration=D, tremolo_freq=6, max_db_dev=10,
                                   waveform_table=S, sample_rate=FS)),
        Case('T_', 'tremolos',
             lambda: ns['T_'](d=lists(SEQ_D[:2]),
                              fa=lists(SEQ_FV[:1] + ((5, 6.2, 21),)),
                              dB=lists(SEQ_NU[:1] + ((5, 7, 9),)),
                              alpha=lists(SEQ_ALPHA[1:3][:1] + ((1, 1, 1),)),
                              taba=lists(seq_tables()[:2]), fs=FS),
             lambda: music.tremolos(durations=SEQ_D[:2],
                                    tremolo_freqs=SEQ_FV[:1] + ((5, 6.2, 21),),
                                    max_db_devs=SEQ_NU[:1] + ((5, 7, 9),),
                                    alpha=SEQ_ALPHA[1:3][:1] + ((1, 1, 1),),
                                    waveform_tables=seq_tables()[:2],
                                    sample_rate=FS),
             notes='the reference assigns into its own arguments, so it '
                   'requires lists where the package accepts any sequence'),
        Case('AM', 'am',
             lambda: ns['AM'](d=D, fm=50, a=0.4, tabm=S, fs=FS),
             lambda: music.am(duration=D, fm=50, max_amplitude=0.4,
                              waveform_table=S, sample_rate=FS)),
        Case('AD', 'adsr',
             lambda: ns['AD'](d=D, A=20, D=20, S=-5, R=50, fs=FS),
             lambda: music.adsr(envelope_duration=D, attack_duration=20,
                                decay_duration=20, sustain_level=-5,
                                release_duration=50, sample_rate=FS)),
        Case('ADS', 'adsr_stereo',
             lambda: ns['ADS'](d=D, A=20, D=20, S=-5, R=50, fs=FS),
             lambda: music.adsr_stereo(duration=D, attack_duration=20,
                                       decay_duration=20, sustain_level=-5,
                                       release_duration=50, sample_rate=FS)),
        Case('L', 'loud',
             lambda: ns['L'](d=D, dev=10, alpha=1, method='exp', fs=FS),
             lambda: music.loud(duration=D, trans_dev=10, alpha=1,
                                method='exp', sample_rate=FS)),
        Case('L_', 'louds',
             lambda: ns['L_'](d=(0.012, 0.01, 0.008), dev=(5, -10, 20),
                              alpha=(1, 0.5, 20), method=('exp',) * 3, fs=FS),
             lambda: music.louds(durations=(0.012, 0.01, 0.008),
                                 trans_devs=(5, -10, 20), alpha=(1, 0.5, 20),
                                 method=('exp',) * 3, sample_rate=FS)),
        Case('F', 'fade',
             lambda: ns['F'](d=D, out=True, method='exp', dB=-80, fs=FS),
             lambda: music.fade(duration=D, fade_out=True, method='exp',
                                db=-80, sample_rate=FS)),
        # ---- spatialisation ------------------------------------------------
        Case('loc', 'localize',
             lambda: ns['loc'](sonic_vector=mono, theta=30, dist=1, fs=FS),
             lambda: music.localize(sonic_vector=mono, theta=30, distance=1,
                                    sample_rate=FS)),
        Case('loc2', 'localize_linear',
             lambda: ns['loc2'](sonic_vector=mono, theta1=70, theta2=-70,
                                dist1=0.1, dist2=0.1, fs=FS),
             lambda: music.localize_linear(sonic_vector=mono, theta1=70,
                                           theta2=-70, dist=0.1,
                                           sample_rate=FS),
             expect=REFERENCE_BROKEN,
             reason='the reference declares dist1 and dist2 and its body '
                    'reads an undefined dist, so loc2 raises NameError on '
                    'every call and has never run'),
        Case('loc_', 'localize2',
             lambda: ns['loc_'](sonic_vector=mono, theta=-70,
                                method='ifft', fs=FS),
             lambda: music.localize2(sonic_vector=mono, theta=-70,
                                     method='ifft', sample_rate=FS),
             expect=DIVERGENT, bound=1.1,
             reason='four corrections the package carries and documents in '
                    'place: the FFT bin spacing read 2*fs/Lambda rather than '
                    'fs/Lambda, so every frequency was an octave high; the '
                    'interaural delay was applied with the sign that advances '
                    'the far ear rather than delaying it; the delay was '
                    'wrapped into one period of f before becoming a phase, '
                    'which is redundant and wrapped the two branches '
                    'inconsistently; and the reference prints rather than '
                    "raises on an unknown method. The reference's brute "
                    'branch additionally builds n.zeros((2, maxsize)) from a '
                    'float and raises TypeError'),
        Case('D', 'note_with_doppler',
             lambda: ns['D'](f=220, d=D, tab=Tr, x=(-10, 10), y=(1, 1), fs=FS),
             lambda: music.note_with_doppler(freq=220, duration=D,
                                             waveform_table=Tr, x=(-10, 10),
                                             y=(1, 1), sample_rate=FS)),
        Case('D_', 'note_with_vibrato_seq_localization',
             lambda: ns['D_'](f=list(SEQ_F), d=lists(SEQ_D_LOC),
                              fv=lists(SEQ_FV), nu=lists(SEQ_NU),
                              alpha=lists(SEQ_ALPHA_LOC), x=[-10, 10, 5, 3],
                              y=[1, 1, 0.1, 0.1], method=['lin', 'exp', 'lin'],
                              tab=lists(seq_tables()), fs=FS),
             lambda: music.note_with_vibrato_seq_localization(
                 freqs=SEQ_F, durations=SEQ_D_LOC, vibratos_freqs=SEQ_FV,
                 max_pitch_devs=SEQ_NU, alpha=SEQ_ALPHA_LOC,
                 x=(-10, 10, 5, 3), y=(1, 1, 0.1, 0.1),
                 method=('lin', 'exp', 'lin'),
                 waveform_tables=seq_tables(), sample_rate=FS),
             expect=DIVERGENT, bound=2.5e-4,
             reason='the package folds the running phase into one table '
                    'period as it goes rather than accumulating it with '
                    'cumsum, so the two round to different table indexes at a '
                    'boundary; the difference is bounded by one step of the '
                    'table and does not grow with the length of the render '
                    '(issue #102)'),
        # ---- filters ------------------------------------------------------
        Case('FIR', 'fir',
             lambda: ns['FIR'](samples=np.array([1.0, 0.5, 0.25]),
                               sonic_vector=mono, freq=False, max_freq=False),
             lambda: music.fir(samples=np.array([1.0, 0.5, 0.25]),
                               sonic_vector=mono, freq=False, max_freq=False),
             expect=REFERENCE_BROKEN,
             reason="the reference's convolve recurses into itself in both "
                    'branches and never terminates, so FIR raises '
                    'RecursionError for every input; its frequency-domain '
                    'branch also builds a symmetric kernel and then discards '
                    'it, convolving with the raw samples either way'),
        Case('IIR', 'iir',
             lambda: ns['IIR'](sonic_vector=mono, A=np.array([1.0, -0.5]),
                               B=np.array([0.3])),
             lambda: music.iir(sonic_vector=mono, a=[1.0, -0.5], b=[0.3]),
             notes='the reference multiplies its coefficients elementwise, so '
                   'it requires arrays where the package accepts lists'),
        Case('R', 'reverb',
             lambda: ns['R'](d=D, d1=0.02, decay=-50, fs=FS),
             lambda: music.reverb(duration=D, first_phase_duration=0.02,
                                  decay=-50, sample_rate=FS),
             seed=0,
             expect=REFERENCE_BROKEN,
             reason='the reference reads an undefined decay1 where its own '
                    'signature declares decay, so R raises NameError on every '
                    'call and has never run'),
        # ---- mixing and rhythm --------------------------------------------
        Case('mix2', 'mix2',
             lambda: ns['mix2']([mono, mono * 0.5], end=False, offset=0,
                                fs=FS),
             lambda: music.mix2([mono, mono * 0.5], end=False, offset=0,
                                sample_rate=FS),
             notes="compared without an offset: the reference's offset "
                   'branch zero-pads by the whole offset sequence rather '
                   "than by each vector's own offset, and raises for any "
                   'offset given'),
        Case('rhythymToDurations', 'rhythm_to_durations',
             lambda: ns['rhythymToDurations'](durations=[4, 2, 2, 4],
                                              duration=0.25),
             lambda: music.rhythm_to_durations(durations=[4, 2, 2, 4],
                                               duration=0.25)),
        # ---- tables -------------------------------------------------------
        Case('Tr', 'WAVEFORM_TRIANGULAR', lambda: Tr,
             lambda: WAVEFORM_TRIANGULAR,
             expect=DIVERGENT, bound=2.5e-4,
             reason='the reference builds the triangle as hstack((ramp, '
                    'ramp[::-1])), which duplicates the sample at the peak '
                    'and tops out at 1 - 2/8192 instead of 1; the package '
                    'reaches full amplitude at the midpoint of its period'),
        Case('S', 'WAVEFORM_SINE', lambda: S, lambda: WAVEFORM_SINE),
        Case('Q', 'WAVEFORM_SQUARE', lambda: Q, lambda: WAVEFORM_SQUARE),
        Case('Sa', 'WAVEFORM_SAWTOOTH', lambda: Sa, lambda: WAVEFORM_SAWTOOTH,
             expect=DIVERGENT, bound=1.5e-4,
             reason='the reference ramps with linspace(-1, 1, Lt) including '
                    'the endpoint, so its step is 2/16383 and the table does '
                    'not tile: the wrap is a jump of 2.0 rather than one '
                    'step. The package excludes the endpoint'),
    ]


def run(case: Case) -> dict:
    """Run both sides of a case and classify the outcome."""
    out: dict = {'mass': case.mass, 'music': case.music}
    for side, call in (('reference', case.reference),
                       ('package', case.package)):
        if case.seed is not None:
            np.random.seed(case.seed)
        try:
            out[side] = np.asarray(call(), dtype=float)
        except Exception as exc:  # noqa: BLE001 - the failure is the finding
            out[side + '_error'] = f'{type(exc).__name__}: {exc}'
    if 'reference_error' in out:
        out['status'] = REFERENCE_BROKEN
        return out
    if 'package_error' in out:
        out['status'] = 'package-error'
        return out
    a, b = out['reference'], out['package']
    if a.shape != b.shape:
        out['status'] = DIVERGENT
        out['delta'] = float('nan')
        out['shapes'] = (a.shape, b.shape)
        return out
    delta = float(np.max(np.abs(a - b))) if a.size else 0.0
    out['delta'] = delta
    out['status'] = EXACT if delta == 0.0 else DIVERGENT
    if case.bound is not None and delta > case.bound:
        out['over_bound'] = case.bound
    return out


def digest(array: np.ndarray) -> str:
    """A stable digest of an array's exact bytes, shape and dtype."""
    h = hashlib.sha256()
    h.update(str(array.shape).encode())
    h.update(str(array.dtype).encode())
    h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def write_fixture(cases: list[Case]) -> dict:
    """Record what the reference produced, as samples.

    The samples rather than a digest of them. A digest would be smaller, and
    it was what this wrote first, but it asserts something untrue: that two
    NumPy builds agree on the last bit of `np.sin`. They need not, nothing
    promises they will, and CI proved they do not -- so a fixture recorded
    here failed on a runner while the package was correct.

    What survives that is the size of the difference, which is what the test
    reads these back for. This tool stays the bit-exact check, because it
    runs both implementations in one process against one NumPy, where
    sample-exact is a claim that means something.

    Nothing here is reference source: it is the numbers that came out of
    running it, so the fixture carries none of the reference's licence into
    this repository.
    """
    payload: dict = {}
    for case in cases:
        output = case.result.get('reference')
        if output is None:
            continue
        payload[f'{case.mass}.samples'] = output
        payload[f'{case.mass}.digest'] = np.array(digest(output))
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE, **payload)
    return payload


def register_rows(cases: list[Case]) -> list[dict]:
    """The register, as rows ready to render or to assert against."""
    rows = []
    for case in cases:
        r = case.result
        rows.append({
            'mass': case.mass,
            'music': case.music,
            'expect': case.expect,
            'status': r['status'],
            'delta': r.get('delta'),
            'error': r.get('reference_error') or r.get('package_error') or '',
            'reason': case.reason,
            'notes': case.notes,
            'agrees': r['status'] == case.expect and 'over_bound' not in r,
        })
    return rows


def render_register(rows: list[dict]) -> str:
    """The register as a Markdown table."""
    label = {EXACT: 'exact', DIVERGENT: 'divergent',
             REFERENCE_BROKEN: 'reference does not run'}
    lines = ['| MASS | `music` | Outcome | Why |', '|---|---|---|---|']
    for row in rows:
        why = row['reason'] or 'sample-exact agreement'
        if row['notes']:
            why += f" ({row['notes']})"
        if row['status'] == REFERENCE_BROKEN and row['error']:
            why += f" — `{row['error'].split(':')[0]}`"
        elif row['delta']:
            why += f"; max&nbsp;\\|Δ\\|&nbsp;=&nbsp;{row['delta']:.3g}"
        lines.append(f"| `{row['mass']}` | `{row['music']}` | "
                     f"{label[row['status']]} | {why} |")
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mass', help='path to a MASS checkout')
    parser.add_argument('--write-fixture', action='store_true',
                        help='refresh tests/fixtures/mass_reference.npz')
    parser.add_argument('--register',
                        help='write the markdown register to FILE')
    args = parser.parse_args()

    ns = load(args.mass)
    cases = build_cases(ns)
    for case in cases:
        case.result = run(case)
    rows = register_rows(rows_cases := cases)

    width = max(len(c.mass) for c in cases) + 2
    disagreements = []
    for case, row in zip(rows_cases, rows):
        detail = row['error'] or (f"max |Δ| = {row['delta']:.3e}"
                                  if row['delta'] is not None else '')
        mark = ' ' if row['agrees'] else '!'
        print(f"{mark} {case.mass:<{width}} {case.music:<38} "
              f"{row['status']:<17} {detail}")
        if not row['agrees']:
            disagreements.append(
                f"{case.mass}: register says {case.expect}, "
                f"measurement says {row['status']}"
                + (f" (max |Δ| = {row['delta']:.3e}, over the "
                   f"{case.bound:.3g} the reason accounts for)"
                   if 'over_bound' in case.result else ''))

    counts = {status: sum(r['status'] == status for r in rows)
              for status in (EXACT, DIVERGENT, REFERENCE_BROKEN)}
    print(f"\n{counts[EXACT]} sample-exact, {counts[DIVERGENT]} divergent, "
          f"{counts[REFERENCE_BROKEN]} where the reference does not run "
          f"({len(rows)} compared)")

    if args.write_fixture:
        payload = write_fixture(cases)
        print(f'wrote {FIXTURE.relative_to(FIXTURE.parents[2])} '
              f'({len(payload)} entries, '
              f'{FIXTURE.stat().st_size / 1024:.0f} KiB)')

    if args.register:
        target = Path(args.register)
        table = render_register(rows)
        if target.is_file() and BEGIN in target.read_text():
            existing = target.read_text()
            head = existing.split(BEGIN)[0]
            tail = existing.split(END)[1]
            target.write_text(f'{head}{BEGIN}\n{table}{END}{tail}')
        else:
            target.write_text(table)
        print(f'wrote {args.register}')

    if disagreements:
        print('\nThe register disagrees with the measurement:')
        for line in disagreements:
            print(f'  {line}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
