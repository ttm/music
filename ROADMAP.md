# Repository findings and roadmap

Reviewed 2026-09-15 against `7a8f07e` and the published `music` 1.8.0.
This records the review baseline; implementation and validation progress
for the correctness patch is recorded below.

## State at review

The working tree was clean and matched GitHub's `master`. The 1.8.0 wheel
and source distribution matched PyPI's hashes. Latest CI and docs builds
passed. Local verification found 3,403 passing tests and nine skips, 100%
line and branch coverage, clean Ruff and mypy results, a successful Sphinx
build with warnings treated as errors, and 13 passing examples with the
external singing-engine example skipped.

The package has substantial tests against equations, spectra, musical
properties and recorded MASS outputs. The findings below are parameter
combinations those tests did not adequately check.

## Correctness patch for 1.8.1

### Stereo normalization outside full scale

`normalize_stereo([[0, 1], [1, 2]], remove_bias=False)` returned
`[[-1, 1], [1, 3]]`, violating its documented `[-1, 1]` range. The shared
normalization used a global minimum but an individual channel's range.
`write_wav_stereo` then clipped the second channel and lost its dynamics.
Default mean-removing normalization was unaffected.

The patch fixes the shared affine scale and tests differently offset
channels, their relative excursions, and exported samples.

### File-export fades calculated at the wrong sample rate

Both WAV writers omitted `sample_rate` when applying ADSR. A 100 ms fade
on an 8 kHz file reached full level at 551 ms; at 48 kHz fades were about
8% short. `write_audio` and FLAC inherited the same behavior. Disabled
fades and the default 44.1 kHz rate were unaffected.

The patch forwards the output rate and checks actual file timing at
8, 44.1, 48 and 96 kHz, for mono and stereo. It also fixes a related input:
`fades=np.array([10, 30])` raised an ambiguous-truth-value error before
reaching the helper that already accepts arrays.

### Session ramps mishandling short phases

Two 100 ms callable phases with a one-second interior ramp raised a NumPy
broadcasting error. Three constant phases of 1, 0.1 and 1 seconds with
one-second interior ramps overshot to 1.9 with linear crossfades. A single
100 ms phase with one-second opening and closing ramps silently lost its
closing fade and ended near full scale.

The patch resolves oversized and competing ramps before placing phases.
Tests cover ordinary layouts and callable session duration, both curve
shapes, endpoint fades, short middle phases, and arrays.

### Patch status

- Implementation completed in commit `896b493` on 2026-09-16. Regression
  tests first reproduced 22 normalization/export failures and 31 session
  failures against the original code.
- Shared stereo normalization now uses the combined range; file writers
  forward the output sample rate and accept NumPy fade pairs.
- Session ramps are shortened before placement, sharing one effective
  length between neighbors. Competing ramps retain both ends when there
  are samples for them; one- or two-sample phases with both outer fades
  render silence.
  Callable phases that round to zero samples are not invoked, avoiding
  the synthesis API's zero-count default-duration convention.
- Independent review probed mixed array/callable and mono/stereo layouts,
  checked matching transitions and output duration, and compared ordinary
  layouts with the previous implementation. It also found an unbalanced
  outer-ramp case, now covered by a regression test.
- Validation completed: **3,488 passed, 9 skipped**, with **100% line and
  branch coverage** (2,807 statements and 866 branches). Ruff and mypy
  pass; Sphinx passes with warnings treated as errors; 13 examples pass
  and the external singing-engine example skips. Assessment figures have
  been remeasured.
- The built source distribution passes all 3,488 tests from its unpacked
  contents. The wheel passes `twine check` and, installed in a temporary
  directory outside the checkout, reproduces the corrected normalization,
  fade timing and short-phase session behavior.
- These figures describe the correctness checkpoint before the mutation
  audit's additional tests. The current figures are in `ASSESSMENT.md`.
  Version 1.8.1 carries the patch and the strengthened tests; publishing,
  tagging and updating its archival citation follow `RELEASING.md`.

### Release verification

Version 1.8.1 was published on 2026-09-16:
[PyPI](https://pypi.org/project/music/1.8.1/),
[GitHub](https://github.com/ttm/music/releases/tag/v1.8.1),
[archival DOI](https://doi.org/10.5281/zenodo.22802569).
The complete release gate passed, including 3,556 passing tests and nine
skips, 100% line and branch coverage, and the unpacked sdist's tests.
An installed wheel outside the checkout reproduced all three fixes.
Both uploaded artifact hashes match the local builds. GitHub CI and docs
passed for the tagged commit, including Python 3.10–3.14 and minimum
dependency versions.

**Completed: Zenodo metadata sync, 2026-09-17.** The API recovered after
the prior day's repeated HTTP 504 responses. The published record now has
19 controlled subjects, 45 keywords, the current abstract and the 1.8.1
release notes, verified through Zenodo's DataCite export. The citation
already names the new archive; the package, Git tag and DOI are unchanged.

## Completed: measure whether the tests detect wrong answers

The first mutation audit, the `export` area, covers normalization, export
and session envelopes. It added 68 test cases and strengthened existing assertions;
the final selected tests detect 698 of 767 mutations, up from 617.
The 69 survivors were reviewed and accepted with reasons. No additional
production defect was found. [MUTATION_AUDIT.md](MUTATION_AUDIT.md) records
the reproduction command, tool limitation, runtime and survivor IDs.

## Completed: automate the remaining release checks

The release gate now runs examples, strict article coverage, live MASS
reconciliation and strict vocabulary verification. It validates the
selected reference path and shows the scientific and vocabulary reports,
including known exceptions. Missing resources and unexpected verification
gaps fail; they are not silently skipped.

Every release build now checks the installed wheel outside the checkout,
including package and metadata origins, version, typing marker and audio
samples. CI uses the same checker. `release.py verify` runs the gate and
build checks during development without requiring an unpublished version.
[RELEASING.md](RELEASING.md) documents dependencies, external resources,
the two known vocabulary exceptions and the broad `--skip-gate` bypass.

Validated on 2026-09-17 with
`python tools/release.py verify --mass ../mass`: 3,613 tests passed and
nine skipped with 100% line and branch coverage; docs, examples,
assessment figures, sdist tests and installed-wheel checks passed. The
external checks reproduced 46/47 labelled equations (all 46 testable),
26 exact/5 divergent/4 broken-reference routines, and 17 confirmed
subjects with the two declared exceptions. The new failure-path and
wheel-isolation regressions account for 57 additional test cases.

## Completed: the envelope mutation audit, and the three defects it found

Reviewed 2026-09-20. `tools/mutation_audit.py` now takes an `--area`, and
the second area covers the note-level amplitude envelopes: `am`, `tremolo`,
`tremolos`, `adsr`, `adsr_vibrato`, `adsr_stereo`, `fade` and `cross_fade`.
The test selection was chosen by measuring which tests reach those modules
(`pytest --cov-context=test`, then a query over the coverage database)
rather than by guessing, and covers every line and branch of all three
files, so no mutant went unreached.

The first run killed 439 of 550 mutants. `adsr_vibrato` killed **none** of
its three: its whole body could be replaced by `adsr(**adsr_dict)`,
dropping the note it exists to render, and the suite stayed green. Unlike
the first audit, this one found production defects.

### Three corrections

- **A distorted tremolo was not a real number.** `tremolo` raised the
  signed oscillation straight to `alpha`, so a fractional index returned
  NaN for half of every envelope — behind a NumPy warning, which
  `tests/test_degenerate.py` names as the one thing no routine here may
  return — and an even index rectified the pattern into one that could
  only boost. The index now applies to the magnitude with the sign kept:
  `alpha=1` is bit-identical, taking the branch the correction did not
  touch, so the `T` and `T_` rows of `RECONCILIATION.md` stay
  sample-exact; a whole odd `alpha` agrees to within the last bit. The article gives
  the tremolo no `alpha` at all; `DISCREPANCIES.md` records that.
- **`adsr` never reached zero.** `to_zero` is a duration in milliseconds
  and reached `fade` as a bare ratio where `fade` reads a percentage, a
  hundred times too small: below about 2.3 ms at 44.1 kHz it rounded to no
  samples, so `adsr(to_zero=1)` was byte-identical to `adsr(to_zero=0)`.
  The reference carries the same line, so `AD` and `ADS` are now divergent
  rows of `RECONCILIATION.md` with a stated reason rather than exact ones,
  and `trill` carries the correction through the notes it shapes. The
  register is 24 sample-exact, 7 divergent, 4 where the reference does not
  run.
- **`cross_fade` placed the overlap at the wrong rate**, cutting the fades
  at `sample_rate` while `mix_with_offset` used its own default of
  44.1 kHz, so at any other rate the two sounds met at full level: a
  steady 3 and a steady 5 crossfaded at 8 kHz summed to 8. The same defect
  the 1.8.1 export fades had. No mutant found this one — an argument that
  is absent cannot be mutated — it came out of writing the tests that kill
  the mutants around it. A zero or oversized `duration` also reached NumPy
  as a broadcast failure; both now raise `ValueError` naming the duration.

### Validation

3,663 tests pass with nine skips and 100% line and branch coverage (2,813
statements, 870 branches). Ruff and mypy are clean. The final audit detects
535 of 570 mutations; all 35 survivors were reviewed and accepted, and none
changes what a supported call returns: 23 are equivalent substitutions, 4 a
boundary that cannot bind, 4 an argument the callee does not read, and 4
diagnostic wording. Nothing timed out, was skipped or went untested.
[MUTATION_AUDIT.md](MUTATION_AUDIT.md) records both areas, the survivor
IDs and the reproduction commands.

## Completed: the oscillator vibratos, and the four defects they hid

Reviewed 2026-09-21. The third area, `oscillators`, mutates
`music/core/synths/notes.py`: 1,396 mutations judged by 29 test files that
between them cover every line and branch of it.

**The vibratos are done; the rest of the area is not.** Two passes: the
first went after the defect the envelope area pointed at, the second closed
every routine that carries a vibrato. Between them the five vibrato
routines went from 112 survivors to 22, and three further defects came out.
At that checkpoint, 154 survivors remained unread, 80 of them in
`note_with_vibrato_seq_localization`. The subsequent localization pass is
recorded below; the oscillator area as a whole remains open.

### The defect

The signed `** alpha` corrected in the tremolo also sat in the vibrato of
`note_with_vibrato`, `note_with_two_vibratos`,
`note_with_glissando_vibrato`, `note_with_two_vibratos_glissando` and
`note_with_vibrato_seq_localization`. There the distorted quantity is a
*frequency*, so the NaN did not stay visible: it reached the accumulated
phase and then an `int64` cast, which turns NaN into `INT64_MIN`, and
modulo the table length that is one fixed index.
`note_with_vibrato(duration=0.2, vibrato_freq=5, alpha=0.5)` rendered
4,411 samples of a note and then 4,409 samples of constant −1.0 —
full-scale DC. Every sample finite and inside full scale, so nothing
checking for NaN, finiteness or clipping could see it.

Thirteen mutants of `note_with_vibrato` survived the first run, seven of
them arithmetic edits to the one line that computes the distorted
frequency, and one that swapped the branch with its own `else`. The branch
ran under every test; nothing measured the pitch it produced.

This is the defect `_require_a_ratio` already names for the *glissando*
endpoints — the guard written for the first did not reach the second. The
five glissando uses of `alpha` are unaffected: those raise the article's
own non-negative ramp.

### Three more, from the second pass

- **A sixth routine carried the same signed power.**
  `note_with_vibratos_glissandos` raises its vibrato on a line whose `**`
  ends one line and whose index begins the next, so the text search that
  found the other five could not see it and the first pass recorded five
  where there were six. The list is now from an AST walk over every `Pow`
  node whose exponent names an index; none is left.
- **The second vibrato read the first one's waveform table.** In
  `note_with_two_vibratos` and `note_with_two_vibratos_glissando`, `tabv2`
  and `sec_vibrato_waveform_table` were accepted, documented and used only
  for their length, so a square second vibrato under a sine first one gave
  back two sines; two tables of different lengths raised `IndexError`. The
  reference has the same line, and both reconciliation cases pass the same
  table twice, so `VV` and `PVV` stay sample-exact.
- **A glissando sweeps frequencies below one hertz.** Nothing swept from
  or to a frequency in (0, 1], so `_require_a_ratio`'s "positive" guard
  could have read `> 1` at either end.

Neither of the first two was found by a mutant. One was a name rather than
an operator; the other sat among survivors that read like the rest of the
distorted-path block. Both came out of writing the tests that kill the
mutants around them, as the `cross_fade` sample rate did.

### The correction

One shared `music.utils._signed_power`, used at all seven vibrato sites and
by the tremolo. It returns an index of exactly 1 untouched rather than
computing `x ** 1`, so every undistorted render — which is every case in
`RECONCILIATION.md` — is bit for bit what it was, with no dependence on how
a NumPy build rounds `power`.

### Validation

3,715 tests pass with nine skips and 100% line and branch coverage. Ruff
and mypy are clean. The area now detects 1,248 of 1,402 mutations, up from
1,124 of 1,375; `note_with_vibrato` and `note_with_two_vibratos` are closed
completely, and the glissando forms hold two survivors each. Four
mutants time out rather than answering, in both runs: `trill` accumulates
samples in a `while` loop and those four make it run forever. They are
counted as detected, and the runner no longer calls a run containing them
incomplete.

The test that closed `note_with_vibrato` is the one the defect could not
have survived: a square vibrato table holds each extreme for half a cycle,
so the note is two steady tones, and the frequency of each is *measured*
from its zero crossings and matched against `freq · 2^(±(dev/12)^α)`.

## Completed: sequential localization, 2026-09-23

The bounded pass through `note_with_vibrato_seq_localization` reviewed its
80 survivors. Tests now measure rendered pitch, waveform changes, Doppler
shift, geometric gain and interaural delay, including short and fractional
segments. The pass corrected inconsistent vibrato sample counts, a
one-sample glide that corrupted the remaining phase, and a finished path
that held gain just before its destination. It also added clear refusals
for invalid pitch endpoints and unrenderable movement durations, and made
documented array-like waveform tables work.

The routine now detects **521 of 523 mutations**. The two accepted
survivors change diagnostic decoration or choose an equivalent padding
branch at zero interaural delay. The full oscillator area detects
**1,350 of 1,426**, including the same four known `trill` timeouts.
`MUTATION_AUDIT.md` records the exact snapshot and survivor IDs.

The reference fixture is unchanged. `D_` now explicitly accounts for the
four surplus samples the reference creates and for its pre-destination
tail gain; the existing phase-comparison bound remains intact.

Validation: **3,801 passed, nine skipped**, with 100% line and branch
coverage. Ruff, mypy, strict Sphinx, all 13 runnable examples and the
installed-wheel checks pass. Live MASS reconciliation remains 24 exact,
seven explained divergences and four broken reference routines. The
assessment figures and changelog are current; these are unreleased changes.

## Next: the other oscillator routines

The other 74 survivors remain outside the completed localization pass.
Start with the 18 in `note_with_vibratos_glissandos`, then the 14 in
`note_with_doppler`. The former still has the analogous fractional-duration
and short-glide expressions; this pass did not change that routine or close
its audit.

Two things the three areas have taught, worth carrying into the next one:

- **An absent argument cannot be mutated.** The `cross_fade` sample-rate
  defect was not found by any mutant, because no edit can expose an
  argument that was never passed. It came out of writing the tests that
  kill the mutants around it.
- **A branch can run under every test and still be unasserted.** Both
  defects found so far sat inside a branch with full line *and* branch
  coverage, under arithmetic that nothing measured.

Add an area to `AREAS` rather than widening an existing one. None of the
three is a whole-package mutation score.

## Other maintenance

- **Completed:** correct stale assessment and roadmap prose: interval
  naming exists;
  article coverage is 46/47; mypy checked 47 files; 13 examples ran.
  The assessment now describes the selected figures that are checked;
  expanding those checks remains optional follow-up work.
- **Completed:** refresh the local editable installation without changing
  dependencies. Its old `1.3.0.dist-info` made current code report 1.3.0
  outside the repository; after the release it reports 1.8.1 from either
  location.
  Published artifacts were unaffected.
- **Completed:** installed-wheel CI runs on Linux, macOS and Windows
  with Python 3.12, alongside the full Ubuntu Python 3.10–3.14 matrix and
  minimum-dependency job. Checkout, Python setup and Pages actions now
  use Node 24 releases.
- **Completed:** shape the notes in the documentation landing-page example,
  as the README and tutorial already do, to avoid raw concatenation clicks.

## Feature choices after correctness work

Choose according to the next real use case:

1. **Stimulation interoperability:** consume and emit SSTIM stimulus and
   session specifications, with RDF tooling in an optional extra
   ([#75](https://github.com/ttm/music/issues/75)). Registering the tool in
   SSTIM itself is separate work in that repository
   ([#73](https://github.com/ttm/music/issues/73)).
2. **Musical timbres:** extract waveform tables from WAV recordings before
   adding SoundFont parsing ([#3](https://github.com/ttm/music/issues/3)).
3. **Other bounded musical work:** bell tunings and ambience
   ([#1](https://github.com/ttm/music/issues/1)), or simplifying the singing
   engine while preserving control of pitch and duration
   ([#5](https://github.com/ttm/music/issues/5)).

Partial typing, legacy classes, wavetable aliasing, and externally fetched
HRTF/singing resources remain documented limitations. Broad lint or typing
rewrites come after the demonstrated correctness issues.
