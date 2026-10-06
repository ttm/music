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
- Validation completed: **3,488 passed, 9 doctest items skipped by explicit
  `+SKIP` directives**, with **100% line and branch coverage** (2,807
  statements and 866 branches). Ruff and mypy
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
every routine that carries a vibrato. Those passes left 22 survivors across
the five unlocalized vibrato routines, and three further defects came out.
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

## Completed: the unlocalized sequence and Doppler oscillator

Reviewed 2026-09-23, following the localization checkpoint in `c01827f`.
The next two targets were the 18 survivors in
`note_with_vibratos_glissandos` and the 14 in `note_with_doppler`.

Twelve new sequence regression cases failed against the old source. The
routine now floors vibrato durations to whole samples, renders one-sample
glides without corrupting subsequent phase, rejects nonpositive pitch
endpoints, and accepts nested list/tuple tables while preserving existing
array dtypes. The independent tests measure curved glides, multiple
vibratos, changing timbres and the state held after each sequence ends.

The Doppler pass found no production defect. It now measures each ear's
radial frequency and inverse-distance gain, temperature effects, initial
delay, diagonal motion and whole-waveform symmetries, including empty and
single-sample renders. Its docstring explicitly accounts for the initial
stereo delay padding in the returned sample count.

The `PV_` MASS fixture remains unchanged; its comparison accounts for the
four removed fractional-duration samples while checking the complete
original waveform and a separate render with the old vibrato counts.
Live reconciliation is now **23 exact, eight explained divergences and
four broken reference routines**.

The unlocalized sequence detects **151 of 151 mutants**. Doppler detects
**181 of 182**, with one equivalent centered-source delay branch accepted.
`MUTATION_AUDIT.md` records the numerical checks and accepted survivor.

Validation: **3,862 passed, nine skipped**, with **100% line and branch
coverage** (2,822 statements and 872 branches). Ruff, mypy, strict Sphinx,
all 13 runnable examples, strict article coverage and installed-wheel
checks pass. Independent review found no remaining issue.

## Completed: the remaining oscillator routines

Reviewed 2026-09-23. The last 42 oscillator survivors were 17 refusal-text
edits, 14 changed defaults, four equivalents and seven unasserted
behaviors. Seventeen regression cases failed against the old source:

- `trill` passed its sample rate to `note` but not to `adsr`, so at 8 kHz
  every attack, decay and release lasted 5.5 times as long.
- A one-sample glissando divided zero by zero in three routines, reading
  an arbitrary table entry. It now sounds the starting frequency.
- An exponential path refused a coordinate held at zero, such as a source
  straight ahead. It now holds any coordinate that does not change.

New tests measure short glissando endpoints, trills at 8 kHz and at one
note a second or fewer, single zero path endpoints, which refusals
suggest `method="lin"`, and the remaining routines' declared defaults.

The area now detects **1,442 of 1,460 mutations**. All 18 survivors are
accepted as refusal text or equivalent; `MUTATION_AUDIT.md` lists them.
The MASS reconciliation fixtures are unaffected, and `DISCREPANCIES.md`
records the three edge cases where the package now departs from the
reference.

## Completed: the stimulus generators

Reviewed 2026-09-23, as the fourth mutation area, `stimuli`. The first run
left **75 of 404** mutations alive: the module was tested for its shapes
and spectra, not its samples. The amplitude envelope, the orbit's
trajectory, the isochronic ramp, the sign of a frequency sweep and every
carrier's sample rate went unmeasured.

Ten regression cases failed against the old source:

- `isochronic_tones` raised `ZeroDivisionError` at a zero rate with a
  ramp, and ran backwards at a negative one. It now requires a positive
  `pulse_rate`.
- `amplitude_modulation` and `frequency_modulation` treated a zero rate
  as their modulator held at its first entry, halving the carrier or
  shifting its pitch. Zero now leaves the carrier alone, as
  `modulated_noise` documents. Both refuse a negative rate, as it does.
- `spatial_motion` refuses a stereo sound with a clear message.

The area now detects **423 of 427**, with four equivalent or text-only
survivors. `MUTATION_AUDIT.md` records them.

The release tooling also gained a guard. The Zenodo summary keeps only
each changelog entry's bold headline, and silently dropped the entries
without one. The gate, the sync and a test on every push now refuse them.

## Completed: the localization filters

Reviewed 2026-09-23, as the fifth mutation area, `localization`. The first
run left **85 of 611** mutations alive, 54 of them in `localize2`, whose
`brute` method survived even losing its accumulation.

Thirty-one regression cases failed against the old source:

- `brute` resynthesized each partial a quarter cycle early, reading the
  FFT's cosine angles into a sine table, and sized its buffer without
  `zeta`, some thirty samples past any delay.
- The fractional delay behind `localize_linear` and `spatial_motion` held
  the first sample for the whole interaural delay, so a click at the
  start reached the far ear as a 27-sample plateau.
- A source exactly on an ear returned NaN from the moving routines, and
  `localize` divided by zero on the left ear.
- `localize` rejected a list, although documented as array_like.

A zero angle read as "use `x` and `y`" in `localize` and `localize2`, so
no source could be placed at zero degrees. `None` is now the sentinel,
noted for anyone upgrading. The area detects **631 of 652**, with 21
survivors that are text, equivalent at a zero angle or checked by
experiment to be equivalent conversions. `MUTATION_AUDIT.md` records
them.

## Completed: a review of the work since 1.8.3

Reviewed 2026-09-23. A package-wide scan for the patterns the audits kept
finding found two more of the first. `reverb` asked `noise` for a band up
to half its rate without passing the rate, so at 8 kHz its tail sat below
680 Hz. The three localization routines rendered their default sound at
44.1 kHz whatever rate they were given. The other patterns turned up no
further cases: a zero read as "not given", an FFT angle read into a sine
table, and a clamped read outside a signal.

A second pass widened the scan and found five more defects:

- `pan_transitions` *added* its envelopes to the sound it was given
  rather than scaling it by them, so nothing was panned. A parameter the
  function never reads was the lead: `method`, documented as ignored.
- `louds` given sample counts passed its arguments one place over, so
  `trans_devs` was never read. A scan for positional arguments landing
  in a differently named parameter finds no other case.
- `reverb` failed for a response with no second period, and `trill` for
  notes shorter than a sample: both handed `number_of_samples=0` to a
  routine that reads it as "not given".
- `localize2` refused an empty sound, and `sing` a note outside its
  table, with a numpy and a `KeyError` message respectively.

It also gave `sing` its missing docstring, replaced the MASS names (`d`,
`nsamples`, `L()`) still used in parameter descriptions, and corrected
nine misspellings. Calling every routine that takes a sound with a plain
list found no other that refuses one.

A third pass tried what the first two had not: rates written into
function bodies as literals (none but one checked on purpose), every
duration rendered at 8 and 16 kHz (all 28 scale), casts that could turn
a NaN into a table index, and the modules no audit has reached. It found
`stretches` falling short of the durations it was given, by up to 26,460
samples for a short fragment; a negative start in the sequencer failing
as a broadcast error; and `convert_to_stereo` returning one channel for
a single row.

## Released: 1.9.0, 2026-09-24

The stimulus, localization and review corrections went out as 1.9.0, a
minor release because some calls now return something different; its
changelog section opens with a note for anyone upgrading. Zenodo archived
it as 10.5281/zenodo.22947874, about ninety minutes after accepting the
release event rather than the usual two.

## Completed: the `utils.py` mutation audit

Reviewed 2026-09-25. The nine-file selection passes 996 tests and covers all
154 branch outcomes in `music/utils.py`. The mutation run killed 821 of 915
mutants, timed out four, and left 90 reviewed survivors; none were untested
or skipped. `MUTATION_AUDIT.md` records the grouped survivor review.

The audit found defects that line and branch coverage had not exposed:

- `profile` squared narrow integer and `float32` samples before widening
  them, and described a multichannel audio buffer's channel count as its
  sample count. Statistics now use `float64`; multichannel duration uses the
  last axis, and integer stereo is flattened before block analysis.
- `convert_to_stereo` returned a one-row mono array as one channel and could
  overflow while summing integer PCM. It now duplicates that row and sums in
  `float64`.
- `resolve_stereo` changed the caller's argument mapping while splitting its
  channels. It now uses copied per-channel arguments.
- `rhythm_to_durations` treated 120 BPM as a two-second beat and did not let
  `total_duration` override BPM. It now uses `60 / BPM` seconds per beat,
  honors the documented precedence and accepts NumPy frequency sequences.
  The MASS comparison and release notes now record this deliberate API
  correction.

The final suite run passed 4,137 tests; nine doctest items were skipped by
explicit `+SKIP` directives.

## Completed: the remaining `core/filters/` mutation audit

Reviewed 2026-09-25. The bounded selection covers filter design, FIR/IIR,
loudness transitions, reverberation and stretching, beyond the earlier
ADSR/fade and localization audits. Its 178 selected tests detect 505 of 531
mutants; all 26 survivors were reviewed. The run found and fixed a loudness
padding defect, stricter method validation, missing sample-rate and shape
guards, and the loss of iterable stretch durations. See
`MUTATION_AUDIT.md` for the scope, reproduction command and survivor review.

The full repository suite passed 4,175 tests; nine doctest items were skipped
by explicit `+SKIP` directives.

## Completed: the `music/theory/` mutation audit

Reviewed 2026-09-27. Only the two theory test files and eight parametrized
cases elsewhere reach scales, chords and intervals. The first run detected
192 of 204 mutants; eleven of the twelve survivors were missing tests, on
interval degrees the article's worked examples do not use, the minor
ninth, sub-hertz frequencies and three unread defaults. With an exhaustive
oracle for the interval notation, the final run detects 213 of 214, and the
one survivor is equivalent. Reading around the mutants found four defects
none could show: `interval_names(16.0)` named `M10.0`, `consonance`
truncated a fractional interval, `interval` stripped whitespace after the
table lookup, and `interval_between` failed inside `round()` for NaN and
infinity. See `MUTATION_AUDIT.md`.

The full repository suite passed 4,495 tests; nine doctest items were skipped
by explicit `+SKIP` directives.

## Completed: one set of transition names

Reviewed 2026-09-27. `loud` matched its method exactly, `fade` by
substring and the glissandi against `"exp"` alone, so `adsr` could accept
a name in its attack and refuse it in its decay, and `"exponential"` swept
a glissando linearly. All five routines now share one check. The four
mutation areas that cover them were re-run at that commit; none had an
untested mutant or a survivor in the new code.

## Completed: the `music/structures/` mutation audit

Reviewed 2026-09-27. Probing before the first run found
`transpose_permutation` reversing three-cycles and merging swaps, an
unread `hunts` argument, `print_peal` failing at nine bells, `even_odd`
accepting non-permutations and one element failing inside sympy. The
first run then left 179 of 725 mutants: `print_peal`'s colours, which no
test could see without a terminal, and permutation families the tests
counted rather than read. Tests against definitions that do not go
through sympy's groups found two more: as many hunts as bells passed the
check and crashed, and sympy's two-element alternating group sits on one
point. The final run detects 669 of 684, and the 15 survivors are
equivalent or write a trace nothing reads. See `MUTATION_AUDIT.md`.

The full repository suite passed 4,779 tests; nine doctest items were skipped
by explicit `+SKIP` directives.

## Completed: the `noises` and `sequencer` mutation audits

Reviewed 2026-09-28. `noise` rounded its band down at both ends and never
drew a phase for an odd length's highest component; `gaussian_noise`
doubled what lay past the Nyquist frequency; both, and `silence`, took
degenerate arguments to silence, NaN or numpy errors. The sequencer's
tests never compared a render with the notes it is made of, so 44 of 176
mutants survived at first; all 157 of the final run are detected, and
`noises` detects 280 of 285. A surprising kill led to an unseeded test
that failed for one random state in four; eight runs of the whole suite
with every test seeded differently found no other. See
`MUTATION_AUDIT.md`.

The full repository suite passed 4,867 tests; nine doctest items were skipped
by explicit `+SKIP` directives.

## Completed: the rest of the package

Reviewed 2026-09-28: `bonds`, `tables`, `hrtf`, `singing` and `legacy`,
which leaves no module outside a mutation area but `symmetry.py`, which
only re-exports. Among what they found: a stepped bond that read a
generator of thresholds from where it last stopped; a NaN elevation read
as -40 degrees; a singing requirement, `abc2midi`, the check left out;
scores written an octave above their reference, which the default
transposition made up for; fractional durations written into ABC as
decimals; and a legacy `V_` that passed on only the pitch, so every
note a `Being` played lasted two seconds. See `MUTATION_AUDIT.md`.

On 2026-09-29 the singing engine was installed with everything it needs
and run end to end, which it had not been in this work. It had never read
its configuration, since Perl 5.26 stopped `do` looking in the current
directory, so `lang`, `transpose` and `effect` did nothing; it needed
`sox` and four Perl modules the check did not ask for; and a failed
render passed `make` unseen. `test_singing_engine.py` now sings and
measures the pitch wherever the engine is set up.

The sixteen areas have taught, worth carrying into a later audit:

- **An absent argument cannot be mutated.** The `cross_fade` and `trill`
  sample-rate defects were not found by any mutant, because no edit can
  expose an argument that was never passed. Both came out of reading the
  code around the mutants.
- **Sort survivors by what they change, not where they are.** Grouped by
  function, the last 42 oscillator survivors looked like pitch-curve
  work; most were refusal text and defaults.
- **A spectrum is not a sample.** A test that measures where energy sits
  lets through every edit that keeps the rate.
- **An exact position needs exact inputs.** `sin(pi)` is not zero, so a
  source "on the left ear" at 180 degrees never reached the guard it was
  meant to test.
- **A branch can run under every test and still be unasserted.** The
  defects found so far sat inside branches with full line *and* branch
  coverage, under arithmetic that nothing measured.
- **A finite domain can be enumerated.** The article's three compound
  intervals missed every degree where a wrong octave count shows; all 154
  quality and degree pairs to three octaves take a fraction of a second.
- **A survivor can be the fix.** `nhunts >= nelements` survived because
  nothing tried as many hunts as bells, which crashed.
- **A kill can be a flake.** mutmut reorders tests; a surprising kill is
  worth running by hand in its order.
- **The filesystem and the import can hide a mutant.** macOS finds
  `FULL` as `full`, and a table built at import is built before mutmut
  forks. Record what is used, and build afresh in the test.

Add each area to `AREAS` rather than widening an existing one, and select
its tests with per-test coverage contexts. Every module is now in an
area, but the sixteen bounded audits are still not a whole-package
mutation score: each measures its own sources against its own
selection. A change to a module is the time to re-run its area.

## Singing: next

Decided on 2026-09-30: keep eCantorix as the reference, fix it where it
lives, and build the package's own singer beside it, so the two can be
compared. Done: the fork's fixes and pin, the upstream pull request, the
`psola` backend and `tools/compare_singing.py`. Decided on 2026-10-01,
after listening: PSOLA is the default, since it installs with a system
package and a pip extra, where eCantorix is a cloned Perl engine with
four Perl modules, abc2midi and sox; eCantorix stays, behind
`backend="ecantorix"`, as the reference. Next, in
order:

1. **Listened, on 2026-10-01**, to ten scores sung by both from the same
   espeak, matched in loudness: PSOLA ties, and wins in some, sounding
   less artificial, but some of its syllables are less clear. Listening
   also found two of the backend's bugs, fixed, and that neither backend
   sang French "ques": espeak says it as a bare /k/, and both now give it
   a schwa, as a singer does.
2. **Clearer syllables, done 2026-10-06.** eCantorix asks espeak for the
   speed that fits each note, down to 80 words a minute; PSOLA took each
   syllable at espeak's speaking speed and stretched its whole voiced
   part, consonant transitions included.
   `python tools/compare_singing_timing.py` (2026-10-04) sings five
   timings beside eCantorix, matched in level, for listening, and
   `tools/score_singing_asr.py` (2026-10-05) has Whisper hear them. On the
   48 one-syllable words of `--words`, sung a word a note at four lengths,
   Whisper `small` heard the first timing lose its words as the notes
   lengthen: 16 of 48 at a quarter of a second, 7 at one and at two
   seconds. Fitting espeak's speed helped long notes and hurt short ones,
   where espeak made to speak faster than its default 175 words a minute
   is heard less than PSOLA compressing its usual speech; eCantorix, which
   does that, was heard least, 2 of 48. Holding an estimated vowel nucleus
   did more with espeak slowed than alone. `slowed` keeps what helped:
   espeak only ever slowed, and the nucleus held only on a note longer
   than the syllable, so a short note is sung exactly as before. Whisper
   `small` heard 69 of 192 words with it, against 44 with the first timing
   and 59 with eCantorix, and `large-v3-turbo` 106, against 72 and 90;
   pitch and length are as before. Heard on 2026-10-06, it was the best of
   the six in general, though not fantastic, and not always clearly better
   than all the others, and it is now the psola backend's timing. Where
   words are still unclear, the nucleus is a place to look: it is found by
   energy, not by phonemes, so a voiced consonant or a diphthong's glide
   can be held instead of a vowel.
3. **Shape each syllable, done 2026-10-06: a vibrato, and no envelope.**
   PSOLA held a syllable at one pitch. `tools/compare_singing_timing.py`,
   with `--variants slowed vibrato envelope shaped`, sings the shapes
   tried on the psola backend's timing: a vibrato as `note_with_vibrato`
   makes one, 0.35 semitones each way at 5.5 Hz, setting in a quarter of a
   second into a note and growing to full over 0.3 s, so a short note has
   none; `adsr`'s attack, decay to 3 dB down and release, of 20, 150 and
   60 ms; and both. Praat reads a pitch tier in the syllable as said,
   before the stretch, so the vibrato is drawn through the stretch's
   inverse, and keeps its rate on a vowel held twenty times its length:
   measured, 5.6 Hz and 36 cents each way, centred on the note to a cent.
   Whisper heard no difference with the vibrato, and fewer words with the
   envelope, 5 of 48 on quarter-second notes against 16, its attack
   covering the consonants. Heard on 2026-10-06: the vibrato kept, and the
   envelope, no difference to the ear, left out for what it cost the
   words. The psola backend sings every syllable with the vibrato.
4. **Accents for PSOLA, done 2026-10-06,** the rest of what `sing` does
   with eCantorix, whose effects the psola backend sings too (done
   2026-10-01). abc2midi 5.03 gives the first note of the score `sing`
   writes, which has no bar lines, a velocity of 105, a note on a strong
   beat 95 and any other 80; a strong beat comes every three of the
   meter's beats when they divide by three, every two when they divide by
   two, and once a bar otherwise, rather than on every beat as its
   documentation's `%%MIDI beat` suggests. eCantorix has espeak say each
   syllable at amplitude velocity / 127 * 200, which espeak's level
   follows to a twentieth of a decibel, so the psola backend sings each
   note at 165, 150 or 126 parts of 165, up to 2.34 dB apart. A test reads
   abc2midi's own MIDI in sixteen meters and finds the same accents.
5. **Own the PSOLA.** praat-parselmouth is a compiled dependency; a
   pitch-synchronous overlap-add in numpy would remove it, and belongs in
   a package about discrete-time synthesis. Worth it only if it sounds as
   good.
6. **Sing vowels from their spectra**, as issue #5 proposes: a glottal
   source through formant filters, with exact pitch and no speech engine,
   weak on consonants, so beside a speech engine rather than instead of
   it.

## Released: 1.10.0, 2026-10-06

The singing work and the corrections since 1.9.0 went out as 1.10.0, a
minor release because some calls now return something different and
`sing` sings with the package's own singer; its changelog section opens
with a note for anyone upgrading. Zenodo archived it as
10.5281/zenodo.23185652 within seconds of the release event, and the
record carries the nineteen controlled subjects, the fifty keywords and
the release notes, verified through its DataCite export.

## Open after the September 2026 audits

What the audits found and did not fix, and why. Each is recorded where
the behaviour is, too; this is the list in one place.

- **eCantorix's defects are fixed in the fork, and offered upstream.**
  `ttm/ecantorix` at `music-2` reads its control files on Perl 5.26 and
  later, runs with the `perl` on PATH, finds espeak's data where espeak
  says it is, reports a failed render, and keeps flite's renders out of
  espeak's cache; `setup_engine` clones that tag. divVerent/ecantorix#11
  offers the general fixes to the original, which still has them. `sing`
  keeps its workarounds, because an engine cloned before the pin stays
  what it was until its directory is removed.
- **Singing is verified on macOS and in CI, not everywhere.** Both
  backends were run and measured on macOS, and the CI job sings with both
  on Ubuntu, with every effect. The `melt` effect has not been tried with espeak-ng, whose
  data directory is laid out differently, and the engine's other extra
  voices (`poly`, `rubberband`, `mb-en1`, the last needing mbrola) are not
  offered through `effect`.
- **eCantorix's flite effect ignores the melody in flite's rms voice,**
  the one its own flite example sings with: every note comes out near
  90 Hz, 6.5 semitones flat at C3 and 18.5 at C4, measured 2026-10-01.
  In kal and slt it sings in tune, and the psola backend's flite effect
  sings in tune in all three. Not fixed: it is the engine's, in the
  fork, and would need its own measurement of how rms takes a pitch.
- **The two backends are compared by pitch, length and time, not by
  ear.** `tools/compare_singing.py` measures what can be measured. On
  2026-09-30 eCantorix sang every note within five cents on macOS and ten
  on Ubuntu, in CI, and PSOLA within three on both; both lines were as
  long as their scores. PSOLA
  rendered each score in about a quarter of a second; eCantorix took from
  under a second, where its cache already held the syllables, to half a
  minute for a voice it had not met. Which sounds better, and which is
  easier to understand, is for listening to the WAVs it writes.
- **`Being.walk(method='straight')` does not wrap** where every other
  window does. A caller that walks off the grid gets an `IndexError`;
  wrapping would change what an existing caller gets. `stay` was
  changed, because what it read was not the window at all.
- **`Being`'s `intensity_octaves` may stop one short.** Its comment says
  its steps run from 10 dB to half a decibel; `range(1, 20)` stops at
  10/19 dB. Whether 20 was meant is not clear enough to change a table
  someone may read.
- **`print_peal` runs two-digit bells together.** From ten bells on, a
  row such as `1011` could be four bells or two. Ringers write 0, E and T
  for ten to twelve; changing what the function prints was not done
  without someone who reads its output asking for it.
- **`interval("dim1")` is -1**, as the article's rule gives it, where the
  routines that measure upward refuse a negative interval.
- **`PlainChanges` does not read `hunts`.** It now warns. Laying hunts out
  another way needs a check that the peal still comes round, which was
  not written.
- **The legacy demonstration piece and the rest of `Being`'s tables are
  not pinned by tests.** A test of the piece would pin a performance, not
  a correctness; the spectra, the extended scales and the intensities
  have nothing to be checked against but themselves. `MUTATION_AUDIT.md` counts them as content.
- **The mutation audits are bounded and are not run in CI.** Each takes
  from seconds to ten minutes, measures its own sources against its own
  selection, and misses what mutmut cannot reach: decorated properties,
  work done at import, and on macOS any path that differs only in case.

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
