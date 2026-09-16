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
  are samples for them; one- and two-sample outer fades render silence.
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

## Completed: measure whether the tests detect wrong answers

The targeted mutation audit covers normalization, export and session
envelopes. It added 68 test cases and strengthened existing assertions;
the final selected tests detect 698 of 767 mutations, up from 617.
The 69 survivors were reviewed and accepted with reasons. No additional
production defect was found. [MUTATION_AUDIT.md](MUTATION_AUDIT.md) records
the reproduction command, tool limitation, runtime and survivor IDs.

## Next: make release verification complete

The highest-priority remaining maintenance is to put the manual release
checks into the release workflow: run examples and an installed-wheel
smoke check outside the checkout, and explicitly account for article,
MASS and vocabulary checks that need external resources. This release's
manual article/MASS/vocabulary checks reproduced 46/47 labelled equations
(all 46 testable), 26 exact/5 divergent/4 broken reference routines, and
17/19 confirmed subjects (two EuroSciVoc identifiers remain unconfirmable).
The 13 runnable examples passed; the external singing example skipped.

Broader mutation testing remains
[issue #113](https://github.com/ttm/music/issues/113). Expand one area at a
time when changing it; oscillator timing or envelopes are useful next
targets. The completed audit is not a whole-package mutation score.

## Other maintenance

- **Completed:** correct stale assessment and roadmap prose: interval
  naming exists;
  article coverage is 46/47; mypy checked 47 files; 13 examples ran.
  The assessment now describes the selected figures that are checked;
  expanding those checks remains optional follow-up work.
- **Completed:** refresh the local editable installation without changing
  dependencies. Its old `1.3.0.dist-info` made current code report 1.3.0
  outside the repository; it now reports 1.8.0 from either location.
  Published artifacts were unaffected.
- Consider macOS/Windows CI alongside the existing Ubuntu
  Python 3.10–3.14 matrix. The wheel check was run manually for this patch;
  automating it is part of the release-workflow task above.
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
