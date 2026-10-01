# Mutation audits

Line and branch coverage say a test *reached* a line. Mutation testing asks
the question directly: change the code, one edit at a time, and see whether
anything goes red. A surviving mutant is a line the suite runs without
depending on, and neither coverage measure can see one.

These are **bounded audits**, one area at a time, not a whole-package score.
The cost of a run is the test selection multiplied by the mutants, and the
output worth reading is the list of survivors rather than the percentage.
[Issue #113](https://github.com/ttm/music/issues/113) tracks the broader
work; `tools/mutation_audit.py --list-areas` lists what has been done.

| Area | Sources | Measured | Mutations | Detected | Surviving | Reviewed |
|---|---|---|---:|---:|---:|---|
| `export` | `core/functions.py`, `core/io.py`, `stimulation/session.py` | 2026-09-16 | 767 | 698 | 69 | all |
| `envelopes` | `synths/envelopes.py`, `filters/adsr.py`, `filters/fade.py` | 2026-09-21 | 561 | 526 | 35 | all |
| `oscillators` | `synths/notes.py` | 2026-09-23 | 1462 | 1444 | 18 | all |
| `stimuli` | `stimulation/stimuli.py` | 2026-09-23 | 427 | 423 | 4 | all |
| `localization` | `filters/localization.py` | 2026-09-23 | 652 | 631 | 21 | all |
| `utils` | `utils.py` | 2026-09-25 | 915 | 825 | 90 | all |
| `filters` | `filters/design.py`, `impulse_response.py`, `loud.py`, `reverb.py`, `stretches.py` | 2026-09-30 | 527 | 501 | 26 | all |
| `theory` | `theory/chords.py`, `intervals.py`, `scales.py` | 2026-09-27 | 214 | 213 | 1 | all |
| `structures` | `structures/permutations.py`, `peals/base.py`, `peals.py`, `plain_changes.py` | 2026-09-27 | 684 | 669 | 15 | all |
| `noises` | `synths/noises.py` | 2026-09-28 | 285 | 280 | 5 | all |
| `sequencer` | `sequencer.py` | 2026-09-28 | 157 | 157 | 0 | all |
| `bonds` | `bonds.py` | 2026-09-28 | 106 | 103 | 3 | all |
| `tables` | `tables.py` | 2026-09-28 | 77 | 77 | 0 | all |
| `hrtf` | `hrtf.py` | 2026-09-28 | 224 | 217 | 7 | all |
| `singing` | `singing/bootstrap.py`, `paths.py`, `perform.py`, `psola.py` | 2026-10-01 | 1137 | 1126 | 11 | all |
| `legacy` | `legacy/CanonicalSynth.py`, `IteratorSynth.py`, `classes.py`, `tables.py`, `pieces/testSong2.py` | 2026-09-30 | 1645 | 1154 | 491 | all |

Each area names the files it mutates and the tests that judge them, and the
tests must cover every line and branch of those files between them, or
mutmut reports mutants no test reaches. All areas report none.

## `export` — normalization, export and session envelopes

Measured 2026-09-16 on Python 3.12.7, macOS, with `mutmut` 3.7.0.
The final input snapshot is `ef03962`; production code is unchanged from
the correctness patch in `896b493`.

### Result

The audit strengthened tests without finding another production defect.
It added 68 cases and tightened existing assertions. The resulting tests
detect 698 of 767 generated mutations (91.0%). All 69 survivors were
reviewed: 42 affect diagnostic/log text, 25 are equivalent for supported
public calls, and two change the default noise length by one sample.
None timed out, lacked tests, was skipped, crashed or remained untested.

| Module | Mutations | Initially detected | Finally detected | Surviving |
|---|---:|---:|---:|---:|
| `music/core/functions.py` | 150 | 118 | 130 | 20 |
| `music/core/io.py` | 274 | 197 | 235 | 39 |
| `music/stimulation/session.py` | 343 | 302 | 333 | 10 |
| **Total** | **767** | **617** | **698** | **69** |

The initial measurements used `896b493`, after the correctness patch and
its regression tests. Two passes were necessary because the first one
exposed only the session's free function; the table counts that shared
function once. The final pass combines all three files and also selects
two existing tests for quantizer clipping and empty-writer refusals.
Thus the 81 additional detections include the effect of selecting those
existing tests, as well as the new assertions.

### What the stronger tests catch

- Exporting an ADSR envelope instead of the supplied audio could preserve
  absolute levels and pass the previous fade test. It now checks the
  signed waveform, output length and quieter stereo channel too.
- Stereo export now checks the requested encoding and sample precision
  for every supported WAV/FLAC depth. Floating-point reads include DOUBLE
  WAV, as well as FLOAT.
- Normalization uses nonzero offsets, integer input and float32 input, so
  incorrect subtraction and lost floating-point conversion are visible.
  Writer tests check both normalization options and their defaults.
- Playback checks the samples and sampling rate passed to the device;
  no actual audio device is needed.
- Short session transitions use distinct levels and explicit expected
  sample sequences, including asymmetric competing ramps, odd/even
  rounding and array/callable mixtures. Constant unity on both sides
  could hide a misplaced transition.
- Sessions verify sampling-rate forwarding, float64 gain precision,
  zero-sample phases, normal two-sample closing fades and written PCM
  samples. Length-refusal diagnostics must report the actual counts.

### Accepted survivors

IDs below are the numeric suffix of mutmut's identifier for that function,
for example `music.core.functions.x_normalize_mono__mutmut_10`.
Session methods use the `StimulationSession` class prefix. IDs apply to
the recorded snapshot and tool version, not arbitrary later source edits.

#### Diagnostic/log text: 42

These change case, punctuation, explanatory text or logging. Refusals
still happen, and the tests check the useful error category rather than
every character of wording.

| Function | IDs |
|---|---|
| `functions.normalize_mono` | 10, 12–24 |
| `functions.normalize_stereo` | 11, 13–15 |
| `io._audio_format` | 5, 7 |
| `io._subtype` | 4, 6, 7 |
| `io._quantize` | 9–15 |
| `io.read_audio` | 4–9 |
| `io.play_audio` | 4 |
| `StimulationSession.add` | 15, 17, 18 |
| `StimulationSession.__repr__` | 7, 17 |

#### Equivalent for supported public calls: 25

| Function | IDs | Why accepted |
|---|---|---|
| `functions._scaled` | 4, 6 | Public callers already supply float64 arrays. |
| `io._subtype` | 1, 2 | Public callers pass the format explicitly. |
| `io._wav_subtype` | 4 | Omitted WAV argument selects the same default. |
| `io._quantize` | 4, 6 | Public callers already supply float64 arrays. |
| `io._quantize` | 41 | Depth 17 is refused before the changed boundary. |
| `io.read_audio` | 13, 25 | Default reads are float64; dividing an empty array changes nothing. |
| `io.write_wav_mono` | 20, 23, 24 | The omitted uniform-noise bound has the same default; affine noise changes cancel in normalization, apart from floating-point rounding. |
| `io.write_wav_mono` | 49, 54 | SoundFile infers the same validated file extension. |
| `io.write_wav_stereo` | 21, 24, 25 | The same default-bound and normalized-noise reasoning. |
| `io.write_wav_stereo` | 53, 58 | SoundFile infers the same validated file extension. |
| `session._ramp_shape` | 1 | Both zero-count paths produce an empty array. |
| `StimulationSession._layout` | 83 | With positive falling cost, proportional rising allocation is already at most capacity minus one; the changed upper clamp cannot bind. |
| `StimulationSession.render` | 55 | Both `False` and `None` select the falling ramp. |
| `StimulationSession.render` | 61, 62 | NumPy broadcasts mono into stereo; converting an already stereo input leaves it unchanged. |

#### Default noise length: 2

`io.write_wav_mono` 25 and `io.write_wav_stereo` 27 change the generated
default noise from 100,000 to 100,001 samples. The documentation promises
approximately two seconds. Tests check that real noise is rendered with
the documented channels and sampling rate; they do not freeze that
arbitrary length. These are observable changes, accepted explicitly.

## `envelopes` — the note-level amplitude envelopes

Measured 2026-09-20 on Python 3.12.7, macOS, with `mutmut` 3.7.0, on the
tree this file is committed with. No adapter: these three modules are plain
functions, so the decorated-class limitation below does not apply to them.

### Result

Unlike the first audit, this one **found three defects in the package**,
each in a documented parameter that no test depended on.

| Module | Mutations | Detected | Surviving |
|---|---:|---:|---:|
| `music/core/synths/envelopes.py` | 151 | 145 | 6 |
| `music/core/filters/adsr.py` | 188 | 178 | 10 |
| `music/core/filters/fade.py` | 222 | 203 | 19 |
| **Total** | **561** | **526** | **35** |

The first run, against the code as it stood, killed 439 of 550 with 110
survivors and one timeout. The totals differ because the corrections
changed the code being mutated: the three fixes added lines, and the
`tremolo` branch later collapsed into a call to the shared
`_signed_power` the oscillator area needed, removing nine mutants without
changing a sample. Nothing timed out, was skipped, crashed or went
untested in any run.

| Function | Mutations | Detected | Surviving |
|---|---:|---:|---:|
| `am` | 32 | 31 | 1 |
| `tremolo` | 39 | 38 | 1 |
| `tremolos` | 80 | 76 | 4 |
| `adsr` | 111 | 107 | 4 |
| `adsr_vibrato` | 3 | 3 | 0 |
| `adsr_stereo` | 74 | 68 | 6 |
| `fade` | 135 | 125 | 10 |
| `cross_fade` | 87 | 78 | 9 |

`adsr_vibrato` killed **none** of its three mutants to begin with. Its
whole body could be replaced by `adsr(**adsr_dict)` — dropping the note it
exists to render — and the suite stayed green, including its own doctest,
which asked only for `ndim == 1`.

### What it found

**A distorted tremolo was not a real number.** `tremolo` raised the signed
oscillation straight to `alpha`, so `tremolo(alpha=0.5)` was NaN wherever
the waveform was negative — half of every envelope, behind a NumPy warning
— and an even `alpha` rectified the pattern into one that could only boost.
Seven mutants of that branch survived, including one that swapped it with
its own `else`: the branch ran, and nothing looked at what came out. The
index now applies to the magnitude with the sign kept. `DISCREPANCIES.md`
has the table; the article gives the tremolo no `alpha` at all.

**`adsr` never reached zero.** `to_zero` is a duration in milliseconds and
reached `fade` as a bare ratio where `fade` reads a percentage, a hundred
times too small: below about 2.3 ms at 44.1 kHz it rounded to no samples,
so `adsr(to_zero=1)` was byte-identical to `adsr(to_zero=0)`. The mutant
that changed the default from 1 to 2 survived because the parameter did
nothing. The envelope now departs from zero and returns to it, and `AD`
and `ADS` are divergent rows of `RECONCILIATION.md` instead of exact ones,
because the reference carries the same line.

**`cross_fade` placed the overlap at the wrong rate.** The fades were cut
at `sample_rate` while `mix_with_offset` used its own default of 44.1 kHz,
so at any other rate the two sounds met at full level: a steady 3 and a
steady 5 crossfaded at 8 kHz summed to 8. No mutant found this one — an
argument that is absent cannot be mutated. It came out of writing the test
that kills the mutants around it, which is the other thing an audit is for.
A zero or oversized `duration` also reached NumPy as a broadcast failure
rather than a refusal; both now raise `ValueError` naming the duration.

The same signed `** alpha` sits in `note_with_vibrato`,
`note_with_two_vibratos`, `note_with_glissando_vibrato` and
`note_with_two_vibratos_glissando` (`notes.py` lines 559, 563, 1006, 1007,
1239, 1344 and 1345). There it is worse: the NaN frequencies reach an
`int64` cast of the accumulated phase, which turns them into `INT64_MIN`,
so the output is finite and wrong rather than visibly NaN. That is the
oscillator area, and it is left for the audit that covers it rather than
corrected blind here.

### What the stronger tests catch

- Every default in all eight signatures. Nothing called `am()`,
  `tremolo()`, `tremolos()`, `adsr()`, `adsr_stereo()`, `fade()` or
  `cross_fade()` bare and looked at the result, so the durations, rates,
  depths and frequencies they document were free to change. The tests now
  pin the length, the oscillation period and the depth.
- The settings these routines pass on. `tremolos` forwards five arguments
  to `tremolo` down two branches, `adsr` hands `transition`, `alpha` and
  `db_dev` to three separate calls, and `adsr_stereo` forwards eight to
  `adsr` twice; each could be dropped unnoticed. They are now compared
  against the routine they delegate to, called directly.
- That a fade stays inside `[0, 1]` and moves in one direction. Scaling
  the linear tail by the reciprocal of its junction made an envelope leap
  to ten thousand, and no test saw it.
- That a crossfade between a steady 3 and a steady 5 is never quieter than
  3 nor louder than 5 — which is what distinguishes it from a sum, and
  what the level-1 signals the old test used could not tell apart.
- That `to_zero` buys the milliseconds of straight line it names, at both
  ends, and that an odd distortion index is still the plain power.

### Accepted survivors

All 35 were reviewed. None changes what a supported call returns.

| Kind | Count | Mutants |
|---|---:|---|
| `sonic_vector=0` → `1`, which `as_sonic_vector` maps alike | 6 | `am` 5, `tremolo` 6, `tremolos` 2, `adsr` 10, `adsr_stereo` 10, `fade` 8 |
| `sonic_vector1/2 = 0` → `None`/`1`, the same mapping | 4 | `adsr_stereo` 19–22 |
| `"exp"` → `"XXexpXX"` and `"linear"` → `"XXlinearXX"`: `fade` and `loud` dispatch on the substring, so the padded name still selects the same branch | 8 | `adsr` 5, `adsr_stereo` 5, `fade` 3, 53, 65, 99, 128, `cross_fade` 2 |
| A falsy sentinel replaced by another: `to=0` → `None`, `fade_out=0` → `None` | 5 | `adsr` 51, `fade` 58, 106, 121, `cross_fade` 70 |
| `sample_rate` passed to `fade` beside a `number_of_samples`, which makes it unread | 4 | `cross_fade` 61, 64, 69, 73 |
| A boundary that cannot bind: `stages >= lambda_adsr` scales by exactly 1.0; `amax = 1` when every tremolo has a sample; `<` → `<=` pads by nothing | 4 | `adsr` 38, `tremolos` 48, 60, 69 |
| Diagnostic wording. The refusal still happens and the tests check the category, not the characters | 4 | `fade` 16, `cross_fade` 12, 13, 14 |

IDs are the numeric suffix of mutmut's identifier for that function, for
example `music.core.filters.fade.x_fade__mutmut_16`, and apply to this
snapshot and tool version rather than to arbitrary later edits.

## `oscillators` — the notes, their vibratos and their glissandi

Measured 2026-09-23 on Python 3.12.7, macOS, with `mutmut` 3.7.0. No
adapter. The selection is now 35 test files: the previous 29 plus three
dedicated to sequential localization, two for the unlocalized sequence
and Doppler oscillator, and one for the glissandi, trills and defaults.
Together they reach every line and branch of `notes.py`.

**The area is closed, with all 18 survivors reviewed and accepted.** The
first two passes found the vibrato defects below. The third reviewed
all 80 survivors in `note_with_vibrato_seq_localization`, found the timing,
gain and input defects described below, and left only two accepted mutants.
The next pass closed the unlocalized sequence and Doppler targets below,
and the last one the 42 survivors in the remaining routines.

### Result

| | Mutations | Detected | Surviving |
|---|---:|---:|---:|
| First run | 1375 | 1124 | 251 |
| After the latching correction | 1396 | 1165 | 231 |
| After closing the vibrato routines | 1402 | 1248 | 154 |
| After closing sequential localization | 1426 | 1350 | 76 |
| After closing the unlocalized sequence and Doppler | 1439 | 1394 | 45 |
| After closing the remaining routines | 1460 | 1442 | 18 |
| After the second review's `trill` refusal | 1462 | 1444 | 18 |

Four mutants time out rather than returning a wrong answer in every run.
They are counted as detected: `trill` accumulates samples in a `while`
loop, and `pointer -= ns`, `i -= 1`, `i = 1` and a `*` turned `/` each
make that loop run forever. No suite that finishes accepts them, and the
runner no longer reports a run containing them as incomplete.

| Function | Mutations | Detected | Surviving | Was |
|---|---:|---:|---:|---:|
| `note_with_vibrato_seq_localization` | 523 | 521 | 2 | 81 |
| `note_with_vibratos_glissandos` | 151 | 151 | 0 | 22 |
| `note_with_doppler` | 182 | 181 | 1 | 14 |
| `_exponential_positions` | 29 | 20 | 9 | 8 |
| `_require_a_ratio` | 14 | 11 | 3 | 10 |
| `trill` | 52 | 50 | 2 | 6 |
| `_fit_to_samples` | 17 | 16 | 1 | 1 |
| `note_with_glissando` | 63 | 63 | 0 | 6 |
| `note_with_fm` | 39 | 39 | 0 | 4 |
| `note_with_phase` | 27 | 27 | 0 | 3 |
| `note` | 20 | 20 | 0 | 2 |
| `note_with_glissando_vibrato` | 86 | 86 | 0 | 25 |
| `note_with_two_vibratos_glissando` | 114 | 114 | 0 | 33 |
| `note_with_vibrato` | 57 | 57 | 0 | 13 |
| `note_with_two_vibratos` | 88 | 88 | 0 | 23 |

"Was" is the first run. The initial vibrato passes left 22 survivors across
the five unlocalized vibrato routines; the later sequence passes also
examine timing, timbre and spatial behavior. `_exponential_positions` has
more mutations than it had, because the last pass gave it a branch.

### What it found

**A fractional distortion index latched the render to full-scale DC.**
The same signed `** alpha` the envelope audit corrected in the tremolo sat
in the vibrato of five routines. There the distorted quantity is a
*frequency*, so the NaN did not stay visible: it flowed into the
accumulated phase and then into an `int64` cast, which turns NaN into
`INT64_MIN`, and modulo the table length that is one fixed index.
`note_with_vibrato(duration=0.2, vibrato_freq=5, alpha=0.5)` rendered
4,411 samples of a note and then 4,409 samples of constant −1.0. Every
sample finite, inside full scale, and meaningless.

Thirteen mutants of `note_with_vibrato` survived the first run, seven of
them arithmetic edits to the single line that computes the distorted
frequency, and one that swapped the branch with its own `else`. The branch
ran on every one of them. Nothing measured the pitch it produced, which is
exactly why the defect could sit there.

This is the defect `_require_a_ratio` already names, one parameter over —
"a negative frequency raised to a fractional power is NaN, which was then
cast to an integer table index and read out of the waveform table, so the
render came back finite, plausible and meaningless rather than failing".
That guard was written for the glissando endpoints and did not reach the
vibrato. `DISCREPANCIES.md` records what the article does and does not say.

**A sixth routine carried the same signed power, and a text search could
not see it.** `note_with_vibratos_glissandos` raises its vibrato pattern on
a line whose `**` ends one line and whose index begins the next, so the
`grep '\*\* *alpha'` that found the other five missed it, and the first
pass of this audit recorded five where there were six. The list is now from
an AST walk over every `Pow` node whose exponent names an index. There are
none left.

**The second vibrato read the first one's waveform table.** In
`note_with_two_vibratos` and `note_with_two_vibratos_glissando`, `tv2` was
looked up in `tabv1`, so `tabv2` and `sec_vibrato_waveform_table` were
accepted, documented, and used only for their length: a square second
vibrato under a sine first one gave back two sines. Two tables of different
lengths were worse — the modulus came from the second and the lookup from
the first, so the shorter was indexed past its end and raised `IndexError`.
The MASS reference has the same line; both reconciliation cases pass the
same table twice, so `VV` and `PVV` stay sample-exact.

**No mutant found either.** One was a name, not an arithmetic operator, and
swapping one valid table for another is not an edit mutmut makes; the other
was a line mutmut mutated freely but whose survivors read like the rest of
the distorted-path block. Both came out of writing the tests that kill the
mutants around them — the same way the `cross_fade` sample rate did. The
score measures the tests that exist against the code that exists, and
reading the code while writing them is where the rest comes from.

**A test can pass because of the defect it should catch.** The quadrant
test below named only the primary waveform table, and the table defect was
handing the same square wave to the second vibrato. It measured four clean
quadrants, passed, and broke the moment the defect was corrected.

### What the stronger tests catch

- The bent pitch itself. A square vibrato table holds each extreme for
  half the cycle, so the note is two steady tones and each one's frequency
  can be *measured* from its zero crossings rather than inferred. The
  tests match both halves against `freq · 2^(±(dev/12)^α)` to within a
  tenth of a percent, for four indices. That closed `note_with_vibrato`
  completely: 57 of 57.
- That no render latches, across all five routines that carry the index.
- That the two halves sit either side of the carrier, which is what an
  even index destroyed by pushing both the same way.
- That a glissando sweeps frequencies below one hertz. `_require_a_ratio`
  refuses an endpoint that is not positive, and nothing swept from or to
  a frequency in (0, 1], so the guard could have read `> 1` at either end.

### Sequential localization: completed 2026-09-23

The 80 survivors exposed tests that reached pitch curves, spatial paths and
mono gain without measuring what those paths produced. The new tests use
hand-calculated pitches, measured zero crossings, distinct carrier tables,
geometric distances and exact segment boundaries. They cover independent
vibrato lines, nonunit pitch-curve indices, mono/stereo motion, temperature,
initial interaural delay, and the state held after each control line ends.

The pass corrected these concrete failures:

- Vibrato segments rounded fractional sample counts up while pitch and
  movement rounded down. At 1 kHz, 2.5 ms and 5.5 ms vibratos occupied
  nine samples instead of seven, shifting their boundary and the final
  unmodulated tail. All three controls now use integer sample counts.
- A one-sample pitch glide divided by zero and poisoned the accumulated
  phase for the rest of the note. Positive curve indices now select its
  starting frequency; a zero index retains its immediate jump to the end.
- A finished path held gain one sample before its destination. A ten-sample
  move from `(0.3, 0.4)` to `(0.6, 0.8)` held a distance of 0.95 m instead
  of 1 m, leaving the rest of the mono note 5.26% too loud. The held gain
  now uses the exact final distance, per ear in stereo.
- Nonpositive pitch endpoints and movement durations below one sample
  reached undefined ratios or velocities. They now raise `ValueError`.
- Documented list/tuple waveform tables could not be indexed by arrays,
  and integer carrier tables could not receive floating-point gain. Tables
  are converted to floating-point arrays without modifying caller inputs.

Twenty regression cases failed against the original source before these
fixes. The final tests add array-like inputs and narrow cases identified by
the preliminary mutation run, including a two-sample glide and a curved
glide whose starting frequency is not one. The MASS `D_` comparison retains
its original fixture and tight phase bound while explicitly accounting for
the corrected counts and final gain; see `RECONCILIATION.md`.

The function now detects **521 of 523 mutants**. Both survivors are accepted:

| ID | Reason |
|---|---|
| 21 | Adds `XX` around the movement-duration error text; the same invalid input is refused with the same exception and useful diagnostic. |
| 498 | Changes `x[0] > 0` to `x[0] >= 0`; at zero both ears are equidistant and the initial delay is zero, so either padding branch returns the same samples. |

IDs use the `music.core.synths.notes.x_note_with_vibrato_seq_localization`
prefix. The final run used the working-tree changes on `bbb4d83`, with
`notes.py` SHA-256
`7015dfad1fd11f6bd258ebb531318f815a0b4d5a4a5e3dd2657c8ddd4d0243d0`.
The scratch audit records the base commit and per-file hashes; it is not an
audit of unchanged `bbb4d83`. The standard reproduction command below
archives committed files, so these changes must be committed before using
`--revision HEAD` to reproduce them. No repository commit was needed for
the scratch measurement.

That full area run took 397 seconds with two workers: 1,346 killed, four
timeouts and 76 survivors. Nothing was untested, skipped, suspicious or
interrupted. The two survivors above are the only ones in this function;
18 in `note_with_vibratos_glissandos`, 14 in `note_with_doppler` and 42
elsewhere remained outside that completed pass.

### Unlocalized sequence and Doppler: completed 2026-09-23

The next pass examined all 18 survivors in
`note_with_vibratos_glissandos` and all 14 in `note_with_doppler`.
The new tests measure curved pitch glides, independent modulation lines,
fractional segment boundaries, distinct carrier tables and the pitch held
when a control sequence finishes. Twelve sequence cases failed against
the previous implementation:

- A one-sample pitch transition divided by zero, corrupting the phase of
  subsequent samples. Positive curve indices now use the starting pitch;
  a zero index retains the immediate jump to the endpoint.
- Fractional vibrato durations rounded up, while pitch durations rounded
  down. Both now use the integer sample count for each segment.
- Nonpositive endpoints reached undefined exponential ratios. Each pitch
  segment now uses the same positive-endpoint guard as other glissandi.
- Nested list and tuple tables failed at indexed lookup. Converting each
  table with `np.asarray` supports those inputs and preserves array dtypes.

The `PV_` MASS case consequently changes from 1,590 to 1,586 samples.
Its original fixture is preserved. The comparator checks the complete
render against explicit floored durations and separately restores the
reference's vibrato counts, retaining the previous rare-table-step bound.
Tests reject corrupt original tails and supplementary renders as well as
additional timing, pitch, gain and nonfinite differences.

The Doppler tests found no production defect. They measure source defaults,
empty and single-sample renders, inverse-distance gain on diagonal paths,
temperature-dependent pitch at each ear and the initial interaural delay.
A phase ramp exposes the received frequency for comparison with radial
velocity computed independently from the path. Reflections of the path and
coincident ears additionally check whole-waveform symmetry.

The unlocalized sequence now detects **151 of 151 mutants**. Doppler
detects **181 of 182**; its only survivor is accepted:

| ID | Reason |
|---|---|
| 128 | Changes `x[0] > 0` to `x[0] >= 0`; a centered source has equal ear distances and zero initial delay, so both branches return the same samples. |

This ID uses the `music.core.synths.notes.x_note_with_doppler` prefix.
The final snapshot uses the working-tree changes on `c01827f`, with
`notes.py` SHA-256
`6088967074493c849398fdcfeb57dae60a75c460f9e7d0570b85a439ff9c5c7e`.
The scratch report records every Python input's hash and all 34 selected
test files; source and test hashes were checked against the final checkout.
The standard command below reproduces the audit once these changes are
committed.

The full area run took 334 seconds with two workers: **1,390 killed, four
known `trill` timeouts and 45 survivors**. Nothing was untested, skipped,
suspicious or interrupted. The two accepted localized-sequence survivors
and Doppler's one are unchanged in meaning; the other 42 survivors are
outside these completed passes.

### Remaining routines: completed 2026-09-23

The last pass examined the other 42 survivors. Sorted by what they
changed rather than where: 17 edited refusal text, 14 changed a default
argument, four could not change behavior and seven exposed an unasserted
behavior. The groups did not follow the function names. None of
`note_with_glissando`'s six touched its pitch curve; the pitch-curve
survivors were in the two glissando-vibrato routines.

It found three production defects. Seventeen regression cases failed
against the previous source:

- **`trill` timed its envelope at 44,100 Hz whatever its rate.** It passed
  `sample_rate` to `note` but not to `adsr`, whose durations are
  milliseconds, so at 8 kHz each attack, decay and release lasted 5.5
  times as long. No mutant can remove an argument that is not there. This
  one came from reading around the mutant that removed the rate from the
  `note` call, which survived because nothing ran a trill at another
  rate: the same way the `cross_fade` sample rate was found.
- **A one-sample glissando divided zero by zero** in `note_with_glissando`,
  `note_with_glissando_vibrato` and `note_with_two_vibratos_glissando`.
  The NaN frequency became `INT64_MIN` as a table index, so the sample
  read an arbitrary entry behind a `RuntimeWarning`. It now sounds the
  starting frequency, or the end for a zero index, as the sequences do.
- **An exponential path refused a coordinate held on an axis.** Each
  coordinate moves by its own ratio, and a source straight ahead keeps
  `x = 0`, so it was refused as a path that crosses the listener. An
  unchanged coordinate is now held, and the refusal of a coordinate
  that reaches or crosses zero says so.

The tests now measure:

- Where a glissando lands. The sweep's exponent is `samples / (n - 1)`,
  and an edit to that denominator moves the end of a second-long sweep
  by thousandths of a cent. Five samples at 64 Hz through a ramp table
  make each sample's integrated phase exact, for three curve indices.
- A trill's pitch and envelope timing at 8 kHz, and trills of one note a
  second or fewer. Only a rate of zero had been refused in a test, so the
  guard could have read `<= 1`.
- A single zero endpoint refused at either end and on either axis. Only
  opposite signs had been tested.
- A resting source on an exponential path. The scratch run left one
  survivor in the new branch, which returned a held coordinate as one
  value: that broadcasts invisibly while the other coordinate moves.
- Which refusals suggest `method="lin"`. Only `note_with_glissando`
  offers it, and the other two end with the values they were given.
- That bare calls of `note`, `note_with_phase`, `note_with_fm`,
  `note_with_glissando` and `trill` equal calls with their declared
  defaults, as the vibrato routines' already did.

Fifteen new survivors are accepted, with the three from earlier passes:

| Function | IDs | Reason |
|---|---|---|
| `_exponential_positions` | 14, 17 | `<` to `<=` in the sign test. A zero at either end is refused before it, so both comparisons agree wherever they run. |
| `_exponential_positions` | 20–26 | Case changes or `XX` around the refusal text. The same inputs are refused with the same exception. |
| `_require_a_ratio` | 8, 13, 14 | Case changes or `XX` around the refusal text. The tests check the hint's presence and the values the message ends with. |
| `_fit_to_samples` | 5 | `<=` to `<` at an equal length. The full slice and a zero-length pad return the same samples. |
| `trill` | 7 | `XX` around the refusal text. |
| `trill` | 34 | Drops `waveform_table=WAVEFORM_TRIANGULAR` from the `note` call. That is `note`'s default. |

IDs use the `music.core.synths.notes.x_<function>__mutmut_` prefix. The
final snapshot uses the working-tree changes on `a44c81b`, with
`notes.py` SHA-256
`b31fac556994e3d8cd12ef72848437fdabc78bdebb90f22d6729a677c269d9be`.
The scratch report records every Python input's hash and all 35
selected test files. The standard command below reproduces the audit
once these changes are committed.

The full area run took 582 seconds with two workers: **1,438 killed,
four known `trill` timeouts and 18 survivors**, every one accepted above
or in an earlier pass. Nothing was untested, skipped, suspicious or
interrupted.

The second review after 1.8.3 made `trill` refuse notes shorter than a
sample. A rerun on that source, `notes.py` SHA-256
`8e28e37b8708851c22df243dd0617901aebd17d9085c033f4ea2fab4eb7c8597`,
detects **1,444 of 1,462** with the same eighteen survivors; the `trill`
line deletion that was ID 32 is now 34. It took 357 seconds.

## `stimuli` — the sensory-stimulation generators

Measured 2026-09-23 on Python 3.12.7, macOS, with `mutmut` 3.7.0. No
adapter. The seven selected test files are the six that
`pytest --cov-context=test` found reaching `stimuli.py`, plus the one this
audit added. Together they reach every line and branch of it.

### Result

| | Mutations | Detected | Surviving |
|---|---:|---:|---:|
| First run | 404 | 329 | 75 |
| After the tests and corrections below | 427 | 423 | 4 |

The totals differ because the corrections added refusals and removed
two redundancies. Nothing timed out, lacked tests, was skipped or was
suspicious.

The 75 first-run survivors show that the module was tested for its shapes
and spectra, not its samples. Every arithmetic edit to the amplitude
envelope survived, since the tests only asked where the envelope's energy
sat. So did most edits to the orbit's phase, fold and azimuth, the
isochronic ramp's shape, and the sign of the frequency deviation.
`modulated_noise` could drop `min_freq`, `max_freq` or `sample_rate` on
the way to `noise`. No test ran any stimulus at a rate other than
44.1 kHz, so every call passing the rate to `note` could have dropped it.
Twenty-seven survivors were changed defaults.

### What it found

Ten regression cases failed against the previous source:

- **An isochronic train at a zero rate raised `ZeroDivisionError`** once
  it had a ramp, which divides by the pulse period. A negative rate ran
  the gate backwards, starting each period silent. `pulse_rate` must now
  be positive.
- **A zero modulation rate meant different things in siblings.**
  `modulated_noise` documents zero as unmodulated and returns the noise
  at full level. `amplitude_modulation` held its modulator at the table's
  first entry and scaled the carrier by it, which is half, for the default
  sine at full depth. `frequency_modulation` shifted its pitch the same way
  for any table not starting at zero. Both now leave the carrier alone.
  The survivors that refused a zero or sub-hertz rate prompted the
  decision, since a test of zero needs an answer to test against.
- **Negative modulation rates were refused only by `modulated_noise`.**
  `amplitude_modulation` and `frequency_modulation` now refuse them too,
  for the reason its docstring gives.
- **`spatial_motion` accepted a stereo sound** and failed inside the
  localization with a message about broadcasting. It now says what is
  wrong. A zero duration no longer renders two seconds of tone first.
- **The `float64` conversion of a moved sound is load-bearing.** Its
  mutants looked equivalent. Without it, a `float32` sound interpolates
  its fractional delay in single precision, and a boolean one raises.

Two redundancies went: `_nothing` had a default no caller used, and
`modulated_noise` converted `noise`'s output to the `float64` it already
was.

### What the tests catch

`tests/test_stimuli_audit.py` reads samples rather than spectra. A ramp
table renders a carrier's phase, and a constant table renders an envelope
or a gate by itself:

- Both binaural carriers, the monaural mean, and the gated, modulated and
  orbiting carriers at 8 kHz.
- The amplitude envelope `1 - depth * (1 - m) / 2` at crest, zero and
  trough for three depths, and the same envelope on a seeded noise bed.
- Unmodulated noise equal to `noise` with the same colour, band and rate.
- A frequency sweep that rises while the modulator is up.
- The isochronic gate to the sample, its edge included, with a duty
  cycle of one and with a linear ramp from each edge.
- The orbit against independently computed azimuths through two and a
  half triangular cycles, for two pairs of endpoints.
- One-sample and zero-length renders, `number_of_samples` for the noise
  and the orbit, zero and sub-hertz rates, and each routine's declared
  defaults.

### Accepted survivors

| Function | IDs | Reason |
|---|---|---|
| `isochronic_tones` | 48 | `.astype(None)` for `.astype(np.float64)`: NumPy's default type is `float64`. |
| `isochronic_tones` | 65 | `gate =` for `gate *=` in the ramp. The ramp is already zero wherever the gate is closed and the gate is one wherever it is open. |
| `isochronic_tones` | 68 | Drops the ramp's lower clip. Its values are negative only where the gate is closed, where the product is a zero of the other sign. |
| `spatial_motion` | 43 | `XX` around the refusal text. |

The two ramp mutants agreed with the source in all of 3,000 random
settings of rate, duty cycle, ramp, length and sample rate.

IDs use the `music.stimulation.stimuli.x_<function>__mutmut_` prefix.
The audit's own snapshot used the working-tree changes on `95cf57a`. A
later review corrected one docstring, and a rerun on that source,
`stimuli.py` SHA-256
`872b06e9fdb415ef01863d4617768118a4111e9a64cf887faa4b564d41da0763`,
detects the same 423 of 427 with the same four survivors. The first run
took 60 seconds with two workers.

## `localization` — interaural cues, fixed, per-frequency, moving and convolved

Measured 2026-09-23 on Python 3.12.7, macOS, with `mutmut` 3.7.0. No
adapter. The eighteen selected test files are the seventeen that
`pytest --cov-context=test` found reaching `localization.py`, plus the
one this audit added. Together they reach every line and branch of it.

### Result

| | Mutations | Detected | Surviving |
|---|---:|---:|---:|
| First run | 611 | 526 | 85 |
| After the tests and corrections below | 652 | 631 | 21 |

One first-run mutant timed out and is counted as detected. Nothing lacked
tests, was skipped or was suspicious.

Fifty-four of the 85 first-run survivors were in `localize2`, and its
`brute` method was barely measured. Replacing its accumulation `s += s_`
with `s = s_` survived. So did its amplitude, the arguments of its
resynthesis, its energy cutoff and its buffer size. The `ifft` method's
gain formula, its high-frequency delay coefficient and its 4 kHz boundary
went unmeasured too. `localize` was never checked for a source on the
left, or for an angle at a distance other than one. Nothing checked that
`localize_linear` reaches `theta2` at its last sample.

### What it found

Thirty-one regression cases failed against the previous source:

- **`brute` resynthesized every partial a quarter cycle early.** The
  FFT's angles are a cosine's, and it passed them as the phase of a sine
  table: a sine came back as minus a cosine, a sound of several partials
  as a different waveform. This is the "not giving good results for all
  sounds" the docstring admitted to. The MASS reference has the same line,
  but its `brute` raises `TypeError` before reaching it.
- **`brute`'s buffer ran some thirty samples past any delay.** It was
  sized without `zeta`, and by a coefficient chosen from the highest bin
  kept. Only the missing factor kept that choice from producing a buffer
  shorter than the delays it had to hold.
- **The far ear heard the first sample before the sound arrived.** The
  fractional delay read the nearest sample outside the signal, so the far
  ear held the first one for the whole interaural delay. A click at the
  start reached it as a plateau some 27 samples long. `localize` pads
  with silence, and now the delay line does too.
- **A source on an ear returned NaN.** At zero distance the intensity
  ratio was 0 / 0. Its limit is the near ear in full and the far one
  silent, which `localize` already rendered. `localize` itself divided by
  zero for a source on the left ear, and warned while getting it right.
- **`localize` rejected a list**, although it is documented as
  array_like.

Reading the module also corrected four docstrings: the speed of sound
(331.3, not 331.2), a diffraction delay of 0.7 ms rather than 0.7 s,
`localize_hrtf` claiming to use a `sample_rate` it ignores, and what a
zero angle means to `localize` and `localize2`.

**Then changed: a zero angle is an angle.** Both routines read `theta=0`
as "use `x` and `y`", so a source could not be placed at zero degrees:
`localize` put it at its default position and `localize2` far to one
side. Because `localize2`'s default angle is -70, passing zero was how
its callers gave a position, and existing tests did exactly that. The
sentinel is now `None`, the change is in the changelog's note for anyone
upgrading, and `localize2`'s docstring now says that it measures from
straight ahead where the others measure from the ear axis.

### What the tests catch

`tests/test_localization_audit.py` computes the geometry independently
and compares samples:

- The fractional delay against the textbook Catmull-Rom spline through a
  signal extended by silence, at every sample including both ends, and
  exactly on a parabola, where the spline is exact.
- The far ear's delay and gain on a ramp at three temperatures, and
  `localize`'s delays and gains on both sides, at an angle and distance,
  and truncated to whole samples either side of an integer.
- `localize_linear` against positions computed from its endpoints, for
  two, three and fifty samples, and for `float32` and boolean input.
- `localize2`'s `ifft` method on exact tones at the lowest bin, either
  side of 4 kHz and above a third of the spectrum, for both sides.
- `brute` on tones of three phases, at two sample rates, with the gain
  between the ears exact and the partials it keeps or drops by energy.
- A source exactly on either ear, and which ear an empty impulse response
  error names.

### Accepted survivors

| Function | IDs | Reason |
|---|---|---|
| `localize2` | 14, 15, 160, 161 | Case changes or `XX` around the refusal and the warning text. |
| `localize2` | 146, 222, 232 | `theta_ > 0` to `>=`. At zero the delay is zero and the gain one, so both branches render the same samples. |
| `localize` | 59 | `x > 0` to `>=`, for the same reason at `x = 0`. |
| `localize2` | 199, 201, 202 | Inside the Nyquist branch of `brute`, which the loop bound makes unreachable. |
| `localize2` | 90 | `energy < cutoff` to `<=`, which differs only where a cumulative energy equals 99% of the total exactly. |
| `localize` | 12, 14 | Drops the `float64` conversion. The array is always scaled by or stacked with `float64`. |
| `localize_hrtf` | 4, 6, 17, 19 | Drops the conversion of the sound or of one response; convolution promotes to the other operand's `float64`. |
| `localize_hrtf` | 38, 40 | Drops `dtype=np.float64` from `np.zeros`, whose default it is. |
| `localize_hrtf` | 1 | Changes the default of `sample_rate`, which the routine does not use. |

The conversion survivors were checked, not argued: int8, uint8, bool,
float32 and list inputs gave identical samples with and without each one.

The reviews after the audit made `None` the angle sentinel, rendered the
default sound at the requested rate and answered an empty sound with an
empty stereo array. The final snapshot includes all three.

IDs use the `music.core.filters.localization.x_<function>__mutmut_`
prefix. The final snapshot uses the working-tree changes on `9078d08`,
with `localization.py` SHA-256
`8bd02c2a7940af55d808e1bfa2f69f2d02e75c88ad1d21cee1bf4f573be2ce74`.
The run took 233 seconds with two workers.

## `utils` — conversions, mixing, profiles and rhythmic durations

Measured 2026-09-25 on Python 3.12.7, macOS, with `mutmut` 3.7.0. The
selected set of nine files passed 996 tests and exercised all 154 branch
outcomes in `music/utils.py`; the statement report's misses are imports,
constants and definitions loaded before per-test coverage starts.

The run covered `HEAD` at `2aaa207`, overlaid with the working-tree
`music/utils.py` and its three edited selected test files. The runner normally
archives committed revisions only; after these changes are committed, the
standard command below reproduces the audit without an overlay.

### Result

The run killed 821 of 915 mutations and timed out four more, which mutmut
counts as detected. Ninety survived and were reviewed. There were no
untested, skipped or suspicious mutations, and no crashes. The selected tests
also found real cases that line and branch coverage had missed.

- `profile` squared `int16` and `float32` samples before widening them,
  overflowing the mean square and RMS. It now measures real numeric arrays
  in `float64`, rejects complex arrays as unmeasurable, and counts the last
  axis as frames for multichannel audio.
- `convert_to_stereo` returned a one-row mono array unchanged and accumulated
  extra integer channels in the narrow source type. It now duplicates the
  row and performs channel sums in `float64`.
- `resolve_stereo` replaced stereo values in the caller's argument mapping.
  It now works from copied per-channel arguments.
- `rhythm_to_durations` used BPM as `bpm / 60`, let BPM defeat an explicit
  `total_duration`, and mishandled NumPy and empty frequency sequences. It
  now uses seconds per beat, honors the documented precedence and handles
  the tested array, empty and nested cases. The corrected BPM path is a
  documented divergence from MASS.
- The tests now assert exact mixed samples, offsets, block statistics,
  rhythm subdivisions and warning/error behavior. `mix` is also checked not
  to mutate its longer input.

### Accepted survivors

| Function | Mutants | Review |
|---|---|---|
| `waveform_table`, `horizontal_stack` | 44; 1 | At the triangle midpoint both comparison branches return 1; `False` and `None` are both false in the stack flag. |
| `mix` | 1–4, 9–15, 19 | Eleven change only the invalid-input message; 19 selects the other path for equal lengths and produces the same sum. |
| `mix_stereo` | 4, 6, 15, 17, 22 | Explicit casts are redundant for real sample arrays because padding promotes the result; 22 changes only the equal-length path. |
| `convert_to_stereo` | 19–24 | Warning text only. |
| `_integrate_phase` | 1, 4, 6, 7, 12, 14 | Block size and branch-boundary variants retain the same bounded phase; omitted dtypes are NumPy's float64 default for these outputs. |
| `mix_with_offset` | 32, 37, 39–52, 63, 66 | Equal-size bound, zero-buffer assignment and clipped slice-end variants are equivalent; the rest change debug logging. |
| `mix_many_with_offsets` | 11, 22, 38 | Error wording, an erased typing cast, and an index increment beyond the final argument. |
| `pan_transitions` | 42, 50, 52, 65 | At equal lengths the added repeat has zero columns; remaining differences are dtype defaults. |
| `_describe_array` | 7, 9, 11, 47, 48, 66, 67, 69, 70, 112, 121, 127 | Equivalent reshape/index forms, NumPy dtype/copy defaults, or a block count that selects no additional samples. |
| `_guess_role` | 41, 44, 76, 77, 90, 92, 95, 103–106, 110, 111 | Redundant heuristic boundaries, defaults with the same truth value, or explanation wording. |
| `rhythm_to_durations` | 8, 15, 28, 30–32, 50, 78 | Refusal wording; mutant 60, `2 / i` in a normalized nested ratio, cancels during normalization. |

The four timed-out mutants were `mix_many_with_offsets` 26, 31, 36 and 37;
each hangs and is therefore detected. `tests/test_mass_reconciliation.py` now
also checks that BPM and total-duration behavior against the fixture rather
than leaving this routine marked exact from its default-duration case.

## Tool limitation and adapter

This applies to the `export` and `sequencer` areas; the other modules
define plain functions or undecorated classes.

`mutmut` 3.7.0 skips decorated classes and property-decorated methods.
Without an adapter, the session file contributes only `_ramp_shape`:
18 mutations, leaving all of the session methods out.

The runner removes each bare `@dataclass` decorator in the temporary copy
and applies `dataclass(ClassName)` immediately after the class instead.
It finds them by parsing the file, which the `sequencer` area needed: the
adapter had named the session's two classes.
Both classes are still dataclasses before any code uses them. Mutmut then
visits the ordinary methods. The adapted baseline tests and mutmut's
forced-failure check must pass before mutation results are collected.
No adapter is applied to the distributed package.

The `duration` property and generated dataclass methods remain outside
mutation scope, though normal tests exercise them. Code outside an area's
selected files is also outside that area. The runner pins the tool version
because its adapter and diff-export API depend on that behavior.
See the [mutmut source](https://github.com/boxed/mutmut) and
[documentation](https://mutmut.readthedocs.io/en/latest/).

## Reproduce

Use a development environment with the package's dev dependencies and
`mutmut==3.7.0`, then run from a Git checkout:

```console
python -m pip install mutmut==3.7.0
python tools/mutation_audit.py --list-areas
python tools/mutation_audit.py --area envelopes --revision HEAD
python tools/mutation_audit.py --area oscillators --revision HEAD
python tools/mutation_audit.py --area stimuli --revision HEAD
python tools/mutation_audit.py --area localization --revision HEAD
python tools/mutation_audit.py --area utils --revision HEAD
python tools/mutation_audit.py --area filters --revision HEAD
python tools/mutation_audit.py --area theory --revision HEAD
python tools/mutation_audit.py --area structures --revision HEAD
python tools/mutation_audit.py --area noises --revision HEAD
python tools/mutation_audit.py --area sequencer --revision HEAD
python tools/mutation_audit.py --area bonds --revision HEAD
python tools/mutation_audit.py --area tables --revision HEAD
python tools/mutation_audit.py --area hrtf --revision HEAD
python tools/mutation_audit.py --area singing --revision HEAD
python tools/mutation_audit.py --area legacy --revision HEAD
python tools/mutation_audit.py --area export --revision ef03962 --max-children 2
```

Omitting `--area` audits `export`, so the command the first audit recorded
still reproduces it. Uncommitted changes are excluded by default. For a
deliberate working-tree run, `--overlay-working-tree` copies the area sources
and selected tests into the revision's temporary archive; the report lists
every overlaid path. The runner leaves the working checkout untouched. It
prints the location of `audit.json`,
`survivors.patch` and mutmut's cache. The JSON records the area, the full
commit, the interpreter, the configuration, the runtime and every mutant's
exit code. The selected tests are in `tools/mutation_audit.py`.

Earlier runtimes, with four workers: 72 seconds for `envelopes`, 210 for
`oscillators`, and 84 for `export` with two. The latest oscillator pass took
582 seconds with two workers, `stimuli` 60 and `localization` 233, and
`theory` 24 to 50 with four, `structures` 172, `noises` 117,
`sequencer` 24, `bonds` 24, `tables` 12, `hrtf` 235, `singing` 55 and
`legacy` 321. None
is a candidate for CI at that cost — these are things to run
deliberately, read, and act on.

## `filters` — design, FIR/IIR, loudness, reverb and stretching

Measured 2026-09-25 on Python 3.12.7, macOS, with `mutmut` 3.7.0. The
178-test selection was built from per-test coverage contexts, then extended
with direct oracles for defaults, exact samples, frequency endpoints and
randomized paths. It passed; every source mutant was exercised.

The run covered `HEAD` at `700cc67`, with 17 working-tree paths overlaid:
the five source modules and their selected tests/support files. It took 85
seconds. Once committed, the standard command above reproduces this audit
without the overlay.

### Result

The run detected 505 of 531 mutations. Twenty-six survivors were reviewed;
none were untested, skipped, suspicious, timed out or crashed.

The full suite at this tree passed 4,175 tests. Its nine skipped doctest
items are `PrimaryTables.draw_tables`; the HRTF module examples,
`available_azimuths`, `hrir` and `setup_hrtf`; `singing.bootstrap`'s
`get_engine`, `setup_engine` and `make_test_song`; and `io.play_audio`. Each
is explicitly marked `+SKIP` in its docstring, for table drawing, external
HRTF or singing-engine setup, or audio-device playback. A focused run
confirmed 8 neighboring doctest items pass and all 9 skips come from that
option, not failures or missing-dependency skips.

The audit found several behavioral gaps:

- `louds` extended a shorter input signal with the final envelope gain,
  creating sound beyond the input where its documentation promises silence.
  It now pads with zeros.
- `loud` treated method names by substring and could silently accept an
  unknown method. It now accepts `lin`, `linear`, `exp` and `exponential`
  explicitly and raises `ValueError` for other values.
- `fraction_of`, `reverb` and `stretches` now reject nonpositive sample
  rates. `reverb` also rejects a negative first-phase duration.
- `stretches` now validates mono and two-channel stereo shapes, handles
  duration generators without consuming them during validation, and returns
  a correctly shaped empty result when no durations are requested.

The new exact tests also caught gaps in assertions for the FIR default
Nyquist endpoint, the reverb noise band and random stream, first-phase
incidence boundaries, and no-tail output dtype. A transition test now uses
a non-unit final gain so a wrong continuation formula cannot pass by
coincidence.

### Accepted survivors

IDs are mutmut suffixes for the recorded snapshot and tool version.

| Function | IDs | Why accepted |
|---|---|---|
| `fir` | 5, 7, 10, 12 | Removing either `float64` conversion leaves the other operand promoted to `float64`; the FFT path also yields a `float64` kernel. |
| `iir` | 3, 5, 8, 10, 13, 15 | The casts are redundant for the documented real PCM and coefficient inputs; differences require extended-precision or non-real inputs, for which no precision behavior is specified. |
| `iir` | 22, 28–30 | Error wording changes; invalid empty or zero divisors are still refused. |
| `iir` | 50, 67 | The longer coefficient slice is capped at the coefficient array's length, so both forms include the same terms. |
| `loud`, `louds` | 8; 2 | Replacing the scalar `0` sentinel with scalar `1` still makes `as_sonic_vector` return `None`. |
| `louds` | 33, 41 | On equal lengths the changed comparison only appends zero samples; the result is unchanged. |
| `reverb` | 23, 24 | The final explanatory wording changes; the duration refusal and its tested cause remain. |
| `reverb` | 5 | Both scalar defaults are treated as the same no-signal sentinel. |
| `reverb` | 78, 80 | `as_sonic_vector` and the response already produce `float64`, so the explicit result conversion is redundant. |
| `stretches` | 32 | `False` and `None` are both false in the mono/stereo branch. |

Remeasured 2026-09-30 at `841e809`, after `reverb` began holding its
noise's floor at the Nyquist frequency below 30 Hz, with the two tests of
that added to the selection: 501 of 527. The accepted survivors are
those above; `reverb`'s 78 and 80 are numbered 87 and 89 there.

## `theory` — scales, modes, chords, intervals and the harmonic series

Measured 2026-09-27 on Python 3.12.7, macOS, with `mutmut` 3.7.0. Per-test
coverage contexts over the full suite found that only `test_theory.py`,
`test_theory_properties.py`, four `test_degenerate.py` cases and four
`test_public_api.py` cases reach `music/theory/`; the selection is those,
and `test_theory_audit.py`, 659 tests in all.

The final run covered `HEAD` at `6f44763`, with 8 working-tree paths
overlaid: the three source modules and the selected test files. It took 50
seconds with four workers. Once committed, the standard command above
reproduces it without the overlay.

### Result

The first run detected 192 of 204 mutations. Eleven of the twelve survivors
were tests that did not exist:

- The article's three worked compound intervals, `P11`, `M9` and `m16`,
  and the simple ones read straight from the table left the parser
  unasserted on degrees 6, 7, 8 and 13 and above. Counting octaves from
  `degree + 1` rather than `degree - 1`, or taking the octave's degree 8
  as one octave up, gave `aug6`, `dim7`, `aug8` and `M13` the wrong size
  and passed.
- Nothing classified or named a minor ninth, so treating 13 semitones as
  a compound octave passed in both `consonance` and `interval_names`.
- `interval_between` was never given a frequency below 1 Hz, nor an upper
  frequency of zero with a message to check.
- `invert` was never asked for two octaves, where `12 / octaves` and
  `12 * octaves` part, and three defaults (`invert`'s degree,
  `mode_by_rotation`'s kappa and `harmonic_series`'s count) were run but
  not read.

`test_theory_audit.py` checks every quality with every degree up to three
octaves against an oracle built from the major scale rather than from the
module's table, and names, reads back and classifies every size up to five
octaves. The final run detected 213 of 214; the ten added mutations come
from the fixes below.

Reading the code around the mutants found four defects no mutant could
show, because each is an input the tests never gave:

- `interval_names(16.0)` returned `('M10.0',)`, a name `interval` cannot
  read, and `interval_names(4.5)` returned an empty tuple.
  `consonance(4.7)` truncated to 4 and called it a major third. Both now
  take a whole number of any numeric type and refuse a fraction.
- `interval` stripped whitespace only after looking the name up in the
  table, so `" M3"` was read and `"TT "` refused. It now strips first.
- `interval_between` passed NaN and infinity to `round()`, which failed
  with a message about converting to an integer. They are now refused as
  not positive and finite.
- `invert` read a negative `degree` from the top, as Python indexes,
  where its documentation promised an `IndexError`. That is useful and
  now documented rather than refused.

### Accepted survivor

| Function | IDs | Why accepted |
|---|---|---|
| `interval_names` | 31 | `continue` becomes `break` after the one name without a degree. That is `TT`, the last name of the tritone once `tri` is filtered out, so nothing follows it to skip. |

`interval("dim1")` is -1, as the article's rule gives it, where
`consonance` and `interval_names` refuse a negative interval. It is left as
the rule gives it, and `test_theory_audit.py` records the choice.

## Transition names — four areas re-run, 2026-09-27

`loud`, `fade`, `adsr`, `note_with_glissando` and
`note_with_vibrato_seq_localization` now share one check of their
transition names, `_transition_is_linear` in `utils.py`. The `utils`,
`envelopes`, `filters` and `oscillators` areas each gained
`tests/test_transition_methods.py`, and each was re-run at `bfd443f`, the
commit that made the change. None had a mutant without tests, and no
survivor was in the changed code: `utils` 836 of 930 with 4 hangs, 90
surviving; `envelopes` 528 of 555, 27; `filters` 492 of 518, 26;
`oscillators` 1435 of 1457 with 4 hangs, 18. `envelopes` lost eight
survivors along with the substring branches that held them. The recorded
results above are unchanged: they describe the revisions they name.

## `structures` — permutation families and change ringing

Measured 2026-09-27 on Python 3.12.7, macOS, with `mutmut` 3.7.0. Per-test
coverage contexts found `test_structures.py`, `test_peals_named.py` and 30
tests in seven other files reaching `music/structures/`, four of them
through the legacy `Being`. The selection is those and `test_structures_audit.py`, 256
tests. `symmetry.py` only re-exports, and is not mutated.

The final run covered `HEAD` at `bfd443f`, with 14 working-tree paths
overlaid: the four source modules and the selected test files. It took 172
seconds with four workers. Twenty-one mutants hang a peal that never comes
back to rounds, which the runner counts as detected.

### What reading and probing found

Before the first run, trying the routines on inputs their tests did not
use found five defects:

- `transpose_permutation` rebuilt the permutation as one cycle through
  its moved points in ascending order. That is right for a single swap,
  which is all the 2016 code used it for, and wrong for anything else:
  `(0 2 1)` came back as `(1 2 3)`, the same points cycled the other
  way, and two swaps `(0 1)(2 3)` as the four-cycle `(1 2 3 4)`. It was
  also sized to its highest point, so a shifted swap of four bells could
  not act on a row of four. It now shifts each cycle, keeps the size, and
  refuses a shift below zero. `test_additional.py` had pinned the old
  answer for `(0 2 1)`.
- `PlainChanges` documented a `hunts` argument that it overwrote on its
  first line, as it had since 2016. It now says so and warns when given
  one; laying the hunts out another way could produce a peal that never
  comes round.
- `print_peal` has eight colours and raised `IndexError` at the ninth
  bell. They now repeat.
- `even_odd` read a repeated entry as a cycle, so `[1, 1]` was odd. It
  refuses a sequence that is not a permutation.
- One element failed inside sympy for `InterestingPermutations` and
  `Peals`, and with an `IndexError` looking for a swap in `PlainChanges`;
  a negative `nhunts` with a `KeyError`; an unknown generation method as
  sympy's `NotImplementedError`. Each is now refused by name.

### Result

The first run, with those fixes and their tests in place, detected 546 of
725 mutations and left 179. Fifty-three were `print_peal`: termcolor
draws no colour when the output is not a terminal, so no test had ever
seen one, and even joining the digits with `XXXX` passed. Most of the
rest were the families `InterestingPermutations` builds, which the tests
counted rather than read: the swaps' order, the neighbour swaps, the
groupings by size and by step, the edge and vertex mirrors, the
alternations outside the polygon, and `PlainChanges.peal_sequence`.
`dist` survived `abs(a + b)` for `abs(a - b)`, as every swap it was given
involved bell 0.

`test_structures_audit.py` now describes each family without sympy's
groups: the dihedral group as the polygon's rotations and reflections,
the permutations by how many points they move, each distance as the
shorter way round, parity against sympy at odd sizes as well as even,
and `peal_sequence` as the change that takes each row to the next. It
prints with termcolor's colour forced on and off.

Two more defects came out of those tests:

- Every number of hunts from the saturating one to one fewer than the
  bells rang the same peal, as the warning says, but exactly as many
  hunts as bells passed the check and failed with an `IndexError`. The
  surviving mutant `nhunts >= nelements` was the correct check.
- sympy builds the alternating group of two elements on one point, so
  `InterestingPermutations(2)` counted rounds as an alternation outside
  the dihedral group. The pair is now written out, as the dihedral group
  of two already was.

Three simplifications removed equivalent mutants by saying what the code
meant: `dist` is `min(diff, size - diff)`, which its parity branches
computed; `transpose_permutation` leaves the size to sympy, which grows it
to fit; and attributes set to `None` and always overwritten are class
annotations, as the other families already were.

The final run detected 669 of 684.

### Accepted survivors

| Function | IDs | Why accepted |
|---|---|---|
| `an_eight_and_forty` | 35, 74 | Searching for rounds from the third row rather than the second: the second is one change from rounds, so it is never rounds. |
| `an_eight_and_forty` | 58 | A hunt stopping one place short of the back: the other whole hunt then hunts down from there, and its first change is the swap the first would have made. The 48 rows are identical. |
| `an_eight_and_forty` | 63 | `turn -= 1` alternates the two hunts as `turn += 1` does, modulo 2. |
| `PlainChanges.__init__` | 12 | `stacklevel=3` for 2. Under mutmut's trampoline the extra frame is the caller's, so the warning lands in the test either way; the test checks it is not attributed to `plain_changes.py`. |
| `initialize_hunts` | 11, 12 | `nhunts <= 0` and `< 1` for `< 0`: zero has already been replaced by the saturating count, so for integers the three are one test. |
| `perform_change` | 47, 78, 80, 81, 102 | `domains` is a trace of the change procedure that nothing in the package or its tests reads. |
| `even_odd` | 25, 28, 29 | Counting cycle lengths down, subtracting them, or adding one rather than taking one per cycle changes the total by an even amount, so the parity is the same. |

## `noises` — coloured noise, Gaussian bands and silence

Measured 2026-09-28 on Python 3.12.7, macOS, with `mutmut` 3.7.0. The
selection is `test_noises_audit.py` and the 42 test files and functions
that per-test coverage contexts found reaching `noises.py`: 705 tests, the
reverb, stimulus and article checks among them. The final run covered
`HEAD` at `27bece4` with 16 working-tree paths overlaid, in 117 seconds.

### What probing found

- `noise` rounded its band down at both ends. A `min_freq` between two
  components let in the one below it, and a component exactly at
  `max_freq` was always left out. The band is now every component from
  `min_freq` to `max_freq`, both included; an edge the resolution divides
  exactly is not lost to rounding in the division.
- `noise` drew `length // 2` phases, one too few for an odd length, so
  its highest component, `(N - 1) / 2`, was always silent: a reverb tail
  of odd length lost the top of its band. An odd length now draws one
  more; an even length draws what it did, so a seeded render after one is
  unchanged.
- `noise` returned silence for a band upside down, NaN for a slope of NaN
  or infinity, and divided by zero at a sample rate of zero. It refuses
  each, and a band above the Nyquist frequency, before counting samples,
  so a noise of no samples still says what was wrong with its arguments.
- `gaussian_noise` zeroed its band after mirroring the spectrum, so a
  band reaching past the Nyquist frequency kept mirrored components at
  twice the level of the rest. It now zeroes, then mirrors, and stops at
  the Nyquist frequency. It also scaled its samples onto [-1, 1] before
  a normalization that takes out the mean and divides by the peak, which
  undoes that; the line is gone, and the output is unchanged.
- With both band ends included, `gaussian_noise(std=0)` became a sine at
  the one component its zero-width band held. A width that is not
  positive is now refused by name rather than by the grid.
- `silence(-1)` raised numpy's "negative dimensions"; like `noise` and
  `note`, it now gives no samples.

### A test that depended on the tests before it

The first runs marked three reference-frequency mutants as killed that
the selection passed when run by hand. mutmut runs a mutant's tests
fastest first, and in that order
`test_full_depth_takes_the_noise_envelope_to_silence` failed. It read the
smallest modulated sample, unseeded: the envelope's lowest point times
whatever noise lay under it, above the bound for 46 of 200 random states.
It passed in the suite only because of the state the tests before it
left. It now seeds, and measures the envelope against the same noise
unmodulated.

To look for others, the whole suite ran eight more times with numpy's
generator seeded before each test from its name and a changing base.
That test was the only failure.

### Result

The first run detected 251 of 303 mutations. The survivors were the
random phases, which nothing compared with the draws they came from; the
defaults of both routines and of `silence`; the flatness of the band at
its highest component; the message for an unknown colour; and every
change to `gaussian_noise`'s redundant rescaling. The final run, after
the fixes and `test_noises_audit.py`, detected 280 of 285.

### Accepted survivors

| Function | IDs | Why accepted |
|---|---|---|
| `_band` | 9, 19 | Rounding the edge to ten decimals rather than nine: both absorb the error of the division, which is near 1e-16. |
| `noise` | 107, 133, 145 | Each rescales the frequency the slope is measured from, which multiplies every component by one constant. The normalization takes it out; the article's test divides it out too. |

## `sequencer` — scheduling, rendering and mixing notes

Measured 2026-09-28 on Python 3.12.7, macOS, with `mutmut` 3.7.0. Eleven
tests reached `sequencer.py`; with `test_sequencer_audit.py` the selection
is 39. The module's two dataclasses needed the adapter above. The final
run covered `HEAD` at `27bece4` with 6 working-tree paths overlaid, in
24 seconds.

### What probing found

- A start of NaN or infinity was accepted and failed in `round()` when
  the sequence rendered. It is refused when the note is added.
- `adsr_params` or `spatial` naming `sample_rate` or `sonic_vector`,
  which the sequencer passes itself, failed as a `TypeError` when the
  note rendered. They are refused when it is added.
- A sample rate of zero rendered every note as no samples. It is refused
  when the sequencer is built.
- Writing a sequencer with no notes reached the normalization, which
  blamed a duration computed as zero. It now says there are no notes.

### Result

The first run detected 132 of 176 mutations. The 44 survivors showed that
nothing compared a render with what it is made of: a note's frequency,
duration and rate could be dropped, overlapping notes could replace
rather than add to one another, and a written file could lose its
samples or its rate, and every test still passed. `test_sequencer_audit.py`
checks each note against the routine the sequencer hands it to, sample
for sample, overlapping mono and stereo mixes against their sums, and a
written file against the render. A vibrato test at a depth of 2 missed a
dropped depth, since that is `note_with_vibrato`'s default; it uses 3.

Two simplifications removed the equivalent survivors that were left:
`render` starts from no samples, which the first stereo note makes stereo
as any later one would, rather than choosing a stereo start itself; and
mixing copies the sequence so far into its silence rather than adding it.
The final run detected all 157.

## `bonds`, `tables`, `hrtf` and `singing`

Measured 2026-09-28 on Python 3.12.7, macOS, with `mutmut` 3.7.0, each
over `HEAD` at `c2fd5d1` with its sources and tests overlaid. Each
selection is the tests per-test coverage contexts found reaching it and a
new `test_<area>_audit.py`: 378 tests for `bonds`, 62 for `tables`, 54
for `hrtf` and 86 for `singing`. Running the engine itself, below,
changed `singing`: it was remeasured on 2026-09-29 at `d3acf37`, over 99
tests, detecting 510 of 517, and again on 2026-09-30 at `be262cc`, after
`lang` and `transpose` were checked, detecting 520 of 527. The accepted
survivors are the same seven. The `psola` backend joined the area on
2026-09-30; see below.

### What reading, probing and the survivors found

- `stepped` read its thresholds on every call, so a generator of them
  was read where the last call left off: 220 Hz fell in the first step
  and, a note later, the second. They are read once, when the bond is
  made.
- `Bonds.note` filled an unbound vibrato or tremolo value with a copy of
  the routine's default. It now passes only what is bound, so the
  routine's own default applies. The copies matched; nothing kept them
  matching.
- `PrimaryTables.make_tables` rebuilt the tables at a new size and left
  `size` at the old one, which `draw_tables` sized its axes by.
- `hrir` took a NaN elevation to -40 degrees: NaN compared false with
  every measurement, so the search kept the first. A NaN or infinite
  azimuth failed in `round()`. Both are refused.
- The singing engine's Makefile turns the score into MIDI with
  `abc2midi`, which the requirements check did not ask for: it passed,
  and the build failed inside `make`. It is a requirement now, and the
  install hint names its package, `abcmidi`.
- The note table named MIDI 60 `c`, which `abc2midi` reads as 72, so every
  score was an octave above its `reference`. It now names 60 `C`.

### Running the engine

With `abc2midi`, `sox` and the Perl modules installed on 2026-09-29, the
engine was run end to end for the first time in this work, and it would
not sing until four more things were fixed.

- `ecantorix.pl` loads `MIDI`, `Math::FFT`, `URI::Escape` and
  `Digest::SHA`, and shapes every syllable with `sox`. The requirements
  check now asks for all of them. Its `#!/usr/bin/perl` is the system
  Perl on macOS, not the one the check found, so `sing` runs it with the
  `perl` on PATH.
- The Makefile pipes the script through `tee`, so `make` succeeded when
  the script failed, and `sing` read the missing file as libsndfile's
  "System error", or read an earlier render back as this one's. It now
  removes the old render first and says what the engine printed.
- The engine loads the configuration with Perl's `do "achant.conf"`,
  which since Perl 5.26 no longer looks in the current directory. So
  `lang`, `transpose` and `effect` had never reached it, and it sang
  everything at its own -24: measured, a note of 0 at every
  transposition came out at 64.6 Hz. Running it with the cache on
  `@INC` fixes that. With the table an octave high, every note had been
  sung at `reference + note - 12`; the default is now -12, which keeps
  it there, and `transpose=0` sings middle C at 262.5 Hz.
- The effects load their files from the engine's `examples/`, relative
  to the directory the engine runs in, and `melt` gives espeak a voice
  of its own, for which the Makefile copies espeak's data from a Linux
  path. `sing` now puts both in the cache. `tremolo` and `melt` render in
  stereo, and came back as `(nsamples, 2)`; they are `(2, nsamples)`.

`test_singing_engine.py` sings and measures: middle C and the C above at
`transpose=0`, the default an octave below the score, another language,
and each effect. It skips where the engine is not set up, and it is not
in the mutation selection: four workers rendering in one cache would
overwrite each other's files.
- A duration of 0.5 went into the score as `0.5`, which is not ABC, and
  `make_test_song` used halves and quarters. A number is now written as
  the fraction ABC takes, `/2` for a half; `-n` still means `1/n`.
- `setup_engine` ran `git clone` into a directory holding something else,
  and reported git's exit status. It refuses, and leaves it alone. An
  unknown `effect` is refused before the engine is looked for.

### What the survivors showed unasserted

`bonds` left its messages and three defaults unread, and a render at
another rate unchecked. `tables` left its install message unread.

`hrtf` surviving mutants were half of them invisible on this machine:
macOS's filesystem is case-insensitive, so `FULL` for `full`, `l` for
`L` and `GZIP` for `gzip` found the same files and program. The tests now
record the paths read, the program asked for and the command run, which
holds on any filesystem. The real measurements are installed here, so
`available_azimuths` without its directory read them and still passed;
the test now uses an elevation where the synthetic grid differs. The
download is now checked for its timeout, for staging beside the target,
for decoding gzip's message when it is not UTF-8, and for the `data`
filter refusing a member that climbs out of the staging directory.

`singing` read the note table through `converter`, built when the module
is imported, before mutmut forks for each mutant, so no change to the
table-building code could reach a test. The tests now build the table
afresh, and record what `sing` hands the engine: the files it writes,
the `make` command, the file it reads back and how.

### Accepted survivors

| Function | IDs | Why accepted |
|---|---|---|
| `Bonds.note` | 40, 42 | Dropping the `float64` conversion: every routine it calls already returns `float64`. |
| `Bonds.render` | 14 | `cast`'s type argument, which does nothing at run time. |
| `hrir` | 1, 2 | A default one degree off rounds to the same measurement. |
| `hrir` | 53 | `astype(None)` is `float64`. |
| `setup_hrtf` | 43, 44, 69, 70 | The names of the decompressed archive and the unpacked tree, inside a temporary directory nothing else reads. |
| `_abc_length` | 22, 23 | Limiting the fraction to 1001 rather than 1000, and `<= 0` for `< 0` after zero is refused. |
| `Notes.make_dict` | 29 | The octave with three apostrophes lies above MIDI 96, and is sliced off. |
| `Notes.make_dict` | 46, 49, 57 | `strict` on a zip of two lengths the slice makes equal. |
| `Notes.make_dict` | 8 | `XX` around the names' string: the pattern does not match `X`. |

## `singing` with the `psola` backend

Measured 2026-09-30 at `ed193c7`, over 1101 mutations of the four singing
modules, with `test_singing_psola.py` added to the selection. That file
runs the backend against stand-ins for espeak-ng and Parselmouth, so it
covers every line where neither is installed; `test_singing_engine.py`
sings with the real ones and is left out of the selection, where four
workers rendering in one engine cache would overwrite each other.

The first run with the new module left 71 of 1097 alive. Most were in
the length and tempo parsers both backends share, `_note_length` and
`unit_seconds`, which nothing yet read exactly: ABC's slashes that
halve, a bare `Q` against `"beat=count"`, and the refusals, word for
word. The rest were the keyword arguments the backend hands
`subprocess.run`, the trim threshold, two boundaries in `_sung`, and the
backend's own defaults, which `sing` never leaves it to use. The final
run detected 1090 of 1101.

Measured again on 2026-10-01, over `19e5e5f` with two fixes to `_sung`
overlaid, both found by singing longer scores. A syllable said in
50 ms or less, as espeak says French "ques", stopped the backend with
Praat's error. And Praat's overlap-add writes into three times the
length of the sound it is given, so a note longer than that was cut
short: a four-second "laa" was sung for one second, then silence.
The fixes' 36 mutations were all detected, 1126 of 1137 in all, and
the same 11 survive, two of `_sung`'s renumbered.

| Function | IDs | Why accepted |
|---|---|---|
| `_require_flite` | 38 | `rpartition(":")` for `partition`: flite's voice line has one colon. |
| `Notes.make_dict` | 8, 29, 46, 49, 57 | As above: the `XX` the pattern ignores, the octave above 96, and `strict` on equal lengths. |
| `psola._fit` | 21 | `endpoint=None` is false, as `False` is. |
| `psola._sung` | 28 | A voiced frame above 1 Hz rather than 0: Praat gives an unvoiced frame 0 and a voiced one at least the 60 Hz floor. |
| `psola._sung` | 33 | Starting an unvoiced syllable's voiced stretch at 1 s: its end, 0, is before its start, so the stretch is not used either way. |
| `psola.sing` | 22 | Note edges left as floats: each is taken `int()` before use. |
| `psola.sing` | 88 | The peak of an empty line taken as 1 rather than 0: nothing divided is nothing. |

## `legacy` — the synthesizers, the Being and the demonstration piece

Measured 2026-09-28 on Python 3.12.7, macOS, with `mutmut` 3.7.0, over
`HEAD` at `c2fd5d1` with its five sources and tests overlaid, and again
on 2026-09-29 at `4d54ece`, after the fixes to `stay` and two reference
tables below. The selection is `test_legacy.py`,
`test_remaining_paths.py`, four more tests that reach the legacy
classes, and `test_legacy_audit.py`.

### What reading and probing found

- `V_` passed only the pitch to `note_with_vibrato`: the duration, the
  vibrato and the table were two seconds, 2 Hz, two semitones and a
  triangle whatever it was given, as they have been since 2017. So every
  note `Being.render` played lasted two seconds, whatever `d_` said.
- `CanonicalSynth.render2` took the table's length from `tables.size`,
  2048, rather than from the table it read, so any other table played at
  the wrong rate and read part of itself. It is now `rawRender` with the
  envelope, which it had repeated but for that.
- `adsrSetup` built a stage of one sample by dividing zero by zero.
- The demonstration piece passed `sounduration=` and `tre_freq=`, names a
  rename of `d` to `duration` had made of `sound=` two keywords ago.
  `absorbState` stored both as attributes nothing reads, so the note its
  comment says sounds like the next one was the tremolo envelope alone,
  and its tremolo rates changed nothing.
- `Being`'s `rhythm4` was `[1/4, 1/4, 1/3]` under the comment its three
  siblings share, "repetition of one second". It is four quarters.
- `freq_sym` took `j` notes of each symmetric step `j`, so its whole-tone
  row was two notes and its tritone row six, two and a half octaves. Each
  row is now one octave, `12 // j` notes, the start of the matching
  `freq_sym_` row.
- `notes_diatonic_` summed each rotation of the diatonic steps, which is
  12 every time, so all seven rows of `freq_diatonic` were the same
  octaves of 110 Hz. It is the scale's degrees, and each row one degree
  in every octave.
- `stay(method='straight')` read the first `seqsize` elements of the grid
  from `pointer % seqsize`, not the window at the pointer that
  `method='perm'` reads, and `'perm'` sliced that window without
  wrapping, so near the grid's end it came up short and the permutation
  refused it. Both now read the window at the pointer, wrapping round the
  end as a `'perm-walk'` window does. A test had pinned the old values;
  they were the window at the start of the grid, and it pins the new.

### Result

The first run detected 999 of 1637 mutations, five of them by hanging.
Of the 638 survivors, 549 were content rather than logic, and are below.
The other 89 were values the legacy tests never read: they checked
shapes and that samples were finite. `test_legacy_audit.py` checks
`rawRender` against the article's vibrato and phase equations, each ADSR
stage against the ramp it describes, `_fit` at both ends,
`IteratorSynth`'s cycling, and the walks, stays, defaults and written
file of a `Being`, exactly. The final run, on 2026-09-30 at `be262cc`,
detected 1154 of 1645, five by hanging.

### Accepted survivors

| Function | IDs | Why accepted |
|---|---|---|
| `TestSong2.__init__`, `TestSong2.render` | 444 mutants | The piece: every note, duration, rate and depth in it. A change to one is a different piece rather than a wrong one. The tests check that it renders, writes its files, and that the two notes its comment says sound alike do. |
| `Being.__init__` | 38 mutants | The reference sequences it stores in `resources` and no method reads: the harmonic and subharmonic spectra, the extended symmetric scales and the intensities. The rhythms, the one-octave symmetric scales and the diatonic rows are checked against what their names and comments say; these have nothing to check them against but themselves. |
| `adsrSetup` | 30, 32, 47, 50 | The `float64` of an integer range that is divided into floats anyway. |
| `adsrApply` | 33, 35 | The `float64` of `ones`, which is its default. |
| `adsrApply` | 6 | Compressing the stages when they exactly fill the note: at a ratio of one, `_fit` returns each stage as it is. |
| `_fit` | 1 | Returning the empty stage for a count of zero before `linspace` would: both are empty. |
| `Being.stay` | 24 | Adding one permutation more than it needs: the sequence is cut to `n`. |

The piece and the tables are left unpinned deliberately. Pinning the
piece would pin a performance: any edit to it, and any change in how a
platform rounds a phase, would fail, and nothing it caught would be a
defect. The tables that are pinned are the ones with a statement of
intent to check against, and that is where the three wrong ones were.

**Expand to another bounded area when changing it.** All sixteen areas
now have reviewed survivors. Pick later areas by measuring which tests reach
their modules with per-test coverage contexts, then add a separate entry to
`AREAS` rather than widening an existing area. These bounded measurements
are not a whole-package mutation score.

What the areas have taught, worth carrying:

- **An absent argument cannot be mutated.** No edit to `cross_fade` could
  expose a `sample_rate` that was never passed to `mix_with_offset`, nor
  to `trill` one that was never passed to `adsr`; both came out of
  reading the code around the mutants. A score measures the tests that
  exist against the code that exists.
- **Sort survivors by what they change, not where they are.** The last
  42 were grouped by function, and the grouping said little: most were
  refusal text and defaults, and the pitch-curve ones sat in routines
  listed as small.
- **A branch can run on every test and still be unasserted.** The
  defects found so far sat inside branches with full line *and* branch
  coverage, under arithmetic that nothing measured. That is the shape to
  look for.
- **A spectrum is not a sample.** The stimulus tests measured where a
  modulation's energy sat and nothing about its shape, so every edit
  that kept the rate survived. A constant or ramp table reads the
  envelope or the phase directly.
- **An apparent equivalent deserves an experiment.** Two `float64`
  conversions in `spatial_motion` looked redundant; one line of other
  input types showed they were not.
- **A finite domain can be enumerated.** The interval notation has seven
  qualities and a handful of degrees per octave. The article's examples
  sampled three of them and happened to miss every degree where a wrong
  octave count differs; checking all 154 against an independent oracle
  takes a fraction of a second.
- **A survivor can be the fix.** `nhunts >= nelements` survived because
  no test tried as many hunts as bells, and that was the check the code
  needed: the original let them through to an `IndexError`. Try the
  mutant's boundary before calling it equivalent.
- **Output that depends on the terminal must be forced.** termcolor drew
  no colour under pytest, so fifty-three edits to `print_peal` changed
  nothing any test could see.
- **A kill can be a flake.** mutmut reorders tests, so an unseeded test
  that passes in the suite's order can fail in another and kill a mutant
  that changes nothing. When a result is surprising, run the selection by
  hand, in mutmut's order.
- **Compare with what the result is made of.** The sequencer delegates
  every note to another routine, so the test of a render is that routine
  called the same way, not a length or a peak.
- **A case-insensitive filesystem hides path mutants.** On macOS `FULL`
  finds `full`. Record the path a routine uses rather than whether it
  found something.
- **Work done at import is out of mutmut's reach.** A table built when
  the module loads is built before the fork, with the original code.

Keep the sample-based assertions when refactoring these paths. Resolve the
property/decorator coverage limitation before interpreting any future
whole-package score.
