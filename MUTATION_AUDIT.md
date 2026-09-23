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
| `oscillators` | `synths/notes.py` | 2026-09-23 | 1460 | 1442 | 18 | all |
| `stimuli` | `stimulation/stimuli.py` | 2026-09-23 | 427 | 423 | 4 | all |
| `localization` | `filters/localization.py` | 2026-09-23 | 644 | 623 | 21 | all |

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
| `trill` | 50 | 48 | 2 | 6 |
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
| `trill` | 32 | Drops `waveform_table=WAVEFORM_TRIANGULAR` from the `note` call. That is `note`'s default. |

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
The final snapshot uses the working-tree changes on `95cf57a`, with
`stimuli.py` SHA-256
`384c0b5e9628ddc2d7c3f399095ab770e494aea2df0c3c26d3d068a72af941da`.
The run took 60 seconds with two workers.

## `localization` — interaural cues, fixed, per-frequency, moving and convolved

Measured 2026-09-23 on Python 3.12.7, macOS, with `mutmut` 3.7.0. No
adapter. The eighteen selected test files are the seventeen that
`pytest --cov-context=test` found reaching `localization.py`, plus the
one this audit added. Together they reach every line and branch of it.

### Result

| | Mutations | Detected | Surviving |
|---|---:|---:|---:|
| First run | 611 | 526 | 85 |
| After the tests and corrections below | 644 | 623 | 21 |

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

**Not changed: a zero angle reads as not supplied.** Both routines take
`theta=0` as "use `x` and `y`". `localize2`'s default angle is -70, so
passing zero is how its callers select a position, and existing tests do
exactly that. Placing a source at zero degrees takes its coordinates.
Making `None` the sentinel would break those callers, so it is now
documented rather than changed.

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
| `localize2` | 14, 15, 154, 155 | Case changes or `XX` around the refusal and the warning text. |
| `localize2` | 140, 216, 226 | `theta_ > 0` to `>=`. At zero the delay is zero and the gain one, so both branches render the same samples. |
| `localize` | 58 | `x > 0` to `>=`, for the same reason at `x = 0`. |
| `localize2` | 193, 195, 196 | Inside the Nyquist branch of `brute`, which the loop bound makes unreachable. |
| `localize2` | 84 | `energy < cutoff` to `<=`, which differs only where a cumulative energy equals 99% of the total exactly. |
| `localize` | 12, 14 | Drops the `float64` conversion. The array is always scaled by or stacked with `float64`. |
| `localize_hrtf` | 4, 6, 17, 19 | Drops the conversion of the sound or of one response; convolution promotes to the other operand's `float64`. |
| `localize_hrtf` | 38, 40 | Drops `dtype=np.float64` from `np.zeros`, whose default it is. |
| `localize_hrtf` | 1 | Changes the default of `sample_rate`, which the routine does not use. |

The conversion survivors were checked, not argued: int8, uint8, bool,
float32 and list inputs gave identical samples with and without each one.

IDs use the `music.core.filters.localization.x_<function>__mutmut_`
prefix. The final snapshot uses the working-tree changes on `b926e7a`,
with `localization.py` SHA-256
`29f4dd2cd0f14b1099f62cf61945551944eda2c8d8636e5ceef7b3b7e8670346`.
The run took 307 seconds with two workers.

## Tool limitation and adapter

This applies to the `export` area alone; the envelope modules are plain
functions and need no adapter.

`mutmut` 3.7.0 skips decorated classes and property-decorated methods.
Without an adapter, the session file contributes only `_ramp_shape`:
18 mutations, leaving all of the session methods out.

The runner removes the two `@dataclass` decorators in the temporary copy
and applies `dataclass(ClassName)` immediately after each class instead.
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
python tools/mutation_audit.py --area export --revision ef03962 --max-children 2
```

Omitting `--area` audits `export`, so the command the first audit recorded
still reproduces it. Uncommitted changes are deliberately excluded: the
runner archives the requested revision into a temporary tree and leaves the
working checkout untouched. It prints the location of `audit.json`,
`survivors.patch` and mutmut's cache. The JSON records the area, the full
commit, the interpreter, the configuration, the runtime and every mutant's
exit code. The selected tests are in `tools/mutation_audit.py`.

Earlier runtimes, with four workers: 72 seconds for `envelopes`, 210 for
`oscillators`, and 84 for `export` with two. The latest oscillator pass took
582 seconds with two workers, `stimuli` 60 and `localization` 307. None
is a candidate for CI at that cost — these are things to run
deliberately, read, and act on.

## Next

**Expand to another bounded area when changing it.** All five areas are
closed, with every survivor reviewed. `utils.py` is the largest module
left, and the one the others lean on most; `core/filters/` beyond the
envelopes and localization is the next most used. Add an area to
`AREAS` rather than widening an existing one, and pick the test selection
by measuring which tests reach the module — `pytest --cov-context=test`
and a query over the coverage database — rather than by guessing at names.

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

Keep the sample-based assertions when refactoring these paths. Resolve the
property/decorator coverage limitation before interpreting any future
whole-package score.
