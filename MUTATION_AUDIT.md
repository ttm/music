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
| `oscillators` | `synths/notes.py` | 2026-09-21 | 1402 | 1248 | 154 | the vibratos; the rest unread |

The `oscillators` row is not a finished audit. It found the defect it went
looking for and closed the class of gap that hid it, and the rest of its
survivors have not been read. Say so rather than letting the row imply
otherwise; `## Next` records what is left.

Each area names the files it mutates and the tests that judge them, and the
tests must cover every line and branch of those files between them, or
mutmut reports mutants no test reaches. Both areas report none.

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

Measured 2026-09-21 on Python 3.12.7, macOS, with `mutmut` 3.7.0, on the
tree this file is committed with. No adapter. The selection is 29 test
files, chosen the same way as the envelopes', and covers every line and
branch of the 382 statements and 104 branches in `notes.py`.

**The vibratos are done; the rest is not.** Two passes: the first went
after one defect and found it, the second closed the routines that carry a
vibrato and found three more. 154 survivors remain unread, 80 of them in
`note_with_vibrato_seq_localization`.

### Result

| | Mutations | Detected | Surviving |
|---|---:|---:|---:|
| First run | 1375 | 1124 | 251 |
| After the latching correction | 1396 | 1165 | 231 |
| After closing the vibrato routines | 1402 | 1248 | 154 |

Four mutants time out rather than returning a wrong answer, in both runs.
They are counted as detected: `trill` accumulates samples in a `while`
loop, and `pointer -= ns`, `i -= 1`, `i = 1` and a `*` turned `/` each
make that loop run forever. No suite that finishes accepts them, and the
runner no longer reports a run containing them as incomplete.

| Function | Mutations | Detected | Surviving | Was |
|---|---:|---:|---:|---:|
| `note_with_vibrato_seq_localization` | 499 | 419 | 80 | 81 |
| `note_with_vibratos_glissandos` | 138 | 120 | 18 | 22 |
| `note_with_doppler` | 182 | 168 | 14 | 14 |
| `_require_a_ratio` | 14 | 6 | 8 | 10 |
| `_exponential_positions` | 20 | 12 | 8 | 8 |
| `note_with_glissando` | 61 | 55 | 6 | 6 |
| `trill` | 48 | 42 | 6 | 6 |
| `note_with_fm` | 39 | 35 | 4 | 4 |
| `note_with_phase` | 27 | 24 | 3 | 3 |
| `note` | 20 | 18 | 2 | 2 |
| `note_with_glissando_vibrato` | 82 | 80 | 2 | 25 |
| `note_with_two_vibratos_glissando` | 110 | 108 | 2 | 33 |
| `_fit_to_samples` | 17 | 16 | 1 | 1 |
| `note_with_vibrato` | 57 | 57 | 0 | 13 |
| `note_with_two_vibratos` | 88 | 88 | 0 | 23 |

"Was" is the first run. The five routines that carry a vibrato held 112
survivors between them and now hold 22.

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

### What has not been read

The remaining 154, of which `note_with_vibrato_seq_localization` holds 80
— the largest single block in the package — `note_with_vibratos_glissandos`
18 and `note_with_doppler` 14. The shape of the first run suggests
most are the same kinds already accepted elsewhere — diagnostic wording,
defaults nothing calls bare, arguments a callee does not read — but that
is a guess until someone reads them, and this file does not record guesses
as findings.

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
python tools/mutation_audit.py --area export --revision ef03962 --max-children 2
```

Omitting `--area` audits `export`, so the command the first audit recorded
still reproduces it. Uncommitted changes are deliberately excluded: the
runner archives the requested revision into a temporary tree and leaves the
working checkout untouched. It prints the location of `audit.json`,
`survivors.patch` and mutmut's cache. The JSON records the area, the full
commit, the interpreter, the configuration, the runtime and every mutant's
exit code. The selected tests are in `tools/mutation_audit.py`.

Runtimes, with four workers: 72 seconds for `envelopes`, 210 for
`oscillators`, and 84 for `export` with two. None is a candidate for CI at
that cost — these are things to run deliberately, read, and act on.

## Next

**Finish reading `oscillators`.** Its 154 survivors are the outstanding
work in this file, and `note_with_vibrato_seq_localization` holds 80 of
them. `note_with_vibrato` shows what closing one costs and what it buys:
one test that measures the pitch it actually renders took it from 13
survivors to none, and that same test is what the defect could not have
survived.

Then expand to another bounded area when changing it. Add an area to
`AREAS` rather than widening an existing one, and pick the test selection
by measuring which tests reach the module — `pytest --cov-context=test`
and a query over the coverage database — rather than by guessing at names.

Two things the three areas have taught, worth carrying:

- **An absent argument cannot be mutated.** No edit to `cross_fade` could
  expose a `sample_rate` that was never passed to `mix_with_offset`; that
  came out of writing the tests that kill the mutants around it. A score
  measures the tests that exist against the code that exists.
- **A branch can run on every test and still be unasserted.** Both defects
  found so far sat inside a branch with full line *and* branch coverage,
  under arithmetic that nothing measured. That is the shape to look for.

Keep the sample-based assertions when refactoring these paths. Resolve the
property/decorator coverage limitation before interpreting any future
whole-package score.
