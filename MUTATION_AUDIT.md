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

| Area | Sources | Measured | Mutations | Killed | Surviving |
|---|---|---|---:|---:|---:|
| `export` | `core/functions.py`, `core/io.py`, `stimulation/session.py` | 2026-09-16 | 767 | 698 | 69 |
| `envelopes` | `synths/envelopes.py`, `filters/adsr.py`, `filters/fade.py` | 2026-09-20 | 570 | 535 | 35 |

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

| Module | Mutations | Killed | Surviving |
|---|---:|---:|---:|
| `music/core/synths/envelopes.py` | 160 | 154 | 6 |
| `music/core/filters/adsr.py` | 188 | 178 | 10 |
| `music/core/filters/fade.py` | 222 | 203 | 19 |
| **Total** | **570** | **535** | **35** |

The first run, against the code as it stood, killed 439 of 550 with 110
survivors and one timeout. The totals differ because the corrections below
changed the code being mutated; nothing timed out, was skipped, crashed or
went untested in either run.

| Function | Mutations | Killed | Surviving |
|---|---:|---:|---:|
| `am` | 32 | 31 | 1 |
| `tremolo` | 48 | 47 | 1 |
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
python tools/mutation_audit.py --area export --revision ef03962 --max-children 2
```

Omitting `--area` audits `export`, so the command the first audit recorded
still reproduces it. Uncommitted changes are deliberately excluded: the
runner archives the requested revision into a temporary tree and leaves the
working checkout untouched. It prints the location of `audit.json`,
`survivors.patch` and mutmut's cache. The JSON records the area, the full
commit, the interpreter, the configuration, the runtime and every mutant's
exit code. The selected tests are in `tools/mutation_audit.py`.

Runtimes, with four workers: 74 seconds for `envelopes`, 84 for `export`
with two. Neither is a candidate for CI at that cost — these are things to
run deliberately, read, and act on.

## Next

Expand to another bounded area when changing it. **Oscillator timing is the
next one**, and it already has a defect waiting: the signed `** alpha` in
the four vibrato routines named above. Add an area to `AREAS` rather than
widening an existing one, and pick the test selection by measuring which
tests reach the module — `pytest --cov-context=test` and a query over the
coverage database — rather than by guessing at names.

Keep the sample-based assertions when refactoring these paths. Resolve the
property/decorator coverage limitation before interpreting any future
whole-package score.
