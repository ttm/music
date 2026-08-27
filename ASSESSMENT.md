# Quality Assessment — `music` 1.0.1

*Assessment date: 2026-08-27 · commit `5a7711a` · 36 modules, 6,306 LOC package + 529 LOC tests*

This document grades the repository honestly, section by section, and ends with a
prioritised plan. Every claim below was verified by running the code — no judgement
here is stylistic guesswork.

> **Status.** Phases 1 and 2 of the plan below are done, on the branch
> `fix/broken-exports` (see `CHANGELOG.md`). All six defects in the register are
> fixed and covered by regression tests, the filters import cycle is broken, CI
> runs ruff + mypy + pytest across Python 3.10–3.13, and coverage rose from
> **43 % to 59 %**. The grades below describe the repository *as assessed at the
> commit named above*, so the "Terrible" section now reads as a record of what
> was fixed rather than of what ships. Phases 3–5 remain open.

## How this was measured

| Check | Command | Result |
|---|---|---|
| Test suite | `pytest` | **38 passed**, 2.3 s |
| Coverage | `pytest --cov=music` | **43 %** (1,884 stmts, 1,075 missed) |
| Type check (as configured) | `mypy music` | **clean**, 36 files |
| Type check (bodies included) | `mypy --check-untyped-defs music` | **184 errors** in 14 files |
| Annotation coverage | AST scan | **21 / 125 functions (17 %)** |
| Docstring coverage | AST scan | **117 / 126 public defs (93 %)** |
| Lint | `ruff check` (E,W,F) | 21 findings; extended rule set: 217 |
| Examples | run all 10 | **9 pass**, 1 needs external engine |
| Public API smoke test | call every zero-arg export | **30 / 33 work, 3 always raise** |

---

## Verdict at a glance

| Grade | What earns it |
|---|---|
| **Exceptional** | Docstrings and scientific grounding |
| **Excellent** | Conceptual architecture; breadth of synthesis primitives |
| **Good** | Curated public API; `Sequencer`; packaging metadata; examples |
| **OK** | Test suite; README; changelog |
| **Bad** | No CI; vacuous type checking; `eval`/`exec`; type-dispatch idiom; version metadata |
| **Terrible** | Six confirmed defects in the *exported* API, incl. two functions that can never succeed |
| **Nothing is Perfect** | See below |

### Is anything perfect?

No, and one thing is close. **`music/core/filters/reverb.py` is at 100 % coverage, fully
documented, clean under the default lint set and clean under mypy.** It is 75 lines. It still
carries the package-wide `type(x) in (np.ndarray, list)` idiom, so even the best module here
inherits a structural flaw. That is the honest ceiling: the bar is reachable in this
repository, and nothing has quite reached it.

---

## Exceptional

**Documentation of individual functions.** 93 % of public definitions carry docstrings, and
they are not one-liners: they are full numpydoc blocks with `Parameters`, `Returns`,
`See Also`, `Examples`, `Notes` and a literature `References` section citing
*Musical elements in the discrete-time representation of sound*. `note()` in
[music/core/synths/notes.py:7](music/core/synths/notes.py#L7) spends 50 lines documenting a
9-line function, and the explanation of why LUT lookup incorporates the vibrato pattern is
the kind of thing almost no audio library writes down.

This is genuinely rare. Most DSP packages document *what* a parameter is; this one documents
the psychophysics behind it and points at the paper. It is the single strongest asset in the
repository and it should be protected in any refactor.

**Grounding in a published framework.** The package is an implementation of MASS rather than
an ad-hoc collection of effects. Sample-accurate state updates (each sample gets its own
instantaneous frequency) is a real, defensible design commitment, not marketing.

## Excellent

**Conceptual decomposition.** `core/{synths,filters,io,functions}` · `structures` · `singing`
· `legacy` · `tables` · `utils` · `sequencer` is a taxonomy that maps cleanly onto the domain.
Someone who knows audio can guess where a function lives. `legacy/` being explicitly named and
quarantined rather than left mixed into the core is a mature call.

**Breadth of primitives.** Vibrato (single, double, sequenced), glissando, FM, AM, tremolo,
Doppler, ADSR (mono/stereo/vibrato), reverb, FIR/IIR, binaural localisation with ITD/IID,
six noise colours, plus change-ringing peals, plain changes and permutation groups. The
campanology/permutation side of `structures/` is unusual and well-scoped — there is little
else in the Python ecosystem that does it.

**A single flat namespace of 70 curated exports.** `import music; music.note(...)` works, and
the `__all__` lists are hand-maintained rather than star-imported. That is deliberate API
design.

## Good

- **`music/sequencer.py`** — the newest module and the best-engineered one: dataclasses, full
  annotations, `from __future__ import annotations`, 91 % coverage. It shows what current
  practice in this repo looks like when unconstrained by history.
- **Packaging metadata** — 24 keywords, 15 classifiers, dual URLs, `dev` extra. Well above
  average for a research package.
- **Examples** — 10 runnable scripts, 9 of which run clean against the working tree. They are
  short, readable, and each demonstrates one idea.
- **`mypy` and `pytest` are configured at all** — in a package of this vintage that is not a
  given.

## OK

- **Test suite.** 38 tests, all green, fast. But it is 529 LOC against 6,306 LOC of package —
  an 8 % ratio — and roughly half the assertions only check `len()` or `.shape`. For a package
  whose central claim is *extreme fidelity*, there is no test that compares a synthesised
  signal against the closed-form equation it implements. `tests/test_spectral.py` is the right
  idea and the right direction; there are three of them.
- **Test bootstrapping is inconsistent and hacky.** [tests/test_utils.py:9](tests/test_utils.py#L9)
  loads modules by file path via `importlib.util.spec_from_file_location`, bypassing the package
  entirely; [tests/test_synths.py:8](tests/test_synths.py#L8) instead does
  `sys.path.insert(0, HERE)`. Neither exercises the real installed import path — which is
  precisely how the `trill` bug below survived.
- **README.** Clear and well-organised, but the Roadmap block shows `music.render_demos()`,
  `music.legacy.experiments`, `music.legacy.songs` and `music.remix()` — none of which exist —
  without labelling them as aspirational, and the file ends with a stray `:::`.
- **CHANGELOG.** Exists, correct format, one entry. Nothing before 1.0.1 is recorded.

## Bad

- **No CI.** There is no `.github/` directory. Nothing runs the tests, the linter or mypy on
  push. Every quality gate in this repo is opt-in and manual.
- **The type checking is vacuous.** `mypy music` reports success — but only 17 % of functions
  are annotated, and mypy skips the bodies of unannotated functions by default. Adding
  `--check-untyped-defs` surfaces **184 errors** — 40 `attr-defined` errors from attributes
  assigned via `exec`, 105 `call-arg` errors, and 156 of the 184 concentrated in `legacy/`.
  The green checkmark is currently measuring almost nothing.
- **`eval` and `exec` in library code.** [music/core/io.py:102](music/core/io.py#L102) and
  [:157](music/core/io.py#L157) do `eval("np.int" + str(bit_depth))` where a dict lookup
  suffices. `CanonicalSynth` and `testSong2` use `exec("self.{}={}".format(i, i))` three times
  to set attributes — which is *why* mypy cannot see those attributes.
- **`type(x) in (np.ndarray, list)` appears 16 times** as the input-dispatch idiom. It fails
  for tuples and for any ndarray subclass, and it fails *silently*. Verified:
  `adsr(sonic_vector=tuple_of_44100_samples)` does not raise — it discards the input and
  returns the default 2-second envelope. A wrong answer is worse than an exception.
- **~2.4 MB of RNG allocated at import.** [music/core/io.py:12-14](music/core/io.py#L12)
  builds 300,000 random samples at module scope purely to serve as default arguments. Import
  cost and memory for something almost no caller wants. (They are also `np.random.uniform`,
  i.e. in `[0, 1)` — a DC-offset signal, not audio.)
- **`requires-python = '>=3.0'` is wrong.** The code uses PEP 604 unions
  (`np.ndarray | None`, [music/sequencer.py:110](music/sequencer.py#L110)) and mypy is pinned to
  3.11. Installing on anything below 3.10 will fail at import. pip is being told the opposite.
- **No `__version__` and no `py.typed`.** The version lives only in `pyproject.toml`, so it is
  not introspectable at runtime; and without a `py.typed` marker, the annotations that *do*
  exist are invisible to every downstream type checker.
- **Three parallel waveform-table implementations** — `utils.WAVEFORM_*`,
  `tables.PrimaryTables`, `legacy/tables.py` — and they have already drifted: `PrimaryTables`
  builds its triangle as `hstack((foo, -foo))` while `utils` uses `hstack((tmp, tmp[::-1]))`,
  giving different peak samples. Duplication that has begun to diverge is duplication that
  will produce a support ticket.
- **`setup_engine()` clones a git repository into the installed package directory.**
  [music/singing/bootstrap.py:21](music/singing/bootstrap.py#L21) writes into `site-packages`
  at runtime. This breaks on read-only installs, containers, and any multi-user environment.
  Its system dependencies (`git`, `make`, `perl`, `espeak`, `abcmidi`) are declared nowhere.
- **Stale build artifacts.** `dist/` holds `music-1.0.0b5` wheels while the project is at
  1.0.1.

## Terrible

These are not style opinions. Each was reproduced by running the code.

**1. `stretches()` can never succeed.** It is exported in `music.__all__`.

```python
obj = object()
obj.foo = s_        # AttributeError: 'object' object has no attribute 'foo'
```

[music/core/filters/stretches.py:38-39](music/core/filters/stretches.py#L38). `obj` is used
nowhere else — it is abandoned scratch code sitting on the only path through the function.
The preceding line, `s_ = durations * sample_rate`, multiplies a *tuple* by 44100, building a
176,400-element tuple by repetition. Coverage confirms it: stretches.py sits at 12 %, and the
missed range is the entire body.

**2. `trill()` can never succeed** — `TypeError: 'module' object is not callable` at
[music/core/synths/notes.py:1275](music/core/synths/notes.py#L1275). The cause is structural:
`music/core/filters/` contains submodules named `adsr`, `fade`, `loud`, `reverb` and
`stretches` that **collide with the function names re-exported from them**. `adsr.py` imports
from `notes.py`, which imports `adsr` back from the partially-initialised `filters` package —
and during that window the name still refers to the *module*. Five colliding names means five
latent instances of this; one has already fired.

**3. `louds()` raises whenever the envelope is shorter than the signal.**
[music/core/filters/loud.py:198](music/core/filters/loud.py#L198) writes the padded result to
`s` instead of `e`, then returns `sonic_vector * e` with the unpadded `e`:

```
ValueError: operands could not be broadcast together with shapes (132300,) (88200,)
```

A one-character typo, on the main path, in an exported function.

**4. `localize_linear()` crashes on its own documented defaults** —
`TypeError: only length-1 arrays can be converted to Python scalars`. The function carries the
comment *"FIXME: here we have missing the correct use of the variables calculated and also the
return statement"* ([localization.py:151](music/core/filters/localization.py#L151)) and returns
a 5-tuple of intermediates. It is knowingly unfinished, and it is exported as public API.

**5. Two of the four documented WAV bit depths are broken.** `write_wav_mono` /
`write_wav_stereo` advertise `bit_depth ∈ {8, 16, 32, 64}` and validate against exactly that
set. Verified round-trip:

| bit_depth | result |
|---|---|
| 8 | `ValueError: Unsupported data type 'int8'` — 8-bit WAV is *unsigned* by spec |
| 16 | works |
| 32 | works |
| 64 | writes a file that this package's own `read_wav` then rejects |

**6. Shipping broken exports is the pattern, not the incident.** Three of the 33
zero-argument public functions raise unconditionally when called as documented. Nothing in
the repository would have caught that, because nothing runs.

## What is lacking

Not bad — simply absent:

- **CI/CD.** No workflow, no matrix across Python versions, no coverage gate, no publish job.
- **Lint configuration.** `ruff`/`flake8` are not configured or pinned; the 217 extended
  findings are unmanaged.
- **Fidelity/regression tests.** No golden-signal comparison against the MASS equations, no
  spectral assertion on vibrato sideband placement, no WAV round-trip test per bit depth. For
  this package specifically, this is the most conspicuous gap: the headline claim is untested.
- **API reference documentation.** 93 % docstring coverage and no Sphinx/MkDocs site to render
  it. The best asset in the repo is invisible to anyone who has not cloned it.
- **`py.typed`**, `__version__`, `CONTRIBUTING.md`, issue/PR templates, `pre-commit`.
- **Audio-domain guards.** No clipping detection, no sample-rate consistency checks between
  combined vectors, no dtype validation at API boundaries.
- **A deprecation policy for `legacy/`.** It is quarantined but not scheduled.

---

## How to raise the quality

Ordered by return on effort. Phase 1 is roughly a day and removes every "Terrible" item.

### Phase 1 — Stop shipping broken exports (highest value)

1. **Fix the six confirmed defects.**
   - `loud.py:198` — `s = np.hstack(...)` → `e = np.hstack(...)`.
   - `stretches.py:38-39` — delete the `obj = object()` lines and the `obj.bar` line; fix
     `s_ = durations * sample_rate` (it is unused once `obj` is gone — delete it too).
   - `notes.py:4` — resolve the collision (see 2 below).
   - `localize_linear` — either finish it or remove it from `__all__` and raise
     `NotImplementedError`. Do not export a function whose body says it has no return statement.
   - `io.py` — map `bit_depth` to `{8: np.uint8, 16: np.int16, 32: np.int32}` via a dict,
     offset-encode for 8-bit, and drop 64 from the advertised set. This also removes both
     `eval` calls.
2. **Break the submodule/function name collisions.** Rename the five colliding modules
   (`adsr.py` → `_adsr.py`, or `envelope.py`), or make every intra-package import fully
   qualified (`from music.core.filters.adsr import adsr`). This is the root cause of #2 and
   four more latent instances.
3. **Add a smoke test that calls every name in `music.__all__` with its documented defaults.**
   Twenty lines. It would have caught three of the six defects above on the day they landed.
4. **Fix `requires-python` to `>=3.10`.**

### Phase 2 — Make the quality gates real

5. **Add GitHub Actions**: `pytest` + `ruff` + `mypy` on 3.10/3.11/3.12/3.13, on every push
   and PR. Nothing else in this list holds without it.
6. **Turn on `check_untyped_defs = true`** in `[tool.mypy]`, then work the 184 errors down
   module by module. **156 of the 184 errors (85 %) are in `legacy/`**, so excluding that
   package initially leaves 28 errors across the rest of the codebase — a single sitting.
   Ratchet, don't bulk-fix.
7. **Add `ruff` with a committed config**; auto-fix the 79 mechanical findings, triage the rest.
8. **Add `music/py.typed`** and a `__version__` single-sourced from package metadata.
9. **Set a coverage floor at the current 43 %** and raise it as you go, so it cannot regress.

### Phase 3 — Test what the package actually claims

10. **Golden-signal tests.** For `note`, `note_with_vibrato`, `note_with_fm`, `adsr`: assert
    against the closed-form MASS expression, not against `len()`. This is what "extreme
    fidelity" means and it is currently unverified.
11. **Spectral assertions** — extend `test_spectral.py`: vibrato sidebands at f ± k·f_v, FM
    Bessel-ratio amplitudes, noise colour slopes (−3 dB/oct for pink, +3 for blue) via
    `scipy.signal.welch`.
12. **WAV round-trip per bit depth**, and a localisation test asserting ITD sign and magnitude
    against the geometric prediction.
13. **Unify test bootstrapping** — install the package in CI (`pip install -e .`) and delete
    every `sys.path` and `spec_from_file_location` hack. Tests must exercise the real import
    path.

### Phase 4 — Structural debt

14. **Replace all 16 `type(x) in (np.ndarray, list)` checks** with
    `sonic_vector is not None` plus `np.asarray()`, and change the sentinel defaults from `0`
    to `None`. This eliminates a whole class of silent-wrong-answer bugs.
15. **Collapse the three waveform-table implementations to one** (`utils.WAVEFORM_*` is the
    canonical set); have `PrimaryTables` and `legacy/tables.py` delegate to it.
16. **Move the module-level RNG defaults into the function bodies** (`if sonic_vector is None:
    ...`) and drop 2.4 MB and the import cost.
17. **Remove `exec` from `CanonicalSynth`** — explicit attribute assignment restores ~40 type
    errors' worth of visibility.
18. **Rework `setup_engine()`** to clone into a user cache dir (`platformdirs.user_cache_dir`),
    never `site-packages`; document the `git`/`make`/`perl`/`espeak` system dependencies and
    check for them with a clear error.

### Phase 5 — Surface the strengths

19. **Publish the API docs.** Sphinx + `numpydoc` + `autodoc` on GitHub Pages. The docstrings
    are already exceptional; rendering them costs an afternoon and is the single biggest
    increase in *perceived* quality available here.
20. **Mark the README Roadmap as aspirational**, fix the trailing `:::`, and correct the
    `noisy.py` docstring (it says "pentatonic scale"; it writes noises).
21. **Add `CONTRIBUTING.md`**, a `legacy/` deprecation note, and backfill the changelog.
22. **Delete stale `dist/` artifacts.**

---

## Summary

The distance between this package's **documentation quality** and its **execution quality** is
the whole story. The docstrings are the work of someone who understands the domain deeply and
cares about explaining it. The code shipping underneath them contains three exported functions
that cannot run, two advertised WAV bit depths that do not work, a type checker configured to
inspect 17 % of the code, and no CI to notice any of it.

None of that is hard to fix — Phase 1 is a day's work and closes every "Terrible" item. What
makes it urgent is that **the defects are all in exported, documented API**: a user following
the README hits them immediately, and concludes the package is unreliable, when in fact its
foundations are unusually good.

Fix the six bugs, add CI, and test the fidelity claim. The rest is refinement.
