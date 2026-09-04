# Quality assessment and known limitations

*A living record, not a point-in-time audit. Last measured **2026-09-04**,
`music` 1.4.0: 40 modules, 9,596 LOC package + 6,668 LOC tests, 90 names
in the public API.*

The first version of this file graded the repository once, in August 2026,
and was already stale four days later: it reported 125 tests at 61 % coverage
against a package that had 504 at 100 %. A snapshot that nobody updates
misrepresents the code it describes, and it undersells it in exactly the
places recent work improved. So this file is now kept current with the code,
and the section that matters most is **Known limitations** — what the package
does not do, stated by the people who know.

Keeping it current is no longer a practice anyone has to remember.
`tools/assessment_figures.py` measures every number below and fails when
the file disagrees with the package; CI runs the cheap half of it on every
push and the release gate runs all of it, so a release cannot go out
describing a version that no longer exists. Issue #70 asked for the
practice; this is the check that replaced it.

```console
python tools/assessment_figures.py           # report any drift
python tools/assessment_figures.py --write   # correct it
```

The wall-clock time in the test-suite row is not checked, being a
property of the machine rather than of the package, and neither is the
history in the paragraph above, which quotes figures from when this file
was wrong and has to keep quoting them.

## How this was measured

Every figure below came from running the code, not from reading it.

| Check | Command | Result |
|---|---|---|
| Test suite | `pytest -q` | **1579 passed**, 16 s |
| Coverage | `pytest --cov=music --cov-fail-under=100` | **100 %** (2,420 stmts, 0 missed) |
| Type check | `mypy music` | **clean**, 40 files |
| Lint | `ruff check music tests examples tools conftest.py` | **clean** |
| Lint, extended rule set | `ruff check --select ALL music` | 1,682 findings |
| Annotation coverage | AST scan | **62 / 168 functions (37 %)**; 39 / 70 exported (56 %) |
| Docstring coverage | AST scan | **140 / 148 public defs (95 %)** |
| Docstring/signature agreement | `tests/test_docstring_signature.py` | every documented parameter exists, in signature order |
| Docstring cross-references | `tests/test_docstring_references.py` | every name a See Also or an example points at exists |
| MASS reconciliation | `tools/mass_reconcile.py` | **26 of 35 routines sample-exact**; 5 divergent with a stated reason, 4 where the reference does not run |
| Examples | `python tools/run_examples.py` | **10 pass**, 1 skipped for the external singing engine |
| Public API | `tests/test_public_api.py` | every export callable on its own defaults |
| Import cost | `import music`, warm, 3.12 | **~185-290 ms**, and no sympy in `sys.modules` |
| Archival subjects | NLM MeSH lookup, per identifier | every term in `.zenodo.json` resolves to the term it declares |

`mypy` runs with `check_untyped_defs = true`, so it inspects function bodies
rather than skipping the unannotated ones — which is most of them. A clean
result here is a real result, not the vacuous one an earlier configuration
produced.

## Where it stands

| Grade | What earns it |
|---|---|
| **Exceptional** | Docstrings and scientific grounding: every routine carries the equation it implements and the article section it comes from, and `RECONCILIATION.md` now measures the correspondence routine by routine rather than asserting it |
| **Excellent** | Conceptual architecture; breadth of synthesis primitives; the release and archival process, which is reproducible and produces a citable DOI per version |
| **Very good** | Test suite and its coverage gate; CI across Python 3.10–3.14 including a job pinned to the declared lower bounds |
| **Good** | Curated flat public API; examples; published API reference; the sensory-stimulation toolkit, whose stimuli are each tested against the property that defines them rather than against their shape |
| **Needs work** | Annotation coverage at 37 %; the `legacy/` subpackage |

## Known limitations

The point of this file. Nothing here is a surprise defect; all of it is
either documented in the code or tracked in the issue list.

### Gaps the code names about itself

- **No head-related transfer function.** `localize`, `localize2`,
  `localize_linear` and `spatial_motion` all say so in their own notes: the
  height of a source, and whether it is in front of or behind the listener,
  are cues an HRTF carries and none of them models. Concretely, azimuths of
  90° and −90° — ahead and behind — render the same two channels, and a test
  now asserts that they do, so adding an HRTF will announce itself by
  breaking it. This is the largest genuine gap in the package, and it is
  research-scale work rather than a fix.
- **The reconciliation compares this package with the reference
  implementation, not with the article.** `RECONCILIATION.md` establishes
  that 26 of the reference's 35 routines are reproduced sample for sample
  and that the other nine differ for stated reasons. Where the two differ,
  the reason argues which is right; that argument is a comment, not a
  proof. `tests/test_fidelity.py` is the file that checks routines against
  the closed-form expressions the article states, and it does not cover
  every routine.
- **Rendering is not verified against the mathematics it documents** for
  the routines outside both files, only against shape and against
  regressions already found. Issue #76 asks for artifact detection a
  listener could not catch.

### Claims the metadata makes that the code cannot

- **The stimulation toolkit renders stimuli; it does not demonstrate that
  they do anything.** Every routine in `music.stimulation` is tested against
  the property that defines it — a binaural beat has no beat in either
  channel, an isochronic train gates at the rate asked for, an
  amplitude-modulated carrier puts its envelope where it said it would — and
  those tests are about the signal, which is all a synthesis library can
  speak to. Whether a 10 Hz stimulus entrains anything in a listener is a
  question for the literature the SSTIM terms point at, not for this
  repository. The `Music Therapy` subjects in `.zenodo.json` and the
  `Intended Audience :: Healthcare Industry` classifier say what the package
  is *for*; they are not evidence of efficacy, and nothing here should be
  read as clinical.
- **Speaking SSTIM is currently a matter of docstrings.** Each stimulus
  names the SSTIM technique it implements and links its IRI, and
  `StimulationSession` borrows that model's vocabulary, but the package
  cannot yet read an `sstim:StimulusSpecification` or emit one. Until it
  can, the correspondence is documented rather than machine-checkable.
  Issue #75.

### Scope and dependencies

- **Singing needs an external engine.** `music.singing` drives eCantorix,
  which `setup_engine()` clones into the user's cache directory. Without it,
  `singing_demo.py` is the one example that cannot run. Issue #5 tracks
  doing synthesis natively from per-phoneme spectra.
- **Waveform tables are synthetic only.** No SoundFont or WAV-derived
  tables; issue #3.
- **matplotlib is an extra**, needed only by `PrimaryTables.draw_tables()`.
  Installing without it makes `import music` about 40 % faster.
- **sympy is required, but no longer imported at `import music`.** The
  permutation and change-ringing structures need it and cannot be written
  without reimplementing a permutation group; two exported signatures take
  and return sympy `Permutation` objects, so it is part of the public API
  rather than an implementation detail. It is now reached through a
  module-level `__getattr__`, so only callers who touch those structures
  pay the roughly 400 ms it costs to import. It remains the largest single
  dependency at 73 MB.

### Debt that is not breakage

- **Annotation coverage is 37 %**, and 56 % across the exported API. The
  package type-checks cleanly with bodies inspected, so this is missing
  documentation of intent rather than missing safety. What remains is not
  a matter of typing time: the functions still unannotated are the ones
  whose array parameters are genuinely permissive -- `array_like` really
  does mean lists as well as arrays here, which was checked -- and
  annotating them honestly needs `np.asarray` coercion through the
  bodies rather than a signature edit. Doing it by signature alone
  produced 583 mypy errors and was reverted.
- **The extended lint set reports 1,682 findings** on `music/`, almost all
  stylistic: 345 quote-style, 296 missing argument annotations, 78 missing
  return annotations. The configured set — `E`, `W`, `F` — is clean. The
  gap between the two is a deliberate choice about which rules earn their
  noise, not an oversight.
- **`legacy/` is 1,151 LOC** kept for `CanonicalSynth`, `IteratorSynth` and
  the `Being` class. It is covered and type-checked, but it is not where new
  work should go.

## No longer true

Items the previous version of this file listed as open, since closed. The
CHANGELOG carries the detail; this is only so the record does not read as
worse than the code.

- **`core/functions.py` was never the file the claim rested on.** This
  entry, and the roadmap in `README.md`, named a 123-line file holding three
  routines. The reference is `src/aux/functions.py` in ttm/mass: 2,997 lines
  and 35 routines, which map onto the whole of `music/core/` and parts of
  `music/utils.py`. The line came from `notes.md`, written before the split
  into `synths/` and `filters/` that the same file's next bullet proposed.
  `RECONCILIATION.md` is the comparison that entry was asking for, and
  `tools/mass_reconcile.py` fails when the register in it disagrees with
  what it measures. Issue #67.
- **Two exported routines multiplied their frequency contour by the wrong
  thing.** `note_with_vibratos_glissandos` and
  `note_with_vibrato_seq_localization` had collapsed the reference's two
  accumulators into one name, so each vibrato discarded the one before it
  and each appended its own concatenation back into the list it was
  concatenating. Both returned an array of the expected length with 99.9 %
  of its samples wrong, which is why the suite never noticed. Found by the
  reconciliation, and now covered by it.
- **Ninety-two docstring cross-references pointed at names that do not
  exist.** `note` said `See Also: V, T` and its example called `H` — MASS's
  names for `note_with_vibrato`, `tremolo` and `horizontal_stack`, none of
  which this package exports. Eight more examples had lost the `...` prompt
  on their continuation lines and were not parseable Python, and three
  called routines with the reference's parameter names. All corrected, with
  `tests/test_docstring_references.py` failing on any of them.
- **`music.profile` raised `NotImplementedError`.** Its body had been a
  commented-out sketch since it was written, so it was first an exported,
  documented function returning `None` while its docstring described a
  dictionary, and then one that raised. The docstring is now a description:
  the function sorts a namespace by what its names hold, measures every
  array in it, and reads each array as PCM samples or as parametrisation.
  Measurement and inference are kept in separate keys, and every guess
  carries the reason that produced it, because the rules the specification
  gave are heuristics and saying so is cheaper than being wrong quietly.
- **Three names reached `music.` that the package neither documents nor
  owns.** `typing.Any`, `typing.TYPE_CHECKING` and
  `importlib.metadata.PackageNotFoundError` were imported unaliased into
  `__init__.py`, so `music.Any` resolved, in an API this file calls
  curated. They are private aliases now, and `tests/test_public_api.py`
  fails on any name in the flat namespace that is neither in `__all__` nor
  a submodule.
- **The six defects in exported API** — including two functions that could
  never succeed and a systematic one-LSB gain error on every WAV the package
  had written — are fixed, with `tests/test_fidelity.py` pinning the
  properties that were wrong.
- **`legacy/` type errors**: gone; `mypy music` is clean.
- **Phase integration drifting on long renders**: the wavetable index was
  accumulated with `np.cumsum`, whose error grew with the render and grew
  in one direction -- 32 table entries of 16384 over an hour. All 14 sites
  now fold the running total into one table period as they go, and the
  error no longer grows with length: 2.0e-7 at five seconds and 2.1e-7 at
  a minute, against 3.6e-6 and 1.3e-2 before. Issue #102.
- **Three duplicate waveform table definitions**: `music.legacy.tables.Basic`
  is now an alias for `music.tables.PrimaryTables` rather than a third copy.
- **Two exported routines documented parameters that did not exist.**
  `note_with_vibrato` said `max_pitch_deviation` for `max_pitch_dev`, and
  `note_with_two_vibratos` said `secondary_vibrato_waveform_table` for
  `sec_vibrato_waveform_table`, so code written from the reference raised
  `TypeError`. Twenty-four defects of that family were fixed, and
  `tests/test_docstring_signature.py` now fails on any of them.
- **The two named peals, and `Being.walk`'s `perm-walk`.** All three
  raised `NotImplementedError`. `twenty_all_over` and
  `an_eight_and_forty` are implemented from the rules Tintinnalogia
  states and checked against the tables it prints; `perm-walk` is a
  reconstruction, and says so.
- **`gaussian_noise` could not take a fractional duration**, having kept
  its sample count as a float. Found by annotating it.
- **`localize_linear`'s worked example moved nothing.** It passed
  `theta1=90, theta2=-90` and called it a pass from the left to the right,
  but this package measures azimuth from the ear axis, so both angles sit on
  the median plane and the example rendered two identical channels. Anyone
  who copied it got a mono sound in a stereo file. The example is corrected
  and the convention is now stated rather than left to be inferred — an
  instance of exactly what issue #67 exists to find, caught by writing a
  test that expected the documented behaviour and getting silence.
- **`setup_engine()` writing into `site-packages`**: it uses the user's cache
  directory, which survives an upgrade and works on a read-only install.
- **No published API docs**: they are at <https://ttm.github.io/music/>,
  built with warnings as errors on every push.
- **No CI**: lint, types, tests and docs run on Python 3.10 through 3.14, plus
  a job that installs the exact lower bounds `pyproject.toml` declares.

## Is anything perfect?

Still no, and the same file is still closest. `music/core/filters/reverb.py`
is 76 lines at 100 % coverage, fully documented, clean under both the default
lint set and the type checker. It is also small enough that saying so proves
little — which is the honest version of the compliment.
