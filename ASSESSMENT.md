# Quality assessment and known limitations

*A living record, not a point-in-time audit. Last measured **2026-08-31**,
`music` 1.2.1 at commit `be7990a`: 37 modules, 7,431 LOC package + 3,888 LOC
tests, 73 names in the public API.*

The first version of this file graded the repository once, in August 2026,
and was already stale four days later: it reported 125 tests at 61 % coverage
against a package that had 504 at 100 %. A snapshot that nobody updates
misrepresents the code it describes, and it undersells it in exactly the
places recent work improved. So this file is now kept current with the code,
and the section that matters most is **Known limitations** — what the package
does not do, stated by the people who know.

Update it whenever a release changes what is true here. Issue #70 tracks the
practice.

## How this was measured

Every figure below came from running the code, not from reading it.

| Check | Command | Result |
|---|---|---|
| Test suite | `pytest -q` | **504 passed**, 18 s |
| Coverage | `pytest --cov=music --cov-fail-under=100` | **100 %** (2,080 stmts, 0 missed) |
| Type check | `mypy music` | **clean**, 37 files |
| Lint | `ruff check music tests examples tools conftest.py` | **clean** |
| Lint, extended rule set | `ruff check --select ALL music` | 1,554 findings |
| Annotation coverage | AST scan | **31 / 140 functions (22 %)** |
| Docstring coverage | AST scan | **126 / 133 public defs (95 %)** |
| Examples | run all 10 | **9 pass**, 1 needs the external singing engine |
| Public API | `tests/test_public_api.py` | every export callable on its own defaults |

`mypy` runs with `check_untyped_defs = true`, so it inspects function bodies
rather than skipping the unannotated ones — which is most of them. A clean
result here is a real result, not the vacuous one an earlier configuration
produced.

## Where it stands

| Grade | What earns it |
|---|---|
| **Exceptional** | Docstrings and scientific grounding: every routine carries the equation it implements and the article section it comes from |
| **Excellent** | Conceptual architecture; breadth of synthesis primitives; the release and archival process, which is reproducible and produces a citable DOI per version |
| **Very good** | Test suite and its coverage gate; CI across Python 3.10–3.14 including a job pinned to the declared lower bounds |
| **Good** | Curated flat public API; examples; published API reference |
| **Needs work** | Annotation coverage at 22 %; the `legacy/` subpackage; `core/functions.py` not yet reconciled with the MASS reference implementation |

## Known limitations

The point of this file. Nothing here is a surprise defect; all of it is
either documented in the code or tracked in the issue list.

### Unimplemented, and raising rather than pretending

- **`Peals.twenty_all_over` and `Peals.an_eight_and_forty`** raise
  `NotImplementedError`. They are exported and documented; they do not work.
- **`Being.walk`'s `perm-walk` method** was never restored from the code
  this package succeeded.

### Gaps the code names about itself

- **No head-related transfer function.** Both `localize` and `localize2` say
  so in their own notes: the height of a source, and whether it is in front
  of or behind the listener, are cues an HRTF carries and neither models.
  This is the largest genuine gap in the package, and it is research-scale
  work rather than a fix.
- **`core/functions.py` has not been reconciled, routine by routine, with
  the MASS reference implementation.** The package's central claim is
  fidelity to a published framework; until that pass is done, the claim
  rests on the docstrings rather than on a comparison. Issue #67.
- **Rendering is not verified against the mathematics it documents**, only
  against shape and against regressions already found. Issue #67 again;
  issue #76 asks for artifact detection a listener could not catch.

### Scope and dependencies

- **Singing needs an external engine.** `music.singing` drives eCantorix,
  which `setup_engine()` clones into the user's cache directory. Without it,
  `singing_demo.py` is the one example that cannot run. Issue #5 tracks
  doing synthesis natively from per-phoneme spectra.
- **Waveform tables are synthetic only.** No SoundFont or WAV-derived
  tables; issue #3.
- **matplotlib is an extra**, needed only by `PrimaryTables.draw_tables()`.
  Installing without it makes `import music` about 40 % faster.

### Debt that is not breakage

- **Annotation coverage is 22 %.** The package type-checks cleanly with
  bodies inspected, so this is missing documentation of intent rather than
  missing safety.
- **The extended lint set reports 1,554 findings** on `music/`, almost all
  stylistic: 368 missing argument annotations, 286 quote-style, 92 missing
  return annotations. The configured set — `E`, `W`, `F` — is clean. The
  gap between the two is a deliberate choice about which rules earn their
  noise, not an oversight.
- **`legacy/` is 1,110 LOC** kept for `CanonicalSynth`, `IteratorSynth` and
  the `Being` class. It is covered and type-checked, but it is not where new
  work should go.

## No longer true

Items the previous version of this file listed as open, since closed. The
CHANGELOG carries the detail; this is only so the record does not read as
worse than the code.

- **The six defects in exported API** — including two functions that could
  never succeed and a systematic one-LSB gain error on every WAV the package
  had written — are fixed, with `tests/test_fidelity.py` pinning the
  properties that were wrong.
- **`legacy/` type errors**: gone; `mypy music` is clean.
- **Three duplicate waveform table definitions**: `music.legacy.tables.Basic`
  is now an alias for `music.tables.PrimaryTables` rather than a third copy.
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
