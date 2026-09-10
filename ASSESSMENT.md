# Quality assessment and known limitations

*A living record, not a point-in-time audit. Last measured **2026-09-10**,
`music` 1.7.0: 47 modules, 11,752 LOC package + 10,133 LOC tests, 126 names
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

The version in the stamp above is checked too, against `pyproject.toml`,
here and in `RECONCILIATION.md` and `DISCREPANCIES.md`. All three still
said 1.4.0 at 1.7.0, with every number around them correct and current,
because a stamp was the one figure the script never read.

`tests/test_assessment_figures.py` checks the other half of it: that every
figure is still *where* the script looks for it. These checks are regexes
over prose that people rewrite, and one that no longer matches is a check
that no longer runs. The seven needing `pytest` and `ruff` to answer were
built only by a full run, so until that test existed a reflowed table row
could have gone from one release to the next without anyone learning the
figure had stopped being read.

The wall-clock time in the test-suite row is not checked, being a
property of the machine rather than of the package, and neither is the
history in the paragraph above, which quotes figures from when this file
was wrong and has to keep quoting them. The date in the stamp is written
by a full run rather than checked, since a date that is not today is not
drift -- it is a file nothing has had to correct since.

The test-suite row counts what `pytest` collects rather than what passes.
A few tests skip when an optional external resource is absent -- the KEMAR
measurements `music.hrtf` reads, the eCantorix engine `music.singing`
drives -- so the number that passes depends on the machine it ran on, and
a figure that does is not a figure. Everything collected that runs, passes.

## How this was measured

Every figure below came from running the code, not from reading it.

| Check | Command | Result |
|---|---|---|
| Test suite | `pytest -q` | **2548 tests**, 16 s |
| Coverage | `pytest --cov=music --cov-fail-under=100` | **100 %** (2,739 stmts, 0 missed) |
| Type check | `mypy music` | **clean**, 40 files |
| Lint | `ruff check music tests examples tools conftest.py` | **clean** |
| Lint, extended rule set | `ruff check --select ALL music` | 2,112 findings |
| Annotation coverage | AST scan | **101 / 207 functions (49 %)**; 63 / 94 exported (67 %) |
| Docstring coverage | AST scan | **169 / 180 public defs (94 %)** |
| Docstring/signature agreement | `tests/test_docstring_signature.py` | every documented parameter exists, in signature order |
| Docstring cross-references | `tests/test_docstring_references.py` | every name a See Also or an example points at exists |
| MASS reconciliation | `tools/mass_reconcile.py` | **26 of 35 routines sample-exact**; 5 divergent with a stated reason, 4 where the reference does not run |
| Article coverage | `tools/article_coverage.py` | **46 of 47 labelled equations** cited by a test; **all 46** a test could settle |
| Docstring examples | `pytest --doctest-modules` | **62 examples run**, 1 skipped |
| Examples | `python tools/run_examples.py` | **10 pass**, 1 skipped for the external singing engine |
| Public API | `tests/test_public_api.py` | every export callable on its own defaults |
| Rendering artifacts | `tests/test_artifacts.py` | every render swept for clicks and DC offset; the steps that remain are registered with a reason |
| Aliasing | `tests/test_artifacts.py` | a sine strays **1.2e-08** of its energy off the harmonics at any frequency; a sawtooth strays **24 %** at 10 kHz |
| Round-trip noise | `tests/test_artifacts.py` | **48 dB** at 8-bit, **121** at 16, **145** at 24, against what was written |
| Import cost | `import music`, warm, 3.12 | **~185-290 ms**, and no sympy in `sys.modules` |
| Archival subjects | `tools/verify_subjects.py` | **15 of 17** resolve to the term they declare; 2 unconfirmable, EuroSciVoc serving an empty graph |

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
| **Needs work** | Annotation coverage at 49 %; the `legacy/` subpackage |

## Known limitations

The point of this file. Nothing here is a surprise defect; all of it is
either documented in the code or tracked in the issue list.

### Gaps the code names about itself

- **The head-related transfer function is a dataset the user fetches, not
  something the package models.** `localize`, `localize2`,
  `localize_linear` and `spatial_motion` say in their own notes that they
  carry neither the elevation of a source nor whether it is in front or
  behind: they model a head as two points on an axis, so azimuths of 90
  and -90 degrees render the same two channels, and a test asserts that
  they still do.

  `music.hrtf` closes that for callers who want it. `setup_hrtf()` fetches
  Gardner and Martin's KEMAR measurements -- 710 directions, freely
  redistributable, about 1.3 MB -- into the user's cache, `hrir()` reads
  one direction out of them, and `localize_hrtf` convolves a sound with
  the pair. A source in front and one behind differ by 0.37 in the impulse
  response and an elevation of 40 degrees by 0.54 -- measured, but only
  where the measurements are installed. Those tests skip otherwise, so CI
  never runs them: what CI checks is the reading, the two azimuth
  conventions and the nearest-angle search, against a synthetic dataset in
  MIT's layout. The two figures above are the weakest claims in this file,
  and they are weak in a way nothing here can fix without shipping the
  data.

  What it is not is a model. The package computes no transfer function of
  its own, so a direction MIT did not measure is answered with the nearest
  one they did -- ten degrees of elevation, and five of azimuth at best.
  The measurements are of a mannequin, an approximation for any listener
  whose own ears differ. And the geometric routines are untouched, so
  anything that used them before is unchanged.

- **One of the article's equations is not checked, and could not be.**
  `tests/test_article.py`, `tests/test_theory.py`, `tests/test_bonds.py` and
  `tests/test_filter_design.py` check routines against the article's
  numbered equations, citing each by the label its LaTeX source gives it,
  and `tools/article_coverage.py` measures the result across `body.tex`,
  `spectra.tex` and `notesInMusic.tex`: 46 of 47, which is **all 46 that a
  test could settle**. The two left are `eq:intervalos`, the interval
  nomenclature, which is naming rather than sounding and which this package
  does not do; and `eq:vinculos`, which is a schema rather than a formula.
  `music.bonds` is where such a formula would go, and its tests check that
  place rather than a function the article declines to name.
- **Nothing here says the article is right.** The tests establish that the
  package computes what the paper writes. Where the paper and the reference
  implementation disagree with each other, or where the paper disagrees
  with itself, `DISCREPANCIES.md` records which one the package follows and
  why. Six such disagreements are on that list, two of them typographic
  errors in the paper found by reconstructing what it specifies.
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
- **Two archival subjects cannot be confirmed from their identifiers.**
  `tools/verify_subjects.py` resolves each subject in `.zenodo.json`
  against its own vocabulary and compares the label that comes back:
  fourteen MeSH terms and one GEMET concept agree with the file. The two
  EuroSciVoc identifiers answer 200 at both `data.europa.eu` and
  `publications.europa.eu` and return an empty RDF graph with no
  `skos:prefLabel` in it, so nothing confirms them. That is a fact about
  the service rather than evidence the file is wrong, and the tool reports
  it as unconfirmable rather than as an error. This row used to claim that
  every term resolved, on a lookup that had been done by hand and that
  nothing could repeat.
- **Speaking SSTIM is currently a matter of docstrings.** Each stimulus
  names the SSTIM technique it implements and links its IRI, and
  `StimulationSession` borrows that model's vocabulary, but the package
  cannot yet read an `sstim:StimulusSpecification` or emit one. Until it
  can, the correspondence is documented rather than machine-checkable.
  Issue #75.

- **Notes concatenated raw click at the join, and whether they do
  depends on arithmetic rather than on anything musical.** A note ends
  wherever its phase lands and the next one opens at the bottom of its
  wavetable, so `horizontal_stack` of two raw notes steps -- by up to the
  full scale, against a largest step of 0.04 within a note. It happens
  only when the frequency and the duration do not multiply out to a whole
  number of cycles, which is why it is so easy to miss: 440 Hz for a
  quarter second is 110 cycles and joins cleanly, and 443 Hz for the same
  quarter second steps by 1.04. A chromatic scale from 440 Hz is whole
  cycles at the root and mid-cycle nearly everywhere else, so nine of its
  twelve joins clicked in the example the README opens with, which is the
  first code anyone here runs.

  This is what a bare oscillator does rather than a defect, and the
  envelopes are the answer: `adsr` runs a note to silence at both ends
  and the join falls to 1e-4. The README and the tutorial now shape their
  notes before stacking them and say why, and
  `tests/test_artifacts.py` measures both the raw seam and the shaped one,
  so neither can change without someone deciding to change it. What the
  package does not have is a routine that joins notes *without* an
  envelope -- no zero-crossing alignment, no automatic micro-fade at a
  seam. Issue #76.

- **The synthesis aliases, and how much depends on the table and the
  note.** A wavetable is read sample by sample with nothing band-limiting
  it, so every partial above the Nyquist frequency folds back down and
  lands where it does not belong. A sine has one partial and nothing to
  fold: it strays 1.2e-08 of its energy off the fundamental at any
  frequency, which is the table read itself. The rich tables carry
  partials all the way up, and at 10 kHz a **sawtooth has 24 % of its
  energy away from any harmonic of the note being played** -- a
  triangular 1.5 %, a square 19 %. At 1 kHz the same three are 2.7 %,
  0.0016 % and 1.8 %.

  This is what the method costs rather than a defect in it. MASS
  specifies a table read at every sample, and a band-limited table is a
  different instrument. It is measured and pinned so that it cannot
  change unnoticed, and so that anyone rendering high notes from a rich
  table knows what they are getting.

  Worth knowing about the measurement too: it is blind whenever the
  sample rate is a whole multiple of the frequency, since then the folded
  partials land on multiples of the fundamental and cannot be told from
  harmonics. A sawtooth at 100 Hz reads as clean as arithmetic allows and
  is not clean. A test keeps that written down.

- **Writing normalizes, so a file's level is not the render's level.**
  `write_wav_mono` runs `normalize_mono` over everything it is given: a
  passage at a hundredth of full scale and one at twenty-six times it both
  arrive at exactly full scale. The docstring says so. The consequence it
  does not say is that **a piece written a phrase at a time is a piece
  whose dynamics are gone** -- each phrase comes back at the same peak,
  and nothing warns. Stack the phrases and write once, or scale by hand.
  Both tested.

- **A note carries thirteen bits of amplitude, whatever the file says.**
  The default wavetable holds 16,384 entries but only 8,193 distinct
  values, every one a multiple of 1/4096. So a 16-bit file cannot lose
  anything a bare note has -- the round trip measures 121 dB where the
  format's theory says 98 -- and a 24-bit file buys nothing at all.
  Anything that shapes a note afterwards, an envelope or a mix, leaves the
  grid behind; a bare note does not.

- **An offset below one sample is discarded rather than rounded.**
  `mix_with_offset` takes seconds and delays by `int(seconds * rate)`, so
  half a sample is no offset at all, and a caller sweeping an offset finely
  gets a staircase rather than a sweep. Honest for a routine that only
  indexes, but it puts comb filtering and fractional delays out of reach
  this way. Pinned rather than fixed: rounding would be a different
  routine and resampling a much larger one.

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

- **Annotation coverage is 49 %**, and 67 % across the exported API. The
  package type-checks cleanly with bodies inspected, so this is missing
  documentation of intent rather than missing safety. What remains is not
  a matter of typing time: the functions still unannotated are the ones
  whose array parameters are genuinely permissive -- `array_like` really
  does mean lists as well as arrays here, which was checked -- and
  annotating them honestly needs `np.asarray` coercion through the
  bodies rather than a signature edit. Doing it by signature alone
  produced 583 mypy errors and was reverted.
- **The extended lint set reports 2,112 findings** on `music/`, almost all
  stylistic: 345 quote-style, 296 missing argument annotations, 78 missing
  return annotations. The configured set — `E`, `W`, `F` — is clean. The
  gap between the two is a deliberate choice about which rules earn their
  noise, not an oversight.
- **`pan_transitions` accepts a `method` it does not read.** The
  signature offers 'lin', 'circ' and 'exp', and the docstring explains
  what each would do to a cross-fade; the body interpolates linearly
  whatever it is given, so all three render the same ramp. The docstring
  now says so, and a test pins it, so implementing the three laws is a
  deliberate change rather than a discovery. The same routine's legs are
  fixed -- see the changelog.
- **`legacy/` is 1,170 LOC** kept for `CanonicalSynth`, `IteratorSynth` and
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
