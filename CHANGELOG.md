## [Unreleased]
### Fixed
- **`iir()` was quadratic in the length of the signal.** It rebuilt a reversed
  copy of everything filtered so far on every sample, then threw all but the
  first `len(a)` of it away. One second of audio at 44.1 kHz took about three
  and a half seconds, and ten seconds of audio took over five minutes, which
  made the routine unusable on real material.

  Only the last `len(a)` inputs and `len(b) - 1` outputs are ever read, so the
  slices are now bounded by the filter order. One second of audio takes 192 ms,
  and cost grows linearly, so the gap widens with length.

  **Output values are unchanged, bit for bit.** The slices are multiplied and
  summed rather than dot-producted for exactly this reason: BLAS is free to
  reassociate a dot product, and in a recursive filter that difference
  compounds -- a dot-product version drifted by up to 1.4e-09. Verified
  identical across 405 cases including empty input, filters longer than the
  signal, and a pole at 0.999.

### Changed
- **matplotlib is an optional extra rather than a required dependency.** It was
  imported at the top of `music/tables.py` for `PrimaryTables.draw_tables()`,
  a convenience for looking at the tables, and nothing else in the package uses
  it. Importing it eagerly cost about **40% of `import music`** -- 1237 ms down
  to 743 ms, measured best-of-five -- and pulled contourpy, cycler, fonttools,
  kiwisolver, packaging, pillow, pyparsing and python-dateutil into every
  installation. It is imported inside `draw_tables` now, which raises a message
  naming `pip install 'music[plot]'` when it is absent.
- Development and documentation dependency floors raised to the majors CI
  actually exercises: mypy 2.0, ruff 0.14, sphinx 8.0, numpydoc 1.8. The old
  floors predate rule and default changes in those tools, so a contributor
  could pass locally and fail CI.

### Added
- `iir()` gains a test that pins the recurrence it documents against a plain
  scalar implementation -- including the plus sign on the feedback term, which
  is not the convention `scipy.signal.lfilter` uses -- and one that fails if
  the cost stops being linear.
- A CI job that installs the **exact lower bounds** `pyproject.toml` declares
  -- numpy 1.26.4, scipy 1.12.0, matplotlib 3.7.1, sympy 1.12 -- and runs the
  suite on Python 3.10. Nothing had ever tested them: every other job resolves
  to the newest release, so the floors could drift into fiction without a
  single failure. They currently hold, all 482 tests passing.

- Five fields of Zenodo metadata the deposit had been leaving empty, all
  carried in `.zenodo.json` so they survive every release:
  - **The software block.** `code:codeRepository`, `code:programmingLanguage`
    and `code:developmentStatus` were entirely absent, so nothing told a
    machine reading the record that this is Python.
  - **ROR-identified affiliations.** The author's affiliation was one string
    naming six institutions, which resolves to nothing. Two of them are in
    ROR -- Instituto de Física de São Carlos (`02f1crb75`) and Universidade de
    São Paulo (`036rp1748`) -- and are now linked; the rest stay as names,
    since they have no ROR entry.
  - **A `created` date of 2016-02-20**, the repository's first commit. The
    record's publication date is the release date, which made a decade of work
    look like a day of it.
  - **The MASS article as a reference**, distinct from the related identifier:
    one says the software derives from the article, the other cites it.
  - **The language**, `eng`.

- `tools/zenodo_sync.py` attaches the changelog entry for the record's version
  to the Zenodo deposit as an additional description of type `technical-info`,
  so a version record says what changed in that version and not only what the
  package is. The record's own description is still left alone. It converts
  the subset of markdown the changelog uses -- headings, nested bullets,
  inline code, links, bold -- and holds code spans out of the rest of the
  conversion, because this changelog quotes expressions such as
  `2 ** (bit_depth - 1)` whose asterisks a bold rule would otherwise pair with
  the next ones outside the span and emit tags that cross.

### Fixed
- The README said `structures` held "scales, chords, counterpoint, tunings".
  It holds permutations, peals and symmetry; none of those four exist. The
  sentence now describes what is there and links the issue for what is not.
- `tools/zenodo_sync.py` read the draft it was updating in Zenodo's legacy
  serialization and wrote it back to an API that speaks the other one, where a
  resource type is `{"id": ...}` rather than `{"title": ..., "type": ...}` and
  a relation is an object rather than a string. Writing the wrong shape back
  stripped exactly those fields, and the publish then failed validation on
  them. It now re-reads the draft as `application/vnd.inveniordm.v1+json`.
- A failed publish left a half-written draft on the record. The draft is now
  discarded on failure, so the published record is what it was.

### Changed
- `RELEASING.md` records what the 1.1.1 deposit settled: Zenodo's GitHub
  ingestion reads `.zenodo.json` for the title, the creators, the contributors,
  the description and the free-text keywords, and **ignores the `subjects`
  block**. All nineteen keywords came across; all ten controlled-vocabulary
  terms did not. Running `tools/zenodo_sync.py --write` after a release is
  therefore required rather than precautionary.

## [1.1.1] - 2026-08-29
Documentation and metadata only; no change to any rendered sound. It exists
because the page PyPI shows can only be refreshed by an upload, and the
1.1.0 page predates both the tutorial and the DOI.

### Added
- `.zenodo.json`, so the Zenodo deposit for a release is built from the
  repository rather than from GitHub's guess at it. The 1.1.0 record had to be
  corrected by hand: it had listed the maintainers' GitHub display names as
  authors. This file carries the title, the author, Jacopo Donati as a
  contributor, the description, the keywords and the links to the article, to
  PyPI and to the documentation.
  It also carries the controlled-vocabulary subjects -- the MeSH, GEMET and
  EuroSciVoc terms -- each identified by its vocabulary's own URI, so the
  linked subjects survive a release rather than being retyped into the
  interface each time.
- `tools/zenodo_sync.py`, which pushes `.zenodo.json` onto a published Zenodo
  record through its API -- edit, update, publish, DOI unchanged. It resolves
  each controlled term to the identifier Zenodo knows it by and insists on an
  exact match, so the linked subjects no longer depend on whether the
  release-time ingestion honours them, and a record edited by hand can always
  be brought back in line with the file. Standard library only, and it needs
  no token to show what it would send.
- `tools/release.py`, which checks, builds and publishes a release. The
  version lives in three files that have to agree, the tag has to point at
  the commit the artifacts were built from rather than wherever master has
  since moved to, and PyPI will not release a version number back if any of
  it goes wrong -- so it refuses to proceed on any disagreement, and re-checks
  before it touches anything outside the machine.
- `RELEASING.md`, a checklist, including how to verify what actually landed
  (Zenodo's own record endpoint does not report subjects at all; the DataCite
  export does).
- `Documentation`, `Tutorial`, `Source` and `Changelog` links in the project
  metadata, which PyPI shows in its sidebar. There had been only `Homepage`,
  pointing at the repository, and `Issues`.

### Changed
- `CITATION.cff` matches the curated Zenodo record: the fuller title, a wider
  keyword list, and Renato Fabbri as the sole author, Jacopo Donati having
  been recorded as a contributor. The Citation File Format has no way to
  express a contributor for software, which is why that distinction lives in
  `.zenodo.json`.
- The README carries the DOI badge and links the tutorial.

### Fixed
- `Topic :: Multimedia :: Sound/Audio :: Sound Synthesis` was listed twice in
  the package classifiers.

## [1.1.0] - 2026-08-29
### Note for anyone upgrading
Four of the fixes below change rendered output. All are corrections -- the
previous behaviour was wrong in each case -- but they mean a render from this
version will not be byte-identical to one from 1.0.1:

1. **`PlainChanges(n)` returns a complete peal.** Above five bells the default
   hunt count covered a fraction of the symmetric group: 120 of 720 rows at
   six, 224 of 40320 at eight. Anyone who wants the shorter peal can still ask
   for it with `nhunts=2`.
2. **The waveform tables are exact.** Two of the four had drifted from the
   waveform they name -- the sawtooth stepped by `2/(size-1)` instead of
   `2/size`, and the triangle, the default table for every note, never reached
   full amplitude. A note changes by at most 2.4e-4 (-72 dBFS).
3. **WAV round trips are unity gain.** The writer scaled by
   `2 ** (bit_depth - 1) - 1` while the reader divided by `2 ** (bit_depth -
   1)`, a systematic one-LSB error on every file the package had written.
4. **`fir()` applies the response it is given.** It convolved with the
   magnitudes rather than their inverse transform, so a flat response was a
   boxcar average instead of the identity.

### Fixed
- `normalize_stereo()` corrupted a mono vector instead of rejecting or
  promoting it. A one-dimensional array had its first two *samples* read as
  the two channels, and since the mean of a scalar is itself, both were
  silently zeroed and the rest scaled by the wrong factor. It now gives a
  mono vector two identical channels, so `write_wav_stereo(mono)` produces a
  genuine stereo file rather than a damaged mono one.
- Nine defects that only appeared once every branch was exercised:
  - `pan_transitions()` and `CanonicalSynth.tremoloEnvelope()` truth-tested
    their `sonic_vector`, so passing one raised "the truth value of an array
    with more than one element is ambiguous". Both had a correct
    `is not None` check elsewhere in the same function.
  - `mix_with_offset()`'s stereo path passed `['s1', 's2']` to
    `resolve_stereo`, parameter names that had been renamed, and raised
    `KeyError`.
  - `mix_stereo()` tested for stereo with `len(x) != 2`, so a two-sample
    mono vector was taken for a channel pair and its "channels" indexed as
    scalars.
  - `mix_with_offset_()` rejected tuples, via an exact
    `type(a) not in (np.ndarray, list)` check.
  - `localize2()` raised for every odd-length input: the conjugate mirror
    ran to `max_coef`, which only balances when the length is even.
  - `normalize_mono()` and `normalize_stereo()` returned NaN for any
    constant signal. Only the all-zero case was guarded, and a constant is
    silence once its offset is removed.
  - `PlainChanges.peals` was never populated, so `act_all()` could not run
    on the class that builds two peals.
  - `Being.walk()` raised `UnboundLocalError` for an unrecognised method,
    and `Being.setPar()` was a silent no-op for anything but 'f'.
  - `CanonicalSynth.synthSetup()` left the vibrato and tremolo tables as
    None when the effect was switched off, and both are read
    unconditionally, so turning one off raised `TypeError`. A depth of zero
    already makes the modulation a no-op.
- An exponential position transition across the listener produced NaN audio:
  `start * (end / start) ** curve` has a negative base when the source
  changes sides, and a fractional power of that is not a real number. It now
  raises, naming 'lin' as the method for a path that crosses.
- `fir()` applied a magnitude response by convolving with the magnitudes
  themselves rather than with their inverse transform, so it was not really
  applying the response at all. The decisive case: a *flat* response, meaning
  "pass every frequency unchanged", convolved the signal with a run of ones --
  a boxcar moving average. It now transforms the zero-phase spectrum into an
  impulse response first, so a flat response is the identity and a low-pass
  is about a hundred times more selective than before. This is the default
  path: `freq` defaults to True.
- `iir()` raised `TypeError: can't multiply sequence by non-int` when given
  the "iterable of scalars" its parameters are documented as taking. Only
  numpy arrays worked. Its arithmetic was correct -- a one-pole matches its
  closed form exactly -- so only the input handling changed.
- The waveform lookup tables were three separate implementations of the same
  four definitions -- in `utils`, in `tables.PrimaryTables` and in
  `legacy.tables.Basic` -- and they had drifted: the first built its triangle
  as `hstack((t, t[::-1]))` and the other two as `hstack((t, -t))`, which
  differ at the peak sample. All three now come from
  `music.utils.waveform_table`, and `legacy.tables.Basic` is
  `PrimaryTables` under its historical name.
- Asking `PrimaryTables` for an odd `size` returned tables one sample short,
  the two built from halves having summed to `size - 1`.
- The mono branch of `note_with_vibrato_seq_localization()` had never run.
  Three faults, one behind the other: a per-segment value was assigned over
  the accumulator that the next line appended to, raising `AttributeError`;
  the `durations` parameter was clobbered by a local of the same name, so the
  next iteration indexed into an array; and `extend` was used where the
  stereo branch appends, leaving an inhomogeneous list for `np.prod`. Each
  fix is modelled on the working stereo branch beside it.
- `localize2(method="brute")` computed its buffer size as a float, which
  `np.zeros` refuses. The documented alternative to "ifft" had never worked.
- `music.legacy.pieces.testSong2` bound the *class* `CanonicalSynth` rather
  than an instance, so every call in the module was an unbound method missing
  `self`. The piece could not run at all; it now does.
- `CanonicalSynth.adsrApply()` gave the sustain stage a negative length on
  notes shorter than the attack, decay and release combined -- the same fault
  fixed earlier in `music.adsr`. The stages are compressed proportionally.
- `setup_engine()` cloned the eCantorix engine into the installed package
  directory, which fails on a read-only install, in a container, and for any
  second user of a shared `site-packages`, and which a `pip` upgrade discards.
  It now clones into the user's cache directory, overridable with
  `$MUSIC_ECANTORIX_DIR`. An existing in-package clone is still used, so
  installations set up by an older version keep working.
- The singing module's external dependencies -- `git`, `make`, `perl` and
  `espeak` -- were documented nowhere and checked nowhere. A missing one
  surfaced as a `CalledProcessError` about a path the caller had never seen.
  They are now checked up front, with an error naming what to install.
- `sing()` wrote its `.abc` file before checking anything, so a missing
  engine failed partway through. It now validates and creates the cache
  directory before writing.
- `Peals.transpositions_peal()` built each transposition with
  `Permutation(pair)`, which reads a cycle as an array form:
  `Permutation((0, 1))` is the identity and `Permutation((0, 2))` raises. The
  method could not produce a correct peal for any permutation. Pairs are now
  expanded as cycles over the original domain.
- `Peals.peals` was initialised to a list, but every other use indexes it by
  name, so `transpositions_peal` raised `TypeError`. It is a dict.
- `GenericPeal.act()` and `act_all()` raised `'NoneType' object cannot be
  interpreted as an integer` when no peal had been defined, and gave no hint
  about an unknown peal name. Both now say what is wrong.
- `setup_engine()` returned None when the engine was already present, so its
  return value could not be relied on; it now always returns the path.
- `make_test_song()` had eight notes for seven syllables, and `zip` silently
  dropped the surplus.
- `dist()` raised `IndexError` on the identity permutation, which has an empty
  support. It now returns zero, there being no displaced pair to measure.
- `PlainChanges(2)` and `PlainChanges(3)` warned that "peals are the same if
  there are 2 hunts less", advising the removal of more hunts than existed:
  the threshold was `nelements - 3`, which goes negative below four bells.
- `stretches()` raised `AttributeError` on abandoned scratch code and could
  index one sample past the fragment it was resampling.
- `trill()` raised `TypeError: 'module' object is not callable`. Submodules of
  `music.core.filters` share names with the functions re-exported from them,
  and the filters/synths import cycle bound the module during partial
  initialisation.
- `louds()` raised a broadcasting `ValueError` whenever the envelope was
  shorter than the sonic vector it was applied to.
- `write_wav_mono` / `write_wav_stereo` could not write 8-bit files, and the
  64-bit files they produced could not be read back by `read_wav`. Supported
  depths are now 8, 16 and 32, with the unsigned offset that 8-bit WAV
  requires.
- A scalar `fades` argument to the WAV writers was silently ignored.
- `adsr()` gave its sustain stage a negative length on sounds shorter than the
  attack, decay and release combined; the stages are now compressed
  proportionally. Zero-length stages no longer divide by zero or stretch the
  envelope, and single-sample transitions no longer leak NaN.
- `requires-python` corrected from `>=3.0` to `>=3.10`.
- Docstring markup that broke the API reference. `localize`, `localize2`,
  `mix_with_offset` and `mix_with_offset_` failed to render at all: the first
  two had a `# FIXME: hrtf?` comment inside a `See Also` section and a
  reference to an `hrtf` function that does not exist, the other two pointed
  at `(.functions).mix2`, which is not a resolvable target. A further dozen
  docstrings had malformed reStructuredText -- bullet continuations indented
  under nothing, pseudo-code blocks without a literal marker, `|d|` read as a
  substitution reference, `J_` as a link target, an unindented citation
  continuation in `iir`, and a doctest continuation line missing its `...`.
- Passing a tuple or an ndarray subclass as `sonic_vector` was silently
  ignored by `adsr`, `fade`, `loud`, `louds`, `tremolo`, `tremolos`, `am`,
  `reverb` and `adsr_stereo`: they detected a supplied vector with
  `type(x) in (np.ndarray, list)` and returned a default-duration envelope
  instead of the caller's audio. All sixteen sites now go through
  `music.utils.as_sonic_vector`. The historical scalar `0` sentinel still
  means "not supplied", so existing callers are unaffected.
- The WAV writers rejected a numpy integer scalar as `fades` with an
  `IndexError`, because `np.int64` does not subclass `int`.
- Writing a sonic vector containing NaN or infinity cast it to arbitrary
  integers and wrote them as audio; it now raises `ValueError`.
- `stretches()` raised `ZeroDivisionError` deep in its resample loop when a
  duration was zero; it now rejects non-positive durations up front.
- Writing and reading a WAV back was not unity gain: the writer scaled by
  `2 ** (bit_depth - 1) - 1` while `read_wav` divided by `2 ** (bit_depth
  - 1)`, so a round trip lost about 1.5 quantisation steps rather than the
  half step the quantiser costs. The writer now uses the same scale, and
  exactly representable levels survive a round trip unchanged.

### Changed
- `examples/noisy.py` said it wrote "a pentatonic scale with different
  effects"; it writes noises. Its separator beep is now shaped by an ADSR
  envelope, which is what the separator wanted anyway -- a raw note clicks at
  both ends -- and it writes the same note with and without the envelope so
  the difference can be heard on its own.
- `examples/chromatic_scale.py` gains the generated-scale alternative as a
  comment, starting from the C4 the listed scale starts on rather than from
  A4.
- The README is rewritten for the PyPI landing page it becomes. It opened with
  six shell blocks and no Python: the first code a visitor met was the Roadmap
  at line 128, showing an API that does not exist. It now opens with a working
  example, carries badges and a link to the reference, and gives each of the
  distinctive features -- envelopes, change ringing, spatialisation,
  sequencing, noise colours -- a short worked example. Every code block in it
  is executed as written and produces the files it claims.

  The Roadmap is replaced by *Plans*, listing what the code is actually
  waiting for rather than a sketch of names that were never written. The four
  sections on running the dev toolchain are folded into one *Contributing*
  section below the content a reader came for. Fixed a bullet that swallowed
  the paragraph after it, verified against PyPI's own renderer, and a link
  that pointed `penta_effects` at `chromatic_scale.py`.
- `Peals` inherits `GenericPeal`, which is what `GenericPeal` is for. It holds
  a mapping of named peals -- the model `GenericPeal.act(name, domain)` serves
  -- but did not inherit it, so it could build peals and then had no way to act
  them. `GenericPeal` had looked like dead code as a result: nothing inherited
  it, and `PlainChanges` defines its own `act`. `PlainChanges` still does,
  deliberately, being built from one peal rather than holding several, so its
  `act` takes the domain first.
- `Peals` takes `nelements`, having always passed none through to
  `InterestingPermutations` and so always been four elements -- which also
  fixed the size of the default domain the inherited `act` builds.
  `transpositions_peal` now rejects a permutation of a different size up
  front, rather than letting it fail later inside `act` as a sympy error
  about lengths.
- `localize_linear()` is implemented, and exported again. It positions the
  source at every sample along a straight path between two angles, derives the
  interaural intensity and time differences from that position, and applies
  them -- which is what its body computed but never used. Both cues are
  measured against the nearer ear, as `localize()` measures them against the
  nearer ear of its one fixed position, so a path that stays put reproduces
  `localize()`'s cues exactly and the output keeps its input's length.

  The delay changes by a fraction of a sample between samples, so it is
  applied by cubic Hermite interpolation: rounding to whole samples would
  quantize a smoothly moving source into audible steps, and linear
  interpolation costs about 1.5 dB at 8 kHz where cubic costs 0.3.
- `check_untyped_defs` is on. Most of this package is unannotated, and mypy
  skips the bodies of unannotated functions by default, so it had been
  reporting success while inspecting about a sixth of the code. Turning the
  flag on surfaced 177 errors; all are fixed, and CI now enforces it.
- `CanonicalSynth`, `Being` and `TestSong2` build their attributes by copying
  local variables onto the instance. Three `exec("self.{}={}")` calls did
  that; they are now plain `vars(self).update(...)`, and the attributes each
  class ends up with are declared at class level, so the surface is
  discoverable rather than implicit. `tests/test_legacy.py` pins it.
- Every docstring is numpydoc now. `music.structures`, `music.legacy` and
  `music.tables` used Google style -- `Attributes:`, `Parameters:`, `Returns:`
  with a trailing colon -- across 60 sections in eight files, which numpydoc
  cannot parse; publishing the API reference had worked around it by enabling
  `sphinx.ext.napoleon`. That extension is now removed, and the strict
  (`-W`) documentation build passing without it is the proof the conversion is
  complete: numpydoc alone would otherwise mangle any section left behind.
- `requirements.txt` now installs from `pyproject.toml` rather than repeating
  it. The two had drifted: it pinned `setuptools==69.0.2` (a build tool, not a
  runtime dependency, and the source of three open Dependabot alerts — two
  high) and `percolation==0.2.dev0`, which the package does not import, and
  its `==` pins contradicted pyproject's `>=` ranges.
- `localize_linear()` was never finished — its own body notes the missing
  return statement — and now raises `NotImplementedError` instead of crashing.
  It is no longer re-exported from `music`. Use `localize()` or
  `note_with_doppler()`.
- Module-level `np.random.uniform` and `note()` default arguments moved into
  their function bodies, dropping about 2.4 MB and half the import time.
- Tests import the package the way a user does, via a single root
  `conftest.py`, instead of per-file `sys.path` edits.
- `normalize_mono` / `normalize_stereo`: documented what `remove_bias`
  actually selects between (centre-and-scale versus an affine map onto
  [-1, 1]), and removed a stray duplicate entry from the Returns section.

### Added
- A DOI. Every release is archived on Zenodo from now on;
  [10.5281/zenodo.22151793](https://doi.org/10.5281/zenodo.22151793) resolves
  to the most recent one, and 1.1.0 specifically is
  [10.5281/zenodo.22151794](https://doi.org/10.5281/zenodo.22151794). Both are
  recorded in `CITATION.cff`.
- A [tutorial](https://ttm.github.io/music/tutorial.html) in the documentation,
  walking from a single note to a short stereo piece: what the arrays are, the
  units each parameter is expressed in, and a measurement showing that a
  vibrato's instantaneous frequency really does track the model at every
  sample. Every block on the page was run before it was written down.
- `CITATION.cff`, so GitHub renders a "Cite this repository" button and the
  request the README makes in prose becomes something a tool can act on. It
  names the package as the software and the MASS article as the preferred
  citation, which is what the documentation asks people to cite.
- Sphinx documentation, published to GitHub Pages from `master`. The
  reference groups all 69 exports by role rather than alphabetically, and CI
  builds it with `-W`, so a malformed docstring fails rather than rendering
  wrong. Fixing the docstrings this surfaced is listed under Fixed below.
  The package now uses one docstring style throughout.
- `music/singing/paths.py`, a single source of truth for where the engine
  lives and what it needs. The path was previously computed twice, in
  `bootstrap.py` and in `perform.py`, both at import time.
- Coverage is now 100%, enforced in CI. Reaching it is what surfaced the
  defects listed above: `tests/test_mixing.py`, `test_normalization.py`,
  `test_abc_notation.py`, `test_io_paths.py`, `test_branches.py` and
  `test_remaining_paths.py` cover the alternative branches of parametrised
  routines -- both channel modes, each `method`, each curve -- which is
  where every one of them was hiding.
- `tests/test_filters_response.py`, testing what the FIR and IIR filters do
  to a signal rather than that they return an array: a flat response is the
  identity, a low-pass removes the high band, an impulse response convolves
  as given, a one-pole matches `pole ** n`, and both filters are linear.
  Coverage of `impulse_response.py` went from 14% to 100%.
- `music.utils.waveform_table(kind, size)`, the single generator the tables
  now come from. Each waveform is written directly as a function of phase,
  which is exact at any size, and `tests/test_fidelity.py` asserts that
  against the continuous waveform rather than against whatever the code
  happens to emit.
- `tests/test_legacy.py`, covering the legacy synthesizers, which were the
  least-tested modules in the package. `CanonicalSynth` went from 11% to 97%,
  `IteratorSynth` from 25% to 100% and `testSong2` from 0% to 100%.
- The public API sweep now exercises non-default branches -- both channel
  modes and both localisation methods -- which is where the faults above
  were hiding.
- `tests/test_singing.py`, covering path resolution, the dependency check and
  the failure modes, none of which need the engine itself. It replaces
  `test_bootstrap.py` and `test_perform_failures.py`, which patched module
  internals that no longer exist -- and which had become vacuous: the latter
  passed because a mocked `open` broke `shutil.copy`, not for the reason it
  claimed.
- `tests/test_structures.py`, asserting what campanology and group theory
  actually guarantee about a plain-changes peal: that it visits each of the
  `n!` permutations exactly once, that consecutive rows differ by a single
  *adjacent* transposition — the constraint that makes a peal ringable — and
  that it opens on rounds. Coverage of `structures/peals/plain_changes.py`
  went from 10% to 86%, and of `structures/permutations.py` from 26% to 84%.
- `PlainChanges.saturating_hunts(nelements)`, naming the rule that was
  previously implicit in a warning message.
- `tests/test_fidelity.py`, checking that the synthesized *samples* match the
  equations the docstrings cite, not merely that they have the right shape:
  `note` and `note_with_vibrato` against their closed forms sample for
  sample, the vibrato's semitone span, tremolo and ADSR levels in the
  decibels their parameters are stated in, every noise colour's dB/octave
  slope, `localize`'s interaural delay and amplitude ratio against the ear
  geometry, and WAV round-trip error against the quantiser step.
- `tests/test_public_api.py`, sweeping every export callable with its
  documented defaults, plus a regression test for each defect above.
- GitHub Actions running ruff, mypy and pytest on Python 3.10-3.13. The
  `[tool.mypy]` section no longer pins `python_version`, so mypy targets the
  interpreter it runs under; pinned to 3.11 it could not parse numpy's own
  stubs when run on 3.12+.
- `py.typed`, so the package's type annotations reach consumers.
- `music.__version__`, sourced from package metadata.

## [1.0.1] - 2025-07-21
### Added
- `Sequencer` module for scheduling notes and writing renders.
- `play_audio` utility for quickly previewing sonic vectors.
- `singing_demo` and `binaural_beats` example scripts.
- Mypy configuration and type hints across the package.

### Changed
- WAV reading now detects bit depth automatically.
- Various bug fixes and documentation improvements.

