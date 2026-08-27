## [Unreleased]

### Needs a decision before release
Three entries below change behaviour that callers may depend on. All are
deliberate, all are tracked, and none should reach a release unreviewed.

1. **`localize_linear()` is no longer exported** and raises
   `NotImplementedError`. It never worked — it crashed on its own defaults and
   its body records that the return statement was never written — so nothing
   can regress, but it is an API removal. The open question is whether to
   finish it instead: the missing piece is how the *time-varying* interaural
   time difference should be applied per sample. `localize()` handles a static
   position and `note_with_doppler()` a moving source, so the semantics wanted
   here are a design decision for the author, not something to infer.
2. **`PlainChanges(n)` now returns a complete peal for every `n`.** The
   default hunt count was hardcoded to two for more than four bells, so the
   peal covered a fraction of the symmetric group above five: 120 of 720 rows
   at six bells, 224 of 40320 at eight — silently. The default is now
   `PlainChanges.saturating_hunts(n)`, i.e. `max(1, n - 3)`, which the code's
   own warning already identified as the point past which extra hunts add
   nothing. Anyone who was relying on the shorter peal can ask for it with
   `nhunts=2`; anyone rendering from `peal_direct` at six bells or more will
   get six times the material or more.
3. **The WAV quantiser's scale changed**, so files written from now on differ
   from files written before by one LSB of gain (about 0.0003 dB — inaudible,
   but the bytes differ). This is what makes a write/read round trip unity
   gain, and `tests/test_fidelity.py` now pins that property. Anyone
   byte-comparing against previously rendered WAVs will see a diff.

### Fixed
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
- Sphinx documentation, published to GitHub Pages from `master`. The
  reference groups all 69 exports by role rather than alphabetically, and CI
  builds it with `-W`, so a malformed docstring fails rather than rendering
  wrong. Fixing the docstrings this surfaced is listed under Fixed below.
  The package now uses one docstring style throughout.
- `music/singing/paths.py`, a single source of truth for where the engine
  lives and what it needs. The path was previously computed twice, in
  `bootstrap.py` and in `perform.py`, both at import time.
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

