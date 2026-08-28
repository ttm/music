## [Unreleased]

### Needs a decision before release
Four entries below change behaviour that callers may depend on. All are
deliberate, all are tracked, and none should reach a release unreviewed.

1. **`PlainChanges(n)` now returns a complete peal for every `n`.** The
   default hunt count was hardcoded to two for more than four bells, so the
   peal covered a fraction of the symmetric group above five: 120 of 720 rows
   at six bells, 224 of 40320 at eight — silently. The default is now
   `PlainChanges.saturating_hunts(n)`, i.e. `max(1, n - 3)`, which the code's
   own warning already identified as the point past which extra hunts add
   nothing. Anyone who was relying on the shorter peal can ask for it with
   `nhunts=2`; anyone rendering from `peal_direct` at six bells or more will
   get six times the material or more.
2. **The waveform tables are now exact**, so every rendered note changes
   slightly. Two of the four were built by halves and drifted from the
   waveform they name: the sawtooth stepped by `2/(size-1)` instead of
   `2/size`, and `WAVEFORM_TRIANGULAR` -- the default table for every note
   this package synthesizes -- had a flat two-sample top that peaked at
   `1 - 2/size` rather than 1. Measured against the continuous waveform
   sampled at the same phases, the errors were 1.2e-4 and 2.4e-4; both are
   now zero. A 440 Hz note changes by at most 2.4e-4 (-72 dBFS, inaudible)
   but half its samples differ, so byte comparisons against earlier renders
   will not match.
3. **`fir()` now applies the magnitude response it is given**, which changes
   its output substantially for anyone using the default `freq=True`. The old
   behaviour was a boxcar average scaled by the response rather than the
   response itself; there is no sense in which it was the intended filter, so
   this is listed for visibility rather than as a judgement call.
4. **The WAV quantiser's scale changed**, so files written from now on differ
   from files written before by one LSB of gain (about 0.0003 dB — inaudible,
   but the bytes differ). This is what makes a write/read round trip unity
   gain, and `tests/test_fidelity.py` now pins that property. Anyone
   byte-comparing against previously rendered WAVs will see a diff.

### Fixed
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

