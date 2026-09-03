## [Unreleased]
### Note for anyone upgrading
**Nothing that imports from `music` breaks, and nothing renders
differently.** `music.stimulation` became a package rather than a single
module, and every name it exported is still imported from the same
place. `localize_linear` is bit-identical to 1.3.0; what changed there
is the example in its docstring, which was wrong -- see below.

**The dependency list changed**: `scipy` is gone and `soundfile` takes
its place. Nothing in the public API changes shape, but an environment
that installed `music` for scipy's sake will no longer get it.

### Added
- **Value tests for the localization family** (#67, third pass), and a
  finding they turned up. `localize2` is checked against the geometry it
  models: a source on the median plane leaves the channels identical, a
  source at -theta is the source at +theta with the ears swapped, and
  the level difference grows with both frequency and angle, which is the
  IID the code applies.

  These are deliberately not called tests that localization is correct.
  There is no head-related transfer function anywhere in this package,
  and `localize2` says in its own docstring that its calculations are
  "not standard and are only to illustrate the method".

  **`localize2` worked at twice every frequency.** `df`, the spacing
  between FFT bins, was `2 * sample_rate / lambda_l` where it is
  `sample_rate / lambda_l`, so the interaural delay came out at exactly
  twice the ITD computed on the line above it, the level difference was
  the head-shadow of a tone an octave up, and the 4000 Hz crossover
  between the two delay coefficients fired at a true 2000 Hz. Fixed, and
  the realized delay is now the computed one to within a part in a
  million at every frequency tested.

- **`tools/audit_audio_tests.py`**, which classifies every test that
  renders audio by what it asserts about it and names the ones that
  assert nothing. It is how the batches of #67 are chosen: from a list
  rather than from memory. It under-reports rather than flatters -- a
  test doing something the heuristic does not recognise is filed lower
  than it deserves -- which is the right direction for a tool that picks
  work.

- **Value tests for the filters and envelopes** (#67, second pass).
  `loud` is checked against `10 ** ((n/N) ** alpha * dev / 20)` sample
  for sample, where the existing test checked its two endpoints and any
  monotonic curve between them would have passed. `fade` must arrive at
  the decibels it was given, and its fade in must be its fade out
  reversed. `reverb` must decay by the decibels it was given. `stretches`
  must give *each* repeat the duration it asked for -- the existing test
  checked the total, which one segment wrong in each direction would
  satisfy -- and a squeezed repeat must be the whole fragment read
  faster rather than a truncation of it.

  One of these pins behaviour that is easy to lose: `fade`'s last
  `perc` runs linearly to true zero, because a decibel curve never
  reaches zero and a signal cut off at -80 dB still steps to silence,
  which is a click.

- **Value tests for four synthesis routines that had only their shape
  checked** (issue #67, a first pass). `note_with_phase`,
  `note_with_fm`, `note_with_glissando` and `trill` are now checked
  against the equations their docstrings describe, sample for sample
  where the routine is deterministic and spectrally where the claim is
  about pitch.

  The tests they join asserted a length and an amplitude range, which
  silence and white noise both satisfy. `note_with_fm(max_fm_deviation=0)`
  is now required to be a steady tone with nothing at the modulation
  rate; a glissando between equal frequencies is required not to move;
  a trill is required to alternate.

  One test documents something worth knowing about table synthesis:
  writing `(end - start) * samples / (count - 1)` and
  `(end - start) * (samples / (count - 1))` differ in the last bit, and
  since the lookup index is an integer floor, one bit is enough to
  change a sample. The test reproduces the implementation's grouping
  rather than tolerating the difference away.

- **CI runs the examples**, through `tools/run_examples.py`, which also
  runs them locally in one command. Every example is expected to
  complete with a zero exit status unless it is named in the script's
  `SKIP` table with a reason -- only `singing_demo.py` is, because it
  needs the external eCantorix engine -- so a new example is covered the
  moment it is added rather than when someone remembers to list it. Each
  runs in a scratch directory, since they write WAV files next to
  themselves.

  This exists because of a specific failure. Deferring the
  `music.structures` import removed the submodule attribute that three
  examples use, and the break survived the full suite at 100% coverage,
  a clean mypy, a clean ruff and a docs build. Every one of those checks
  looks at the package; none of them looks at a caller, and the examples
  are the only callers this repository has. The script was verified by
  reintroducing that exact break and confirming it fails on the three
  examples and exits non-zero.

- **`Peals.twenty_all_over` and `Peals.an_eight_and_forty` ring.** Both
  raised `NotImplementedError` while being exported and documented. They
  are implemented as the rules Tintinnalogia (1668) states -- the book
  this class already cited as its core reference -- and the tests check
  them against the tables it prints, row for row and in order, rather
  than by counting rows.

  `twenty_all_over` is a rule rather than a table: every bell hunts from
  the lead to the back in turn, which is `n * (n - 1)` changes on any
  number of bells and twenty on five. `an_eight_and_forty` is a
  composition for five, and says so if built for any other number: the
  fifth and fourth are whole hunts taking turns at the lead, and the
  three bells between them ring the plain changes on three -- the same
  six `music.PlainChanges` gives for three elements, which a test also
  checks. It rings until it comes round, which is what ends a peal;
  the forty eight falls out rather than being counted to.

- **`Being.walk`'s `perm-walk` method**, which had been lost with the
  code this package succeeded and raised rather than guessing. What is
  here is a **reconstruction, and its docstring says so**: it is
  `stay(method='perm')` -- the same cycle through `perms` -- with the
  window walking along the grid by `seqsize` for each permutation and
  the pointer left where it walked to. Staying and walking differ by
  whether the ground moves, and nothing else about the two methods
  differs either. Unlike `stay`, it never reads `domain`, because
  honouring a fixed domain is what would make the walk a stay.

- **FLAC, read and written**, through `music.write_audio` and
  `music.read_audio`. The container comes from the extension and the
  channel count from the array, so a caller with a sound and a path no
  longer dispatches on either. `read_wav` is the same function under its
  older name and reads FLAC too; `write_wav_mono` and `write_wav_stereo`
  are unchanged and follow the extension as well.

  FLAC is lossless, and the tests say so in the only way that matters: a
  FLAC round trip is *bit for bit* the WAV round trip at the same depth,
  at 8, 16 and 24 bits, on a file roughly a third the size. That claim is
  worth testing rather than trusting here, because fidelity between the
  model and the samples is the thing this package sells.

  Lossy containers are deliberately not offered, though libsndfile would
  give them for nothing. Discarding what a listener is unlikely to notice
  is the one thing a package whose subject is psychophysical fidelity
  should not do quietly.

  FLAC has no 32-bit form and stores 8-bit signed where WAV stores it
  unsigned. Both are handled, and `bit_depth=32` on a `.flac` path now
  raises a message naming the depths FLAC has, rather than libsndfile's
  "Invalid combination of format, subtype and endian".

- **`music.modulated_noise`**, broadband noise of a chosen colour,
  optionally amplitude-modulated. Unmodulated it is the continuous
  broadband stimulus SSTIM catalogues as `techBroadbandNoise`, the
  vehicle for stochastic resonance and for masking; modulated it is also
  `techAmplitudeModulation`, whose definition names a carrier tone *or
  noise*. The distinction is in the signal and not only in the label: at
  a rate of zero there is no envelope to find, and the test says so.
- **`music.spatial_motion`**, a source orbiting the listener at a chosen
  rate, rendering `sstim-v:techSpatialAuditory`. SSTIM distinguishes
  structured spatial trajectories from simple left/right crossfades, and
  this is on the right side of that line: the interaural time and
  intensity differences are computed per sample from the geometry. It
  will move a sound it did not synthesize, so a noise bed or an already
  rendered stimulus can be given a trajectory.
- **`music.StimulationSession`**, a protocol: phases in order, each with
  its own stimulus, duration and gain, joined by crossfades rather than
  by cuts. **The session lasts exactly the sum of its phase durations.**
  A ramp is taken half from the phase before it and half from the phase
  after, so a transition is centred on the boundary instead of being
  inserted between the phases and stretching the protocol past the
  length its author wrote down -- ten minutes of stimulation stays ten
  minutes. Boundaries are rounded from elapsed time rather than summed
  from per-phase roundings, which is the same drift the phase
  integration carried until 1.3.0 and would have cost a long session
  several samples of length.

  Phases that disagree about channels are reconciled by promoting the
  session to stereo, never by flattening, because flattening is exactly
  what destroys a binaural beat. Crossfades are equal-power by default,
  since two different stimuli are uncorrelated and a linear pair would
  dig a 3 dB hole at every transition; `ramp_shape='linear'` is there
  for the correlated case, where it is the flat one instead.
- **24-bit WAV, read and written.** `bit_depth=24` was a `ValueError`
  because `scipy.io.wavfile` could not write it; libsndfile can, so it now
  sits alongside 8, 16 and 32 in `BIT_DEPTHS` and in the round-trip tests.
  It is the depth most audio work actually wants.
- **An encoding this package cannot normalize is now refused by name.**
  `read_wav` checks the file's declared subtype and raises
  `unsupported WAV encoding: ...`. libsndfile will decode ADPCM and
  companded formats to float quite happily, but those have no full scale
  that this package's normalization is defined against, so being decoded
  is not the same as being supported. The test that covers it writes a
  real ADPCM file rather than mocking a reader's return value, which the
  two tests it replaces had to do.

### Fixed
- **`localize2` placed sounds on the wrong side of the head, in three
  separate ways.** Both of its methods now put the near ear louder
  *and* earlier, for a source given as an angle or as a position.

  - Both `ifft` branches added ``+2*pi*f*itd`` to the far ear, which
    advances it. A delay is ``-2*pi*f*tau``, so the louder ear arrived
    after the quieter one, on both sides and at every frequency. The
    delay had also been wrapped into one period of ``f`` first; a phase
    is periodic in ``2*pi`` already, so the wrap did nothing except
    wrap a positive and a negative value differently, which is how the
    two branches ended up inconsistent with each other.
  - `brute` stopped resynthesizing one bin short of the one carrying
    the energy over its cutoff -- the loudest bin. It rebuilt a 400 Hz
    tone from everything below 400 Hz and returned it peaking at
    298 Hz. Placing a sound should not change its pitch.
  - `brute` chose the delayed ear from `theta` and the amplified ear
    from `theta_`. Those disagree whenever a caller gives a position
    rather than an angle, since `theta` is 0 there: a source on the
    left came out louder in the left ear and earlier in the right.

  Which ear is near was not decided from outside. `localize` and
  `localize_linear` put the right ear at ``+zeta/2``, and `localize2`
  reaches the same convention by its own route -- ``arctan2(-x, y)``,
  with its IID amplifying the left ear for positive theta. A test now
  checks that `localize` and `localize2` place the same source on the
  same side.

- **`trill` ignored `sample_rate`.** The note length and the loop bound
  were hardcoded to 44100 while the argument was declared, documented,
  and passed to `note()` -- so a trill asked for at 22050 Hz rendered two
  seconds of audio for every one requested, at half the note rate. The
  same defect `number_of_samples` had until 1.3.0: an argument honoured
  in one place and ignored in another. Found by writing a test that
  asked what the duration should be, where the old test asked only
  whether some audio came back.

- **`reverb` named the wrong thing when its phases disagreed.** A
  `first_phase_duration` longer than `duration` left two arrays of
  different lengths, and numpy reported a broadcast failure naming two
  sample counts -- which says nothing about the two durations that
  caused it. `reverb(duration=0.1)` hit it on the default first phase of
  0.15 s. It now refuses, naming both durations.

- **`music.structures` stopped resolving as an attribute.** Deferring
  the structures import took the submodule attribute with it, because
  `music.structures` had only ever been bound as a side effect of the
  eager `from .structures import ...`. Three of the examples use
  `music.structures.peals.PlainChanges` and all three broke. Nothing in
  the suite touched it and CI does not run the examples, so it passed
  every check. `__getattr__` now resolves the submodule as well as the
  names, still lazily, and the tests cover both.
- **`gaussian_noise` could not take a fractional duration.** It kept its
  sample count as a float, so `np.random.uniform` was handed `22050.0`
  as a size and raised `TypeError`. Every duration that was not a whole
  number of seconds failed, which is most of the durations anyone would
  ask for. Found by annotating the signature -- the type checker asked
  what `duration * sample_rate` was, and the answer was wrong.

- **`localize_linear`'s example moved nothing.** It read
  `theta1=90, theta2=-90` and called it "a pass from the left to the
  right", but the azimuth in this package is measured from the ear axis:
  0 is the right ear's side and 180 the left, so 90 and -90 are both on
  the median plane, ahead and behind. Both render the same distance to
  both ears, so the example produced two identical channels and no
  movement whatsoever. The example is now `theta1=180, theta2=0`, and
  the convention is stated in the function's notes rather than left to
  be inferred. Anyone who copied that example was writing a mono sound
  into two channels.

  That the two are indistinguishable is not itself a defect -- front
  from back is an HRTF cue and this package has no HRTF, which it says
  in several places. A test now pins it -- ahead and behind
  must render the same two channels -- so that if an HRTF is ever
  added, that test fails and the notice arrives with it.

### Changed
- **Annotations across the exported API**, from 36 % to 56 % of the
  exported functions and 28 % to 37 % overall. The docstrings already
  stated these types in prose, where nothing checked them; the
  annotations state the same thing where mypy does.

  The rest were left deliberately. Their array parameters are genuinely
  permissive -- `array_like` here really does accept lists as well as
  arrays, which was checked rather than assumed -- so annotating them
  honestly means `np.asarray` coercion through the bodies rather than a
  signature edit. Attempted by signature alone it produced 583 mypy
  errors, and was reverted rather than papered over with a narrower type
  that would have rejected calls that work.

- **`import music` no longer imports sympy**, and takes roughly a third
  of the time it did: **~550-830 ms down to ~185-290 ms**, measured warm
  on 3.12. The permutation and change-ringing structures --
  `InterestingPermutations`, `Peals`, `PlainChanges`, `GenericPeal`,
  `dist`, `transpose_permutation` -- are reached through a module-level
  `__getattr__` rather than imported at the top, because importing
  `sympy.combinatorics` runs `sympy/__init__.py` first and drags in
  `sympy.polys` and the rest of the computer algebra system.

  **Nothing about the API changes.** `music.Peals`,
  `from music import Peals`, `help(music.Peals)` and `dir(music)` all
  behave as before, and sympy loads on the first of them; a caller who
  uses peals pays exactly what they paid, and a caller who only
  synthesizes sound stops paying for them. sympy remains a required
  dependency -- this is about when it is imported, not whether it is
  installed. Removing it would break the structures outright and change
  two signatures that take and return sympy `Permutation` objects.

  `tests/test_import_cost.py` pins it, in a subprocess, because the way
  this regresses is silent: one eager `from .structures import ...`
  anywhere puts the cost back with nothing failing.

- **WAV I/O moved from `scipy.io.wavfile` to `soundfile`, and scipy is no
  longer a dependency.** It was a hard requirement carrying 102 MB and
  imported for nothing but reading and writing WAV files -- two imports in
  the whole package, one of them in `music.singing`. Measured on a clean
  install of 3.12: **230 MB down to 133 MB**, and `import music` from
  **~760 ms to ~580 ms**. Nothing in the public API changes shape, and the
  quantisation contract is unchanged: `tests/test_fidelity.py` still pins
  unity gain and the one-step round trip, now at four bit depths instead
  of three.

  `read_wav` got shorter rather than longer. libsndfile divides an integer
  sample by its own full scale on the way to float, which is the
  normalization the function used to compute by hand from the numpy dtype,
  so the bit-depth arithmetic and the 8-bit midpoint correction are gone.
  What is left is the part that is genuinely this package's decision: a
  float WAV declares no full scale, so it is still scaled by its own peak.
- **The MeSH music-therapy subjects are back in `.zenodo.json`**, along
  with `Acoustic Stimulation`. They were removed at 1.2.0 as a claim the
  code did not back; the toolkit backs it now, and the same reasoning
  applies in reverse. `Intended Audience :: Healthcare Industry`, which
  had been in `pyproject.toml` making that claim on its own, is earned
  for the first time. Every MeSH identifier in the file was checked
  against the NLM lookup service and resolves to the term it declares.
- **`HRTF` is no longer a keyword in `pyproject.toml`.** There is no
  head-related transfer function in this package -- `ASSESSMENT.md`
  calls its absence the largest genuine gap -- and a keyword that
  surfaces the package in searches it can only disappoint is the same
  mistake the music-therapy subjects were, in the other direction.
- **`music.stimulation` is a package** rather than a single module, with
  the generators in `stimulation/stimuli.py` and the session in
  `stimulation/session.py`. Import paths are unchanged.
- **`localize_linear` and `spatial_motion` share their localization
  math** through `_localize_positions`, rather than the second one
  carrying a second copy of it. A source held still by either renders
  identically, and a test asserts it.

## [1.3.0] - 2026-09-01
### Note for anyone upgrading
**Nothing that imports from `music` breaks.** Two functions were renamed
for saying what they do -- `mix2` is now `mix_many` and `mix_with_offset_`
is now `mix_many_with_offsets` -- and both old names stay bound to the new
functions.

Two things do render differently, in both cases because they were wrong
before:

- `note_with_vibrato_seq_localization` and `note_with_vibratos_glissandos`
  now honour `number_of_samples`, which they had declared, documented and
  ignored. Code that passed it was getting a sound of whatever length the
  durations summed to; it now gets the length it asked for.
- Phase is integrated without accumulating drift. Of nine routines checked
  at one second, eight are bit-identical to 1.2.1; only long renders and
  fast modulations differ, and always towards the closed-form phase.

New: `music.stimulation`, five auditory stimuli named for the SSTIM
techniques they implement, and the `WAVEFORM_*` tables are now re-exported
from the top level.

### Added
- **The wavetables are re-exported from the top level**: `WAVEFORM_SINE`,
  `WAVEFORM_TRIANGULAR`, `WAVEFORM_SQUARE`, `WAVEFORM_SAWTOOTH`,
  `WAVEFORMS` and the `waveform_table` builder. They are the names that
  already appear in the synthesis signatures -- `note` documents
  `waveform_table=WAVEFORM_TRIANGULAR` -- but `music.WAVEFORM_SINE` raised
  `AttributeError`, so the one family of names a reader meets in every
  signature was the one family the flat API did not carry, and the
  reference explained where they were hiding instead. The obvious call now
  works. Purely additive; `music.utils` still holds them.
- **`music.stimulation`, five auditory stimuli for sensory-stimulation
  work**, each rendering one technique catalogued in SSTIM, the Sensory
  Stimulation Vocabulary developed in the W3C Sensory Stimulation
  Vocabulary Community Group. Every routine names the SSTIM term it
  implements and links its IRI, so a rendered stimulus can be described in
  the same words a protocol, a dataset or a device uses:
  `binaural_beats` (`sstim-v:techBinauralBeats`), `monaural_beats`,
  `isochronic_tones`, `amplitude_modulation` and `frequency_modulation`.

  SSTIM records, per technique, whether a rendering puts a modulation
  physically into the world or whether the listener constructs it, and
  that distinction is carried into the code because it decides what a
  recording of the output contains. Four of the five put a real
  modulation into the air. `binaural_beats` does not: each channel is a
  steady tone, neither contains the beat, and summing the two channels
  does not preserve the stimulus -- it produces an envelope numerically
  identical to `monaural_beats`, a different technique with a different
  mechanism and a different evidence base. Anything that downmixes that
  output has silently substituted one for the other, and the test suite
  holds both halves of this.

  `isochronic_tones` takes a `ramp_duration`. An abrupt gate is what the
  technique names, but it is also a step discontinuity twice per pulse:
  with a carrier and a pulse rate whose ratio is not a whole number, the
  step reaches nearly full scale and puts nine tenths of the energy above
  2 kHz into splatter that is not part of the stimulus. A few
  milliseconds of ramp removes it.

  The modulators are read from a wavetable rather than from `np.sin`, and
  `frequency_modulation` integrates the instantaneous frequency into the
  lookup index rather than modulating a finished tone, which is the
  sample-by-sample model the rest of the package follows. Issue #72;
  issues #73 and #75, which give `music` an IRI in SSTIM and have it
  render SSTIM specifications, both needed this to exist first.
- **`examples/sensory_stimulation.py`**, rendering one file per technique,
  and `examples/binaural_beats.py` rewritten to call the new routine
  instead of hand-rolling it.
- **Python 3.14 to the CI matrix**, which stopped at 3.13 although 3.14 was
  released in October 2025. The matrix now runs 3.10 through 3.14, and the
  separate job that installs the exact lower bounds `pyproject.toml` declares
  is unchanged.
- **Per-version `Programming Language :: Python` classifiers**, and
  `Typing :: Typed`. PyPI's Python-version facet had nothing to filter on:
  the only version classifier was the bare `:: 3`, so the package did not
  appear when anyone narrowed by interpreter version. `Typing :: Typed` was
  simply missing although `music/py.typed` has shipped all along. All 25
  classifiers validate against the trove list, which is what a release's
  metadata check enforces.

### Changed
- **`ASSESSMENT.md` records the phase-integration drift** as a known
  limitation, with the measurements. Issue #102 tracks the fix: the
  wavetable index is integrated with `np.cumsum`, whose error grows with
  render length -- 32 index entries of 16384 over an hour -- where
  compensated summation gives none. Inaudible below a minute, but the
  package claims fidelity to a mathematical model, and long sessions are
  what `music.stimulation` is for.
- **`ASSESSMENT.md` is now a living record of known limitations** rather than
  an audit of one commit. It had been written four days earlier and already
  described a package with 125 tests at 61 % coverage, against the 504 at
  100 % that were actually there -- a document that made the code look
  considerably worse than it was. It now leads with what the package does not
  do, every figure re-measured by running the code, and it drops the
  narration of fixes that `CHANGELOG.md` already carries. Issue #70.

- **`mix2` is now `mix_many`; `mix_with_offset_` is now
  `mix_many_with_offsets`.** Neither old name said what its function did.
  `mix2` was a numeral left from when it was the second mixer rather than
  the general one, and `mix_with_offset_` differed from `mix_with_offset`
  by a trailing underscore, which said nothing about the difference it
  marks: `mix_with_offset` takes two sounds and one offset, while
  `mix_many_with_offsets` takes as many sounds as it is given, each mixed
  into the result built so far at the offset that follows it.

  **Both old names stay bound to the renamed functions**, so nothing that
  imports them breaks. The API reference lists only the descriptive names;
  an alias listed beside its target documents one object twice.

  Three `See Also` entries advertised `mix2` as "a better mixer" without
  saying better at what. They now name what it actually offers over `mix`:
  a list of sounds of any lengths, per-sound offsets, and a choice of
  aligning their starts or their ends.

- `mix_many_with_offsets` no longer carries the commented-out alternative
  implementation it had kept since it was called `J_`, nor the `DEPRECATED`
  marker left behind beside it on a line that is not deprecated.

- The `Parameters` section of that same docstring described the arguments
  under the name `J_`, which nothing has exported since the rename to
  `mix_with_offset_`. The rendered reference documented a function readers
  could not import, and the `TODO` asking whether to "enhance/recycle `J_`
  and mix2 or delete them" is answered by the two renames above rather than
  left standing.

### Fixed
- **Phase integration no longer drifts with the length of the render**
  (issue #102). Every routine that synthesises a varying frequency
  accumulated the wavetable index with `np.cumsum`, whose running total
  keeps growing and so keeps losing low bits against it. The error grew
  with the render and grew in one direction -- drift, not noise. Measured
  against the exact phase for a 200 Hz carrier and a 16384 entry table, it
  reached 0.48 table entries at ten minutes and 32 at an hour, enough to
  change the entry that gets looked up.

  All 14 sites -- 13 in `core/synths/notes.py`, one in `stimulation.py` --
  now go through `_integrate_phase`, which folds the running total into
  one table period as it goes so it never grows, and carries between
  blocks through `ndarray.sum` and its pairwise summation. The error stops
  growing with length: 2.0e-7 at five seconds and 2.1e-7 at a minute,
  where the old way gave 3.6e-6 and 1.3e-2.

  **Almost nothing renders differently.** Of nine routines checked at one
  second, eight are bit-identical to what they produced before; only
  `frequency_modulation` moved, in 6 samples of 44100, by one table entry
  each. Longer renders and faster modulations will differ more, always in
  the direction of the closed-form phase.
- **`note_with_vibrato_seq_localization` and `note_with_vibratos_glissandos`
  declared `number_of_samples`, documented it as "the number of samples of
  the sound", and never read it.** A caller asking for a length got whatever
  the durations happened to sum to instead. Nine other routines in the same
  module take the parameter and honour it, so the two were silently
  inconsistent with their own module's convention.

  A sequence routine has no single `duration` to override -- its length is
  the sum of its segments -- so the two now fit the rendered result to the
  requested length, truncating it or padding it with silence along the last
  axis, which leaves a stereo result stereo.

  Only one of them carried a `FIXME`, and that marker had drifted onto the
  private helper above it when the helper was inserted between the comment
  and the function it described. The other carried nothing at all, so the
  defect read as one instance rather than two.

## [1.2.1] - 2026-08-31
Metadata only; the package itself is byte-identical to 1.2.0. It exists
because Zenodo could not archive 1.2.0, and the fix has to be inside the tag
Zenodo reads -- moving the 1.2.0 tag would have separated it from the sdist
PyPI was built from, which is the one thing the release procedure guarantees.

### Fixed
- **`.zenodo.json` failed Zenodo's release-time validator**, so 1.2.0 was
  published to PyPI and GitHub but never archived and never given a DOI.
  Zenodo answered the webhook with `202` both times and recorded the failure
  only on the repository's own page in its settings, where it reads
  `Extra metadata load failed` -- invisible from GitHub and from the API.

  The `dates` entry added in 1.2.0 for the repository's first commit used the
  REST API's shape, `{"date": ..., "type": "created"}`, in a file Zenodo
  validates against its *legacy* deposit schema, which wants
  `{"start": ..., "type": "Created"}`. Posting the file to Zenodo's own
  validator names it exactly: `metadata.dates`, "Invalid date provided."
  Nothing else in the file was wrong.

### Added
- `tests/test_zenodo_metadata.py`, which checks statically what that validator
  would have said: only keys Zenodo knows, dates in the ingestion's shape,
  names written family-first, and subjects carrying a scheme and a resolvable
  identifier. A malformed field there does not degrade a deposit, it fails the
  archive, and nothing else catches it until a release has already gone out.

## [1.2.0] - 2026-08-30
### Note for anyone upgrading
**matplotlib is no longer installed with the package.** If your code calls
`PrimaryTables.draw_tables()`, or relied on `import music` having pulled
matplotlib in for you, install `music[plot]` instead of `music`. Nothing else
in the package uses it. Everything that synthesises, filters or writes audio
is unaffected.

`iir()` returns exactly the values it did before, bit for bit; it is only
faster. `iir()` and `fir()` now raise instead of accepting a stereo array --
`iir()` used to return two samples of nonsense for one. Filter each channel
separately.

### Added
- A CI job that installs the **exact lower bounds** `pyproject.toml` declares
  -- numpy 1.26.4, scipy 1.12.0, matplotlib 3.7.1, sympy 1.12 -- and runs the
  suite on Python 3.10. Nothing had ever tested them: every other job resolves
  to the newest release, so the floors could drift into fiction without a
  single failure. They currently hold, with the whole suite passing.
- `tools/zenodo_sync.py` attaches the changelog entry for the record's version
  to the Zenodo deposit as an additional description of type `technical-info`,
  so a version record says what changed in that version and not only what the
  package is. The record's own description is still left alone. It converts
  the subset of markdown the changelog uses -- headings, nested bullets,
  inline code, links, bold -- and holds code spans out of the rest of the
  conversion, because this changelog quotes expressions such as
  `2 ** (bit_depth - 1)` whose asterisks a bold rule would otherwise pair with
  the next ones outside the span and emit tags that cross.
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
- `iir()` gains a test that pins the recurrence it documents against a plain
  scalar implementation -- including the plus sign on the feedback term, which
  is not the convention `scipy.signal.lfilter` uses -- and one that fails if
  the cost stops being linear.

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
- `RELEASING.md` records what the 1.1.1 deposit settled: Zenodo's GitHub
  ingestion reads `.zenodo.json` for the title, the creators, the contributors,
  the description and the free-text keywords, and **ignores the `subjects`
  block**. All nineteen keywords came across; all ten controlled-vocabulary
  terms did not. Running `tools/zenodo_sync.py --write` after a release is
  therefore required rather than precautionary.

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
- **`iir()` returned two samples of nonsense for a stereo signal.** `len()` of
  a two-dimensional array counts channels, so a `(2, n)` input produced a
  two-element result rather than a filtered one -- silently. It raises now,
  as it also does for an empty `b` (which raised `IndexError` from the
  divisor) and for `b[0] == 0` (which produced an array of infinities behind a
  `RuntimeWarning`).
- **`translate_to_abc()` silently dropped notes.** It zipped pitches against
  durations, so the tail of whichever was longer vanished: five notes with
  three durations produced a three-note score, with no error. `write_abc()`
  appends the lyric line separately, so the words then pointed at notes that
  were no longer there. It raises now, naming both counts.
- `fir()` rejected bad input with numpy's own messages -- "object too deep for
  desired array" for a stereo array, "v cannot be empty" for an empty one --
  which name neither the argument nor the problem. It now names both, matching
  the guards `iir()` grew.
- The README said `structures` held "scales, chords, counterpoint, tunings".
  It holds permutations, peals and symmetry; none of those four exist. The
  sentence now describes what is there and links the issue for what is not.
- `Notes.make_dict()` built 96 note names and zipped them against 85 MIDI
  numbers, leaving eleven quietly unused. The slice is explicit now, so the
  range the dictionary covers is a decision rather than an accident of `zip`.
- A doctest in `PrimaryTables`'s class docstring called `draw_tables()` without
  a skip marker, so enabling `--doctest-modules` hung the suite on a plot
  window rather than failing it. All three examples in that module are marked
  now.
- `CITATION.cff` named version 1.1.1 while its version DOI still pointed at
  the 1.1.0 archive. The DOI can only be known after a release, so keeping it
  current is now a documented release step rather than something anyone
  remembers.
- `markdown_to_html` emitted a `<ul>` directly inside a `<ul>`, plus a stray
  `</li>`, for a list whose first item was already indented.
- `tools/release.py` carried a second copy of the changelog parser, with the
  same end-of-file bug fixed above in `zenodo_sync`. It imports the one
  implementation now, so the notes on a GitHub release and on a Zenodo deposit
  cannot disagree.
- `tools/zenodo_sync.py` could not find the changelog entry for the *oldest*
  release: its lookahead required a following heading, so the last section in
  the file never matched and its notes silently failed to attach.
- `tools/zenodo_sync.py` read the draft it was updating in Zenodo's legacy
  serialization and wrote it back to an API that speaks the other one, where a
  resource type is `{"id": ...}` rather than `{"title": ..., "type": ...}` and
  a relation is an object rather than a string. Writing the wrong shape back
  stripped exactly those fields, and the publish then failed validation on
  them. It now re-reads the draft as `application/vnd.inveniordm.v1+json`.
- A failed publish left a half-written draft on the record. The draft is now
  discarded on failure, so the published record is what it was.

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

