# Music

[![PyPI](https://img.shields.io/pypi/v/music.svg)](https://pypi.org/project/music/)
[![Python versions](https://img.shields.io/pypi/pyversions/music.svg)](https://pypi.org/project/music/)
[![CI](https://github.com/ttm/music/actions/workflows/ci.yml/badge.svg)](https://github.com/ttm/music/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-ttm.github.io%2Fmusic-blue.svg)](https://ttm.github.io/music/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/ttm/music/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22151793.svg)](https://doi.org/10.5281/zenodo.22151793)

**Extreme-fidelity synthesis of musical elements.**

Music generates and manipulates sound in LPCM audio, sample by sample. It
implements [MASS (Music and Audio in Sample Sequences)](https://github.com/ttm/mass/),
a collection of psychophysical descriptions of musical elements expressed as
equations and corresponding Python routines.

```python
import music

# a chromatic scale, written to a WAV file
notes = [music.note(440 * 2 ** (i / 12), duration=0.25) for i in range(13)]
scale = [music.adsr(sonic_vector=n) for n in notes]   # so the joins are silent
music.write_wav_mono(music.horizontal_stack(*scale), "scale.wav")
```

The `adsr` is not decoration. A note ends wherever its phase lands and the
next one opens at the bottom of its wavetable, so concatenating them raw
steps -- by up to the full scale, and on nine of this scale's twelve joins.
An envelope runs each note to silence at both ends, which is what makes the
joins joins rather than clicks.

📖 **[Tutorial](https://ttm.github.io/music/tutorial.html)** — from a single
note to a short stereo piece.
📖 **[Where to start](https://ttm.github.io/music/choosing_an_api.html)** — the
few routines a common task needs.
📖 **[API reference](https://ttm.github.io/music/)** — every routine documented
with the equation it implements and the article it comes from.

## Core features

* **Sample-based synthesis.** State is updated at every sample. A note with a
  vibrato has a different instantaneous frequency at each of its samples, and
  the vibrato pattern is folded into the wavetable lookup rather than applied
  afterwards, so the rendered sound is as close as it can be to the
  mathematical model that describes it.
* **Band-limited synthesis, when you ask for it.** A wavetable read at
  every sample folds the partials above the Nyquist frequency back into
  the band -- a 10 kHz sawtooth puts 24 % of its energy between its own
  harmonics -- and `note` keeps that, as MASS specifies.
  `bandlimited_note` renders a fixed pitch without them,
  `render_oversampled` renders any generator at several times the rate and
  filters it, for FM, glissandi and hard gates, and `music.core.synths`
  has oscillators for a pitch that moves.
* **Musical structures** with an emphasis on symmetry and discourse:
  permutation groups, change-ringing peals and plain changes.
* **Music theory** as the MASS companion paper states it: the seven diatonic
  modes and the single step pattern each is a rotation of, the three minor
  scales, the triads and the tetrads that extend them, the intervals and
  their traditional names, and the harmonic series. All in semitones from
  a tonic of zero, so a scale becomes frequencies and then sound in two
  steps.
* **Filter design.** `iir` applies coefficients; `low_pass`, `high_pass`,
  `band_pass` and `band_reject` compute them, from the four designs the
  article specifies.
* **Head-related transfer functions.** `setup_hrtf()` fetches the KEMAR
  measurements into your cache and `hrir()` reads a direction out of them,
  so a source can be placed above you or behind you -- cues the geometric
  `localize` cannot carry. The package ships no measurements and models no
  head; it reads someone else's.
* **Bonds between a note's characteristics**, so a piece decides once how its
  notes behave rather than note by note -- a vibrato that speeds up as the
  line rises, a tremolo that only appears above middle C.
* **Sensory stimulation.** Seven auditory stimuli -- binaural, monaural and
  isochronic beats, amplitude and frequency modulation, modulated noise and
  spatial motion -- each named for the technique it implements in
  [SSTIM](https://w3id.org/sstim), the Sensory Stimulation Vocabulary, and a
  `StimulationSession` that renders a protocol of them: phases in order,
  crossfaded rather than cut, lasting exactly the sum of the durations you
  wrote down. The sample-accurate synthesis is the point here, because the
  frequency difference *is* the stimulus. Each stimulus, and a session,
  can be written as an SSTIM description and rendered back from one, and a
  description another tool wrote is rendered from SSTIM's own terms, given
  only what they cannot say.
* **Singing.** `sing` sings a lyric to a melody, one syllable a note, in
  any language espeak speaks. The package's own singer has espeak-ng say
  each syllable and Praat's PSOLA hold it at its note's pitch: a syllable
  shorter than its note is said again more slowly, only the middle of its
  vowel is lengthened so the consonants keep the length they were said
  with, a long note gets a vibrato, and the strong beats of the meter are
  sung louder. The
  [eCantorix](https://github.com/ttm/ecantorix) engine sings the same score
  as a second backend, kept as the reference, and both sing its tremolo,
  melted and flite voices.
* **`play_audio`** to listen to a result without saving a file.

Music can be used alone or with other packages, and it is well suited to the
audiovisualization of data. It works with
[Percolation](https://github.com/ttm/percolation) and
[Participation](https://github.com/ttm/participation) for harnessing open
linked social data, and with the [audiovisual analytics vocabulary and ontology
(AAVO)](https://github.com/ttm/aavo).

To understand the routines further, read
[Musical elements in the discrete-time representation of sound](https://github.com/ttm/mass/raw/master/doc/article.pdf).
**If you use this package, please cite that article.**

Every release is archived on Zenodo, so a specific version can be cited too:
[10.5281/zenodo.22151793](https://doi.org/10.5281/zenodo.22151793) always
resolves to the newest one. GitHub's *Cite this repository* button reads
[CITATION.cff](https://github.com/ttm/music/blob/master/CITATION.cff) and
gives you both, formatted.

## How to install

```console
pip install music
```

Requires Python 3.10 or newer. Everything needed to synthesise, filter and
write audio comes with it; the dependencies are declared in
[pyproject.toml](https://github.com/ttm/music/blob/master/pyproject.toml).

Two things are optional. `PrimaryTables.draw_tables()`, which plots the
waveform tables so you can look at them, needs matplotlib:

```console
pip install 'music[plot]'
```

Nothing else in the package uses it, and leaving it out makes `import music`
about 40% faster.

`sing` needs espeak-ng, a system program, and Praat, through
praat-parselmouth:

```console
sudo apt install espeak-ng        # or: brew install espeak-ng
pip install 'music[singing]'
```

Its second backend, eCantorix, is a Perl program the package clones for
you; see [Singing](#singing) for what it needs.

`render_oversampled` needs SciPy, and writing stimuli as SSTIM
descriptions needs RDFLib and SSTIM's own validator:

```console
pip install 'music[antialias]'
pip install 'music[sstim]'
```

To hack on it, install from a checkout so your edits take effect immediately:

```console
git clone https://github.com/ttm/music.git
pip install -e music
```

## A closer look

Every routine returns a numpy array of PCM samples, so results compose with
each other and with anything else you can express in numpy.

### Notes and envelopes

```python
note = music.note_with_vibrato(freq=220, duration=2,
                               vibrato_freq=6, max_pitch_dev=0.5)
shaped = music.adsr(sonic_vector=note, attack_duration=80,
                    sustain_level=-6, release_duration=200)
```

Durations are in seconds, envelope stages in milliseconds, levels in decibels
and pitch deviations in semitones — each parameter in the unit it is usually
thought about in.

### Band-limited synthesis

`note` reads its table as MASS does, so a rich waveform's partials above
the Nyquist frequency fold back into the band. `bandlimited_note` drops
them for the note's pitch, and `render_oversampled` renders a whole
generator at a multiple of the rate, for what one pitch's table cannot
help:

```python
folded = music.note(10000, duration=1,
                    waveform_table=music.waveform_table("sawtooth"))
clean = music.bandlimited_note(10000, duration=1, waveform="sawtooth")
swept = music.render_oversampled(music.note_with_fm, duration=1,
                                 freq=6000, fm=300, max_fm_deviation=4000)
```

The sawtooth that kept 24 % of its energy off its harmonics keeps about
1e-16. [Band-limited synthesis](https://ttm.github.io/music/bandlimited_synthesis.html)
has the oscillators for a pitch that moves, by partials or with PolyBLEP,
and what each is measured to do.

### Scales, chords and intervals

Everything is counted in semitones from a tonic of zero, which is what
`pitch_to_freq` takes:

```python
freqs = music.pitch_to_freq(220.0, music.scale("dorian"))
chord = music.pitch_to_freq(220.0, music.chord("minor seventh"))
music.write_wav_mono(music.mix_many(
    [music.note(f, duration=1.5) for f in chord]), "chord.wav")
```

The seven modes are one step pattern read from seven places, and
`mode_by_rotation` reaches them that way rather than by name. Intervals
can be written the way a musician writes them:

```python
music.interval("M3"), music.interval("P11")     # (4, 17)
music.interval_names(6)                         # ('aug4', 'dim5', 'TT')
music.consonance("P4")                          # 'context dependent'
```

### Filters, designed

Cutoffs are fractions of the sample rate — `fraction_of` converts Hertz —
so the same coefficients describe the same filter at any rate:

```python
a, b = music.low_pass(music.fraction_of(1000))
muffled = music.iir(music.noise("white", duration=2), a, b)
```

### Bonds

A vibrato whose rate follows the pitch, and a depth that falls as it rises:

```python
voice = music.Bonds(vibrato_freq=music.proportional(1 / 40),
                    max_pitch_dev=music.inversely_proportional(400))
music.write_wav_mono(voice.render([220, 277, 330, 440], duration=0.5),
                     "bound.wav")
```

### Placing a source with measured ears

`localize` puts a source on the ear axis from its geometry alone, so it
cannot tell front from back. A measured response can:

```python
music.setup_hrtf()                       # once; about 1.3 MB into your cache
_, _, left, right = music.hrir(elevation=30, azimuth=270)   # above, behind
music.write_wav_stereo(music.localize_hrtf(music.note(duration=2),
                                           left, right), "behind.wav")
```

### Change ringing

Permutation groups and the peals of campanology, acted on any domain you like
— here on frequencies, so the peal *is* the melody:

```python
peal = music.PlainChanges(4)                      # every permutation, once
rows = peal.act([220, 275, 330, 440])
notes = [music.note(freq, duration=0.2) for row in rows for freq in row]
music.write_wav_mono(music.horizontal_stack(*notes), "campanology.wav")
```

### Spatialisation

A source moving from one side to the other, its interaural time and intensity
differences computed at every sample from its position:

```python
passing = music.localize_linear(music.note(330, duration=3),
                                theta1=150, theta2=30, dist=0.6)
music.write_wav_stereo(passing, "passing.wav")
```

### Sequencing

```python
seq = music.Sequencer()
for i, freq in enumerate([440, 550, 660]):
    seq.add_note(freq, start=i * 0.25, duration=1.0,
                 adsr_params={"attack_duration": 20, "release_duration": 400})
seq.write("chord.wav")
```

### Noise

Six colours, each defined by its gain per octave — brown at −6 dB, pink at −3,
white at 0, blue at +3, violet at +6, black at −12 — or any number you pass
instead:

```python
colours = [music.noise(kind, duration=0.5)
           for kind in ("brown", "pink", "white", "blue", "violet")]
music.write_wav_mono(music.horizontal_stack(*colours), "colours.wav")
```

A `seed` draws the same noise again, and leaves NumPy's global random
state alone.

### Sensory stimulation

Each stimulus is named for the SSTIM technique it implements, and a
session plays them in order, crossfaded:

```python
session = music.StimulationSession(end_ramp=0.5)
session.add(music.binaural_beats, duration=30, carrier_freq=200,
            beat_freq=10)
session.add(music.modulated_noise, duration=30, ramp=2,
            noise_type="pink", seed=1)
music.write_wav_stereo(session.render(), "session.wav")
```

With `pip install 'music[sstim]'`, the session can be written as SSTIM
descriptions of its stimuli, in the order the session plays them, and
rendered back from them sample for sample:

```python
from music.stimulation.sstim import render_sstim, to_sstim_graph

graph = to_sstim_graph(session, base="https://your-lab.example/plans/1/")
graph.serialize("session.ttl", format="turtle")
again = render_sstim("session.ttl")
```

A description says what a stimulus is, not that anyone heard it or that it
does anything. [Stimuli in SSTIM](https://ttm.github.io/music/sstim.html)
has the rest: a session written as SSTIM's own plan, a description another
tool wrote rendered from SSTIM's terms with only what they cannot say
given, and what can be rendered from a graph before anything is.

### Singing

A lyric is sung one syllable a note: syllables joined by hyphens, words
parted by spaces, each note a pitch in semitones above middle C and a
duration in quarter notes, at 120 a minute unless `L` and `Q` say otherwise,
as ABC notation writes them:

```python
twinkle = music.sing(text="Twin-kle, twin-kle, lit-tle star",
                     notes=(0, 0, 7, 7, 9, 9, 7), durs=(1, 1, 1, 1, 1, 1, 2))
music.write_wav_mono(twinkle, "twinkle.wav")

# in French, trembling in a room: stereo, ringing a second past the line
frere = music.sing(text="Frè-re Jac-ques, dor-mez vous?",
                   notes=(0, 2, 4, 0, 4, 5, 7), durs=(1, 1, 1, 1, 1, 1, 2),
                   lang="fr", effect="tremolo")
music.write_wav_stereo(frere, "frere.wav")

# the same score with eCantorix, the reference: same pitches, same lengths
reference = music.sing(text="Twin-kle, twin-kle, lit-tle star",
                       notes=(0, 0, 7, 7, 9, 9, 7),
                       durs=(1, 1, 1, 1, 1, 1, 2), backend="ecantorix")
```

The default backend, `psola`, is the package's own. It needs espeak-ng and
`pip install 'music[singing]'`. espeak-ng says each syllable, and Praat's
pitch-synchronous overlap-add holds it at its note's pitch and length. A
syllable said in less time than its note is said again, down to 80 words a
minute, as eCantorix has espeak say it; then only the middle of its vowel is
lengthened, so the consonants keep their length. A long note gets a vibrato,
0.35 semitones each way at 5.5 Hz, setting in a quarter of a second into it.
The notes on the meter's strong beats are sung up to 2.3 dB louder, the
accents abc2midi gives eCantorix. With this timing, Whisper, a speech
recogniser, heard 69 of 192 sung test words, against 44 with the plain
stretch it replaced and 59 with eCantorix; the vibrato and the accents left
that where it was. `effect` takes eCantorix's `"tremolo"`, `"melt"`
and `"flite"`, made with the package's own tremolo, reverberation and
resampling.

`backend="ecantorix"` sings with the [eCantorix](https://github.com/ttm/ecantorix)
engine. It was the default until the package had its own singer, and is
kept as the reference. Run `music.singing.setup_engine()` once to clone it,
at the revision this package is pinned to, into your user cache directory
(set `MUSIC_ECANTORIX_DIR` to put it elsewhere). It drives espeak through a
Makefile, so it also needs `git`, `make`, `perl`, `espeak`, `abc2midi` (the
`abcmidi` package) and `sox` on the system. It needs the Perl modules
`MIDI`, `Math::FFT`, `URI::Escape` and `Digest::SHA` too
(`cpan MIDI Math::FFT URI::Escape`). `setup_engine()` says which of these
are missing.

To compare the two:
[singing_backends](https://github.com/ttm/music/tree/master/examples/singing_backends.py)
and [singing_effects](https://github.com/ttm/music/tree/master/examples/singing_effects.py)
sing the same songs and effects with each, side by side.
`python tools/compare_singing.py` measures both against the score: the pitch
of every note, the length and the render time.
`tools/compare_singing_timing.py` renders the singer's timings and syllable
shapes for a level-matched listening review, and
`tools/score_singing_asr.py` has Whisper count the words it hears in each.

## Examples

Inside [the examples folder](https://github.com/ttm/music/tree/master/examples) you can find some scripts that use the main features of Music.

* [chromatic_scale](https://github.com/ttm/music/tree/master/examples/chromatic_scale.py): writes twelve notes into a WAV file from a sequence of frequencies.
* [penta_effects](https://github.com/ttm/music/tree/master/examples/penta_effects.py): writes a pentatonic scale repeated once clean, once with pitch, one with vibrato, one with Doppler, and one with FM, into a WAV stereo file.
* [noisy](https://github.com/ttm/music/tree/master/examples/noisy.py): writes into a WAV file a sequence of different noises.
* [thirty_notes](https://github.com/ttm/music/tree/master/examples/thirty_notes.py) and [thirty_numpy_notes](https://github.com/ttm/music/tree/master/examples/thirty_numpy_notes.py) generate a sequence of sounds by using a synth class (in this case the class [`Being`](https://github.com/ttm/music/tree/master/music/legacy/classes.py)).
* [campanology](https://github.com/ttm/music/tree/master/examples/campanology.py) and [geometric_music](https://github.com/ttm/music/tree/master/examples/geometric_music.py) both use `Being` as their synth, but this time with permutations.
* [isynth](https://github.com/ttm/music/tree/master/examples/isynth.py) also uses a synth class, but of a different kind, [`IteratorSynth`](https://github.com/ttm/music/tree/master/music/legacy/classes.py), that iterates through arbitrary lists of variables.
* [singing_demo](https://github.com/ttm/music/tree/master/examples/singing_demo.py): sings Mary had a little lamb with `music.sing`, and writes it to a WAV file.
* [singing_backends](https://github.com/ttm/music/tree/master/examples/singing_backends.py): sings four songs, in English, French and German, and one slowly, with both singing backends, a file each, to be heard side by side.
* [singing_effects](https://github.com/ttm/music/tree/master/examples/singing_effects.py): sings eCantorix's tremolo, melted and flite voices with both backends.
* [scales_and_chords](https://github.com/ttm/music/tree/master/examples/scales_and_chords.py): renders the seven modes, a I–vi–IV–V7 cadence, and sixteen partials of the harmonic series over their own fundamental, where the octaves land on the tempered scale and nothing else does.
* [filtered_noise](https://github.com/ttm/music/tree/master/examples/filtered_noise.py): puts each of the four filter designs on white noise — the useful thing to hear a filter on — and sweeps a low pass across five octaves.
* [bonds](https://github.com/ttm/music/tree/master/examples/bonds.py): plays one line three times, changing only how its characteristics are bound to its pitch.
* [binaural_beats](https://github.com/ttm/music/tree/master/examples/binaural_beats.py): generates binaural beats using two pure tones with tremolo for relaxation or focus.
* [sensory_stimulation](https://github.com/ttm/music/tree/master/examples/sensory_stimulation.py): writes one file per SSTIM technique with `music.stimulation`, and one three-phase session, which is the form these stimuli are actually delivered in.
* [sstim_descriptions](https://github.com/ttm/music/tree/master/examples/sstim_descriptions.py): writes a stimulus and a planned session as SSTIM descriptions, renders both back, and renders an amplitude modulation another tool might have described, in SSTIM's terms alone.
* [aliasing_demo](https://github.com/ttm/music/tree/master/examples/aliasing_demo.py): writes a 10 kHz sawtooth twice, read from the table as MASS does and band-limited, at one gain, so the folding can be heard.
* [Singing](#singing), above, says what each singing backend needs.

## Package structure

The modules are:

* **core**:
  * **synths** for synthesization of notes (including vibratos, glissandos, etc.), noises and envelopes, and the band-limited notes and oscillators.
  * **filters** for the application of filters such as ADSR envelopes, fades, IIR and FIR, reverb, loudness, and localization, and for designing the coefficients the IIR ones take.
  * **io** for reading, writing and playing audio, both mono and stereo.
  * **functions** for normalization.
* **structures** for higher level musical structures: permutations and the algebraic groups they form, change-ringing peals, and symmetry.
* **theory** for scales, chords and the harmonic series. Counterpoint, harmonic expansion and modulation are described in the companion paper and are [not there yet](https://github.com/ttm/music/issues/1).
* **bonds** for tying a note's vibrato and tremolo to its frequency, which is where an arbitrary construction of that kind goes.
* **legacy** for musical pieces that are rendered with the Music package and might be used as material to make more music.
* **stimulation** for sensory-stimulation work: the seven stimuli above, each carrying the SSTIM term it implements, `StimulationSession` for sequencing them into a protocol, and `music.stimulation.sstim`, which writes them as SSTIM descriptions, reads them back, and reads what other tools write.
* **singing** for `sing` and its two backends: `psola`, the package's own singer, and `perform`, which writes the score and runs the eCantorix engine that `bootstrap` clones and `paths` finds.
* **hrtf** for fetching the KEMAR measurements into your cache and reading a direction's impulse responses out of them.
* **tables** for the generation of lookup tables for some basic waveform.
* **utils** for various functions regarding conversions, mix, etc.
* **sequencer** for scheduling notes into a timeline and exporting audio.

## Plans

See [the findings and prioritized roadmap](ROADMAP.md) for the September
2026 review, the correctness patches since 1.8.1, the
[mutation audits](MUTATION_AUDIT.md) of all sixteen areas of the package as
it stood at 1.10.0 and of the SSTIM package since, and the band-limited
synthesis and SSTIM work since.

Concrete things the code itself is waiting for, rather than a wish list:

* **A head-related transfer function this package computes.** `music.hrtf`
  reads Gardner and Martin's KEMAR measurements, which closes the gap for
  anyone willing to fetch them, but the package models no head of its own:
  a direction nobody measured is answered with the nearest one somebody
  did. (`localize` and `localize2` still carry neither elevation nor
  front-against-back, and say so in their own notes.)
* **`Being.walk`'s `perm-walk` method**, which was never restored from its
  predecessor and is currently a reconstruction. (`Peals.twenty_all_over` and
  `Peals.an_eight_and_forty` no longer raise; this entry used to say they did.)
* **Counterpoint, harmonic expansion and modulation**, which the companion
  paper describes and `music.theory` does not implement.
* **Checking the routines MASS has no counterpart for.**
  [`RECONCILIATION.md`](RECONCILIATION.md) compares the package with the MASS
  reference implementation routine by routine — 22 of 35 are reproduced sample
  for sample, 9 diverge for reasons it states, and in 4 the reference itself
  does not run.
  [`tests/test_article.py`](tests/test_article.py) checks the article's
  equations instead of its code, and covers 46 of 47 labelled equations:
  all 46 that a test could settle. Routines without a MASS counterpart
  also have signal-property and composition tests, and the mutation audits
  measure, one area at a time, whether the assertions detect deliberate
  implementation errors; the areas not yet audited are the next step.
  [`DISCREPANCIES.md`](DISCREPANCIES.md) is where the article, the reference
  implementation and this package are recorded as disagreeing.
* **An article describing the package**, as a companion to the MASS one.

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) is the longer version: what a change has
to survive, why a test that checks the shape of the output is worth little
here, and how to report something that sounds wrong. Participation is under
the [Code of Conduct](CODE_OF_CONDUCT.md).

The test, type-check, lint and documentation tooling comes with the `dev` and
`docs` extras:

```console
pip install -e '.[dev,docs]'
```

```console
pytest                                       # 6,631 tests, 100% line and branch coverage
mypy music                                   # type check
ruff check music tests examples tools conftest.py  # lint, at PEP 8's 79 columns
sphinx-build -b html -W docs docs/_build/html
python tools/run_examples.py                 # run every example
python tools/assessment_figures.py           # the docs' figures vs the package
python -m build --wheel
python tools/check_wheel.py --wheel dist/*.whl # installed artifact, outside the checkout
```

These checks run on every push and every pull request. Lint, types and
tests run on **Python 3.10 through 3.14 on Linux**; a further Linux job
installs the exact lower bounds `pyproject.toml` declares and runs the
tests against those, so the floors cannot drift into fiction. The docs
build, examples and figures run once each. Installed wheels are checked
on **Linux, macOS and Windows with Python 3.12**. CI runs
pytest with `--cov=music --cov-branch --cov-fail-under=100`; Sphinx treats
documentation warnings as errors.

`run_examples.py` exercises the code readers copy. A change that broke
three examples once passed every package check. `assessment_figures.py`
is there because the numbers in
`ASSESSMENT.md` and this file went stale four times in two days when
keeping them current was a habit rather than a check.

Four more run at release time rather than on every push, being slower or
needing something the runner does not have:

```console
python tools/check_sdist.py         # build the sdist, unpack it, run its tests
python tools/article_coverage.py --strict # fail missing sources or test citations
python tools/mass_reconcile.py      # this package against the MASS reference
python tools/verify_subjects.py --strict # fail unexpected unresolved subjects
```

The last three need a [MASS](https://github.com/ttm/mass) checkout or the
network. MASS comparisons are documented in `RECONCILIATION.md` and
`DISCREPANCIES.md`; two known EuroSciVoc exceptions remain explicitly
unverified. Run the complete workflow during development with
`python tools/release.py verify --mass ../mass`. See
[RELEASING.md](RELEASING.md) for prerequisites and release behavior.

Docstrings are [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html)
style throughout, and the code follows
[PEP 8](https://peps.python.org/pep-0008/). For the maths behind a routine,
examples of its use, and the article it comes from, read its docstring — or
the rendered [API reference](https://ttm.github.io/music/).

## Support

`music` has been developed and maintained in the open since 2016. If it is
useful to you, your research or your institution, please consider supporting
its continued development through
[GitHub Sponsors](https://github.com/sponsors/ttm).

Sponsorship pays for the unglamorous work that makes a scientific package
trustworthy — the fidelity tests, the full coverage, the documented
equations, the archived and citable releases — and keeps every bit of it
free for everyone.

**For institutions and companies:** commissioned features, integration
support and sponsored development are available, with the results released
under the same open license. Open an issue or get in touch to discuss scope.

## Further information

Music is primarily intended for artistic use, psychophysics experiments and data sonification.

You can find an example in [Versinus](https://github.com/ttm/versinus), an animated visualization method for evolving networks that uses Music to render the musical track that represents networks structures.
