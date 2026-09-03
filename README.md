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
scale = [music.note(440 * 2 ** (i / 12), duration=0.25) for i in range(13)]
music.write_wav_mono(music.horizontal_stack(*scale), "scale.wav")
```

📖 **[Tutorial](https://ttm.github.io/music/tutorial.html)** — from a single
note to a short stereo piece.
📖 **[API reference](https://ttm.github.io/music/)** — every routine documented
with the equation it implements and the article it comes from.

## Core features

* **Sample-based synthesis.** State is updated at every sample. A note with a
  vibrato has a different instantaneous frequency at each of its samples, and
  the vibrato pattern is folded into the wavetable lookup rather than applied
  afterwards, so the rendered sound is as close as it can be to the
  mathematical model that describes it.
* **Musical structures** with an emphasis on symmetry and discourse:
  permutation groups, change-ringing peals and plain changes.
* **Sensory stimulation.** Seven auditory stimuli -- binaural, monaural and
  isochronic beats, amplitude and frequency modulation, modulated noise and
  spatial motion -- each named for the technique it implements in
  [SSTIM](https://w3id.org/sstim), the Sensory Stimulation Vocabulary, and a
  `StimulationSession` that renders a protocol of them: phases in order,
  crossfaded rather than cut, lasting exactly the sum of the durations you
  wrote down. The sample-accurate synthesis is the point here, because the
  frequency difference *is* the stimulus.
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

One thing is optional. `PrimaryTables.draw_tables()`, which plots the waveform
tables so you can look at them, needs matplotlib:

```console
pip install 'music[plot]'
```

Nothing else in the package uses it, and leaving it out makes `import music`
about 40% faster.

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

## Examples

Inside [the examples folder](https://github.com/ttm/music/tree/master/examples) you can find some scripts that use the main features of Music.

* [chromatic_scale](https://github.com/ttm/music/tree/master/examples/chromatic_scale.py): writes twelve notes into a WAV file from a sequence of frequencies.
* [penta_effects](https://github.com/ttm/music/tree/master/examples/penta_effects.py): writes a pentatonic scale repeated once clean, once with pitch, one with vibrato, one with Doppler, and one with FM, into a WAV stereo file.
* [noisy](https://github.com/ttm/music/tree/master/examples/noisy.py): writes into a WAV file a sequence of different noises.
* [thirty_notes](https://github.com/ttm/music/tree/master/examples/thirty_notes.py) and [thirty_numpy_notes](https://github.com/ttm/music/tree/master/examples/thirty_numpy_notes.py) generate a sequence of sounds by using a synth class (in this case the class [`Being`](https://github.com/ttm/music/tree/master/music/legacy/classes.py)).
* [campanology](https://github.com/ttm/music/tree/master/examples/campanology.py) and [geometric_music](https://github.com/ttm/music/tree/master/examples/geometric_music.py) both use `Being` as their synth, but this time with permutations.
* [isynth](https://github.com/ttm/music/tree/master/examples/isynth.py) also uses a synth class, but of a different kind, [`IteratorSynth`](https://github.com/ttm/music/tree/master/music/legacy/classes.py), that iterates through arbitrary lists of variables.
* [singing_demo](https://github.com/ttm/music/tree/master/examples/singing_demo.py): demonstrates `music.singing.setup_engine()` and `music.singing.make_test_song()` to render a short sung phrase.
* [binaural_beats](https://github.com/ttm/music/tree/master/examples/binaural_beats.py): generates binaural beats using two pure tones with tremolo for relaxation or focus.
* [sensory_stimulation](https://github.com/ttm/music/tree/master/examples/sensory_stimulation.py): writes one file per SSTIM technique with `music.stimulation`, and one three-phase session, which is the form these stimuli are actually delivered in.
* The `music.singing` module provides basic text-to-speech utilities. Run `music.singing.setup_engine()` once to clone the [eCantorix](https://github.com/ttm/ecantorix) engine before using these features. It is cloned into your user cache directory; set `MUSIC_ECANTORIX_DIR` to put it elsewhere. Because eCantorix is a Perl program driving espeak through a Makefile, it also needs `git`, `make`, `perl` and `espeak` installed on the system — `setup_engine()` will tell you which are missing.

## Package structure

The modules are:

* **core**:
  * **synths** for synthesization of notes (including vibratos, glissandos, etc.), noises and envelopes.
  * **filters** for the application of filters such as ADSR envelopes, fades, IIR and FIR, reverb, loudness, and localization.
  * **io** for reading, writing and playing audio, both mono and stereo.
  * **functions** for normalization.
* **structures** for higher level musical structures: permutations and the algebraic groups they form, change-ringing peals, and symmetry. Scales, chords, counterpoint and tunings are [not there yet](https://github.com/ttm/music/issues/1).
* **legacy** for musical pieces that are rendered with the Music package and might be used as material to make more music.
* **stimulation** for sensory-stimulation work: the seven stimuli above, each carrying the SSTIM term it implements, and `StimulationSession` for sequencing them into a protocol.
* **tables** for the generation of lookup tables for some basic waveform.
* **utils** for various functions regarding conversions, mix, etc.
* **sequencer** for scheduling notes into a timeline and exporting audio.

## Plans

Concrete things the code itself is waiting for, rather than a wish list:

* **A head-related transfer function.** Both `localize` and `localize2` say so
  in their own notes: the height of a source, and whether it is in front of or
  behind the listener, are cues an HRTF carries and neither of them models.
* **The remaining peals.** `Peals.twenty_all_over` and
  `Peals.an_eight_and_forty` raise `NotImplementedError`, and `Being.walk`'s
  `perm-walk` method was never restored from its predecessor.
* **Reconciling `core/functions.py` with the MASS reference implementation**,
  routine by routine.
* **An article describing the package**, as a companion to the MASS one.

## Contributing

The test, type-check, lint and documentation tooling comes with the `dev` and
`docs` extras:

```console
pip install -e '.[dev,docs]'
```

```console
pytest                                       # 868 tests, 100% coverage
mypy music                                   # type check
ruff check music tests examples tools conftest.py  # lint, at PEP 8's 79 columns
sphinx-build -b html -W docs docs/_build/html
python tools/run_examples.py                 # run every example
```

All five run in CI on Python 3.10 through 3.14 for every push and pull
request, and both `pytest` and `sphinx-build` are configured to fail on
anything less than full coverage or a docstring numpydoc cannot parse.

The last one is there because the other four look at the package and none
of them looks at a caller. The examples are the only callers this
repository has, and a change that broke three of them once passed every
other check.

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
