# Music

Music is a python package to generate and manipulate music and sounds. It's written using the [MASS (Music and Audio in Sample Sequences)](https://github.com/ttm/mass/) framework, a collection of psychophysical descriptions of musical elements in LPCM audio through equations and corresponding Python routines.

To have a further understanding of the routines you can read the article
[Musical elements in the discrete-time representation of sound](https://github.com/ttm/mass/raw/master/doc/article.pdf).

If you use this package, please cite the forementioned article.

## Core features

The precision of Music makes it the perfect choice for many scientific uses. At its core there are a few important features:

* **Sample-based synth**, meaning that the state is updated at each sample.  For example, when we have a note with a vibrato, each sample is associated to a different frequency. By doing this the synthesized sound is the closest it can be to the mathematical model that describes it.
* **Musical structures** with emphasis in symmetry and discourse.
* **Speech and singing interfaces** available to synthesize voices.

Music can be used alone or with other packages, and it's ideal for audiovisualization of data. For example, it can be used with [Percolation](https://github.com/ttm/percolation) and [Participation](https://github.com/ttm/participation) for harnessing open linked social data, or with [audiovisual analytics vocabulary and ontology (AAVO)](https://github.com/ttm/aavo).

## How to install

To install music you can either install it directly with `pip`:

```console
pip3 install music
```

or you can clone this repository and install it from there:

```console
git clone https://github.com/ttm/music.git
pip3 install -e <path_to_repo>
```

This install method is especially useful when reloading the modified module in subsequent runs of music, and for greater control of customization, hacking and debugging.

### Dependencies

Every dependency is installed by default by `pip`, but you can take a look at [requirements.txt](https://github.com/ttm/music/blob/master/requirements.txt).

To use the singing interface you'll also nead eSpeak, SoX, and abcMIDI, while MIDI.pm and FFT.pm are needed by eCantorix to synthesize singing sequences.

#### Linux

On Ubuntu everything can be installed with:

```console
sudo apt install espeak sox abcmidi
sudo cpan install MIDI
sudo cpan install Math::FFT
```

<!-- TODO: add instructions for other distros -->

#### macOS

On macOS you can first install [Homebrew](https://brew.sh/), and then:

```console
brew install espeak sox abcmidi perl
cpan install MIDI
cpan install Math::FFT
```

<!-- TODO: add instructions for Windows -->

## Examples

Inside [the examples folder](https://github.com/ttm/music/tree/master/examples) you can find some scripts that use the main features of Music.

## Package structure

The modules are:

* **core**:
  * **synths** for synthesization of notes (including vibratos, glissandos, etc.), noises and envelopes.
  * **filters** for the application of filters such as ADSR envelopes, fades, IIR and FIR, reverb, loudness, and localization.
  * **io** for reading and writing WAV files, both mono and stereo.
  * **functions** for normalization.
* **structures** for higher level musical structures such as permutations (and related to algebraic groups and change ringing peals), scales, chords, counterpoint, tunings, etc.
* **singing** for singing with eCantorix. While it's not properly documented, and it might need some tweaks, it's working. Speech is currently achieved through espeak in the most obvious way, using os.system as in:
  * [Penalva](https://github.com/ttm/penalva/blob/master/penalva.py)
  * [Lunhani](https://github.com/ttm/lunhani/blob/master/lunhani.py)
  * [Soares](https://github.com/ttm/soares/blob/master/soares.py)
* **legacy** for musical pieces that are rendered with the Music package and might be used as material to make more music.
* **tables** for the generation of lookup tables for some basic waveform.
* **utils** for various functions regarding conversions, mix, etc.

## Roadmap

Music is stable but still very young. We didn't have the opportunity yet to make Music all we want it to be.

Here is one example of what we're aiming at:

```python
import music

music.render_demos() # render some wav files in ./

music.legacy.experiments.cristal2(.2, 300) # wav of sonic structure in ./

sound_waves = music.legacy.songs.madame_z(render=False) # return numpy array

sound_waves2 = music.core.io.open("demosong2.wav") # numpy array

music = music.remix(sound_waves, soundwaves2)
music_ = music.horizontal_stack(sound_waves[:44100*2], music[len(music)/2::2])

music.core.io.write_wav_mono(music_)

```

## Coding conventions

The code follows [PEP 8 conventions](https://peps.python.org/pep-0008/).

For a better understanding of each function, the math behind it and see examples of their use, you can read their docstring.

## Further information

Music is primarily intended for artistic use, but was also designed to run psychophysics experiments and data sonification.

You can find an example in [Versinus](https://github.com/ttm/versinus), an animated visualization method for evolving networks that uses Music to render the musical track that represents networks structures.

:::
