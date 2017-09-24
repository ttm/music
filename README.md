# music
A python package to make music and sounds
based in the [MASS] (Music and Audio in Sample Sequences) framework.

Please refer to the article
[Musical elements in the discrete-time representation of sound](https://arxiv.org/abs/1412.6853)
for understanding the implementation and cite the work if you use it.

### core features
* sample-based synthesis, meaning methods where each sample is calculated individually. Usually not fit for real-time, but fidelity of sound wave to the mathematical models is maximized.
* musical structures, with emphasis in symmetry and discourse.
* speech and singing interface.
* idealized to be used as standalone and for audiovisualization of data (e.g. in compliance to the participation ontology and the percolation package for harnessing open linked social data).

### install with
    $ pip install music
or

    $ python setup.py music

For greater control of customization (and debugging), clone the repo and install with pip with -e:

    $ git clone https://github.com/ttm/music.git
    $ pip install -e <path_to_repo>

This install method is especially useful when reloading modified module in subsequent runs of music.

### coding conventions
A function name has a verb if it changes state of initialized objects, if it only "returns something", it is has no verb in name.

Classes, functions and variables are written in CamelCase, headlessCamelCase and lowercase, respectively.
Underline is used only in variable names where the words in variable name make something unreadable (usually because the resulting name is big).

The code is the documentation. Code should be very readable to avoid writing unnecessary documentation and duplicating routine representations. This adds up to using docstrings to give context to the objects or omitting the docstrings.

Every feature should be related to at least one legacy/ outline.

#### package structure
The modules are:
* utils.py for small functionalities (e.g. IO and conversions)
* core/ with basic algorithms derived from the MASS framework
* synths.py for useful synthesis routines
* effects.py for effects on a given PCM sonic array
* structures/ for higher level musical structures such as permutations (and related algebraic groups and change ringing peals), scales, chords, counterpoint, tunings, etc.
* singing/ for singing with eCantorix
* legacy/ for musical pieces that are rendered with Music (and might be appreciated directly or used as material to make more music)
* music/ for remixing materials into new pieces and for generating new pieces from scratch (with arbitrary parametrization)

### usage example

```python
import music as M

pe3 = M.structures.symmetry.PlainChanges(3)
M.structures.symmetry.printPeal(pe4.act(), [0])

freqs = sum(pe3.act([220,440,330]), [])
isynth=IteratorSynth()
isynth.fundamental_frequency_sequence=freqs

pcm_samples = M.H(*[isynth.renderInterate() for i in range(len(freqs))])

```

#### idealized usage example
We didn't have the opportunity yet to make Music all we want it to be.
Here is an example of what one should be able to do:

```python
import music as M

M.renderDemos() # render some music wav files in ./

M.legacy.experiments.cristal2(.2, 300) # wav of sonic structure in ./

sound_waves=M.legacy.songs.madameZ(render=False) # return numpy array

sound_waves2=M.io.open("demosong2.wav") # numpy array

music=M.remix(sound_waves, soundwaves2)
music_=M.H(sound_waves[:44100*2], music[len(music)/2::2])

M.oi.write(music_)

```

### dependencies
The Python modules 
sympy, numpy, scipy, colorama, termcolor
are needed (and installed by the setup.py by default).

You also need to install sox and abc2midi (abcmidi package in Ubuntu).
The MIDI.pm and FFT.pm files are needed by eCantorix
and can be installed with:
  $ sudo cpan install MIDI
  $ sudo cpan install Math::FFT

or through your system's packaging system (e.g. apt for Ubuntu).

### further information
Music should be integrated to [AAVO], the participation ontology and the percolation package
to enable [anthropological physics] experiments and social harnessing:
- https://github.com/ttm/percolation

This means mainly using RDF to link between music package facilities and media rendering using diverse data sources as underlying material.

[AAVO]: https://github.com/ttm/aavo
[anthropological physics]: https://www.academia.edu/10356773/What_are_you_and_I_anthropological_physics_fundamentals_

