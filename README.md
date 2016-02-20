# music
a python package to make music and sounds.

## core features
  - sample-based synthesis, meaning methods where each sample is calculated indivudally. Usually not fit for real-time, but fidelity of sound wave to the mathematical models is maximized.
  - speech and singing interface.
  - musical structures, with emphasys in symmetry and discourse.
  - idealized to be used as standalone and for audiovisualization of data, in compliance to the participation ontology and the percolation package for harnessing open linked social data.

## install with
    $ pip install music
or

    $ python setup.py music

For greater control of customization (and debugging), clone the repo and install with pip with -e:

    $ git clone https://github.com/ttm/music.git
    $ pip install -e <path_to_repo>
This install method is especially useful when reloading modified module in subsequent runs of music.

## coding conventions
A function name has a verb if it changes state of initialized objects, if it only "returns something", it is has no verb in name.

Classes, functions and variables are writen in CamelCase, headlessCamelCase and lowercase, respectively.
Underline is used only in variable names where the words in variable name make something unreadable (usually because the resulting name is big).

The code is the documentation. Code should be very readable to avoid writing unnecessary documentation and duplicating routine representations. This adds up to using docstrings to give context to the objects or omitting the docstrings.

Every feature should be related to at least one legacy/ outline.

### package structure
Data is kept in the data/ directory.


#### the modules are:
utils.py for small functionalities that fit nowhere else

## usage example
```python
import music as M

M.renderDemos() # render some music wav files in ./

M.legacy.experiments.cristal2(.2,300) # wav of sonic structure in ./

sound_waves=M.legacy.songs.madameZ(render=False) # return numpy array

sound_waves2=M.io.open("demosong2.wav") # numpy array

music=M.remix(sound_waves, soundwaves2)
music_=M.H(sound_waves[:44100*2],music[len(music)/2:])

M.oi.write(music_)

```

## further information
Music should be integrated to the participation ontology and the percolation package
to enable anthropological physics experiments and social harnessing:
- https://github.com/ttm/percolation

This means mainly using RDF to link music package facilities with media rendering using diverse data sources as underlying material.
