# music
A python package to make music and sounds
based in the [MASS](https://github.com/ttm/mass/) (Music and Audio in Sample Sequences) framework.
MASS is roughly a collection of psychophysical descriptions of musical elements
in LPCM audio through equations and corresponding Python routines.
It thus makes facilitated the creation of Python packages with various functionalities.

Refer to the article
[Musical elements in the discrete-time representation of sound](https://github.com/ttm/mass/raw/master/doc/article.pdf)
for further understanding the routines
and please cite the work if you use this package.

### core features
* Sample-based synthesis,
meaning that the state is updated at each sample.
(For example,  in a note with a vibrato, each sample is associated
to a different frequency.)
Thus, fidelity of the synthesized sound to the mathematical models is maximized.
* Musical structures, with emphasis in symmetry and discourse.
* Speech and singing interface.
* Idealized to be used as standalone and for audiovisualization of data 
(e.g. for harnessing open linked social data in association to the participation ontology (PO)
and the percolation package, or with the audiovidual analytics vocabulary and ontology (AAVO)).

### install with
    $ pip install music
or

    $ python setup.py music

For greater control of customization, hacking and debugging, clone the repository and install with pip using -e:

    $ git clone https://github.com/ttm/music.git
    $ pip3 install -e <path_to_repo>

This install method is especially useful when reloading the modified module in subsequent runs of music.

#### package structure
The modules are:
* core.py
* utils.py for small functionalities (e.g. IO and conversions)
* core/ with basic algorithms derived from the MASS framework. Imports all from:
    - functions.py: this file also imports all the example functions available in the MASS framework.
    These are in functions.py and include various synths, effects and utilities (envelope lines, etc).
    - classes.py: currently holds only the powerful class Being(), which should be further documented and its usage exemplified.
* synths.py have additional synths not included in MASS
    - *NOTE*: one should check the core module synths to know about the synths inherited directly from MASS, they range from very simple to great complexity.
    This module is very incipient compared to the MASS framework.
* effects.py for sonic effects 
    - NOTE: one whoud check core.\* for a number of effects, from tremolo and AM to spatial and spectral manipulations.
    Again, this module is very incipient compared to the MASS framework. In fact, it is not implemented yet.
* structures/ for higher level musical structures
    - such as permutations (and related to algebraic groups and change ringing peals), scales, chords, counterpoint, tunings, etc.
    - implemented are:
      * Plain Changes in any number of bells with any number of hunts (hun, half hunt, etc).
      The limitation here is really your machine and system, but you should be able to obtain
      complete plain changes peals with at least 12 sounds/bells/items.
      This is implemented in structures.peals.plainChanges.py
      * Some organization of basic sets of permutations, such as related to rotations, mirroring, alternating etc.
      This is achieved though [group theory] and arbitrary ordering of the permutations.
      I try to overcome this arbitrary ordering for more than a decade...
      And my hopes to do so is through Group Representation Theory.
      This is implemented in structrures.permutations.py
      * symmetry.py just gathers peals and other permutation sets for an organization of the content.
      * sumfreė.py meant for [sumfree sets] and related structures but sketched routines have not been migrated yet.
* singing/ for singing with eCantorix
    - Not properly documented but working (might need tweaks, the routines use Ecantorix, that uses espeak..) 
        * TODO: make annotation about espeak in setup.py or .cfg
    - Speech is currently achieved through espeak in the most obvious way, using os.system as in:
        * https://github.com/ttm/penalva/blob/master/penalva.py
        * https://github.com/ttm/lunhani/blob/master/lunhani.py
        * https://github.com/ttm/soares/blob/master/soares.py
* legacy/ for musical pieces that are rendered with the music package (and might be appreciated directly or used as material to make more music)
    - currently has only one musical piece (a silly one indeed).
* music/ for remixing materials into new pieces and for generating new pieces from scratch (with arbitrary parametrization)
    - Don't exist yet; the sketches have not been migrated.
    - Should work in cooperation with the legacy/ module.

### coding conventions
A function name has a verb if it changes state of initialized objects, if it only "returns something", it is has no verb in name.

Classes, functions and variables are written in CamelCase, headlessCamelCase and lowercase, respectively.
Underline is used only in variable names where the words in variable name make something unreadable.

The code is *the* documentation.
Code should be very readable to avoid writing unnecessary documentation and duplicating routine representations.
This adds up to using docstrings to give context to the objects or omitting the docstrings and documenting the code lines directly.
TODO: Doxigen or a similar tool should be employed ASAP.

Ideally, every feature will be related to at least one legacy/ routine.

### usage example

```python
### Basic usage
import music as M, numpy as n
T = M.tables.Basic()
H = M.utils.H


# 1) start a ѕynth
b = M.core.Being()

# 2) set its parameters using sequences to be iterated through
b.d_ = [1/2, 1/4, 1/4]  # durations in seconds
b.fv_ = [0, 1,5,15,150,1500,15000]  # vibrato frequency
b.nu_ = [5]  # vibrato depth in semitones (maximum deviation of pitch)
b.f_ = [220, 330]  # frequencies for the notes

# 3) render the wavfile
b.render(30, 'aMusicalSound.wav')  # render 30 notes iterating though the lists above

# 3b) Or the numpy arrays directly and use them to concatenate and/or mix sounds:
s1 = b.render(30)
b.f_ += [440]
b.fv_ = [1,2,3,4,5]
s2 = b.render(30)

# s1 then s2 then s1 and s2 at the same time, then at the same time but one in each LR channel,
# then s1 times s2 reversed, then s1+s2 but jumping 6 samples before using one:
s3 = H(s1, s2, s1 + s2, (s1, s2),
       s1*s2[::-1],
       s1[::7] + s2[::7])
M.core.WS(s3, 'tempMusic.wav')

# X) Tweak with special sets of permutations derived from change ringing (campanology)
# or from finite group theory (algebra):
nel = 4
pe4 = M.structures.symmetry.PlainChanges(nel)
b.perms = pe4.peal_direct
b.domain = [220*2**(i/12) for i in (0,3,6,9)]
b.curseq = 'f_'
b.f_ = []
nnotes = len(b.perms)*nel  # len(b.perms) == factorial(nel)
b.stay(nnotes)
b.nu_= [0]
b.d_ += [1/2]
s4 = b.render(nnotes)

b2 = M.core.Being()
b2.perms = pe4.peal_direct
b2.domain = b.domain[::-1]
b2.curseq = 'f_'
b2.f_ = []
nnotes = len(b.perms)*nel  # len(b.perms) == factorial(nel)
b2.stay(nnotes)
b2.nu_= [2,5,10,30,37]
b2.fv_ = [1,3,6,15,100,1000,10000]
b2.d_ = [1,1/6,1/6,1/6]
s42 = b2.render(nnotes)

i4 = M.structures.permutations.InterestingPermutations(4)
b2.perms = i4.rotations
b2.curseq = 'f_'
b2.f_ = []
b2.stay(nnotes)
s43 = b2.render(nnotes)

s43_ = M.core.F(sonic_vector=s43, d=5, method='lin')

diff = s4.shape[0] - s42.shape[0]
s42_ = H(s42, n.zeros(diff))
s_ = H(s3, (s42_, s4), s43_)

M.core.WS(s_, 'geometric_music.wav')


##############
# Notice that you might relate a peal or any set of permutations
# to a sonic characteristic (frequency, duration, vibrato depth, vibrato frequency,
# attack duration, etc) through at least 3 methods:
# 1) initiate a Being(), set its permutations to the permutation sequence,
# its domain to the values to be permuted, and its curseq to
# the name of the Being sequence to be yielded by the permutation of the domain.
#
# 2) Achieve the sequence of values through peal.act() or just using permutation(domain)
# for all the permutations at hand.
# Then render the notes directly (e.g. using M.core.V_) or passing the sequence of values
# to a synth, such as Being()
#
# 3) Using IteratorSynth as explained below. (potentially deprecated)

pe3 = M.structures.symmetry.PlainChanges(3)
M.structures.symmetry.printPeal(pe3.act(), [0])
freqs = sum(pe3.act([220,440,330]), [])

nnotes = len(freqs)

b = M.core.Being()
b.f_ = freqs
b.render(nnotes, 'theSound_campanology.wav')

### OR
b = M.core.Being()
b.domain = [220, 440, 330]
b.perms = pe3.peal_direct
b.f_ = []
b.curseq = 'f_'
b.stay(nnotes)
b.render(nnotes, 'theSound_campanology_.wav')


### OR (DEPRECATED, but still kept while not convinced to remove...)
isynth = M.synths.IteratorSynth()
isynth.fundamental_frequency_sequence=freqs
isynth.tab_sequence = [T.sine, T.triangle, T.square, T.saw]

pcm_samples = M.H(*[isynth.renderIterate() for i in range(len(freqs))])

M.core.W(pcm_samples, 'something.wav')


#######
## More interesting examples are found in:
# https://github.com/ttm/mass/tree/master/src/finalPiece

```

#### idealized usage example
We didn't have the opportunity yet to make Music all we want it to be.
Here is one example of what we idealize:

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
are installed by setup.py by default.

You also need to install espeak, sox and abc2midi (abcmidi package in Ubuntu) if you want to use the singing interface.
Also, the MIDI.pm and FFT.pm files are needed by eCantorix to synthesize singing sequences,
and can be installed with:
  $ sudo cpan install MIDI
  $ sudo cpan install Math::FFT

or through your system's packaging system (e.g. apt for Ubuntu).

### further information
Music is intended for artistic uses, but was also designed for uses in psychophysics experiments and data sonification. E.g. the [Versinus](https://github.com/ttm/versinus) animated visualization method for evolving networks uses Music (an older version of it, actually) to render the musical track that represents networks structures.

### deployment to pypi
This package іs delivered by running:
  $ python3 setup.py sdist
  $ twine upload dist/

Maybe use "python setup.py sdist upload -r pypi" ?

:::
