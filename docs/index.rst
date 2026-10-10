music
=====

**Extreme-fidelity synthesis of musical elements.**

``music`` generates and manipulates music and sound in LPCM audio. It
implements the `MASS <https://github.com/ttm/mass/>`_ framework — *Music and
Audio in Sample Sequences* — a collection of psychophysical descriptions of
musical elements expressed as equations and corresponding Python routines.

.. code-block:: python

   import music

   notes = [music.note(440 * 2 ** (i / 12), duration=0.4) for i in range(12)]
   scale = [music.adsr(sonic_vector=note) for note in notes]
   music.write_wav_mono(music.horizontal_stack(*scale), "chromatic.wav")

The envelope brings each note to silence at its ends so that joining
different pitches does not introduce a click.

For executable SSTIM descriptions and oversampled DSP, see\n:doc:`sstim_interop`.\n\nNot sure which routine to start with? :doc:`choosing_an_api` maps common\ntasks onto a small supported entry-point set, including band-limited\nstatic-pitch notes.\n\nNew here? The :doc:`tutorial` walks from a single note to a short stereo
piece, and explains what the sample-by-sample model actually buys you.

What it does
------------

- **Synthesis** of notes and their variation, sample by sample: vibrato,
  glissando, FM and AM, Doppler, ADSR envelopes, tremolo, and noise of
  six colours.
- **Filters**: IIR and FIR, reverberation, loudness transitions, and the
  four filter designs the MASS article specifies.
- **Spatial audio**: interaural time and intensity differences computed at
  every sample, a source moving in a line or around the head, and measured
  head-related transfer functions.
- **Music theory**: the diatonic modes, the minor scales, triads and
  tetrads, intervals and their names, the harmonic series.
- **Musical structures**: permutation groups, change-ringing peals and
  plain changes.
- **Bonds** that tie a note's vibrato and tremolo to its pitch, once, for
  a whole piece.
- **Sensory stimulation**: seven auditory stimuli named for the SSTIM
  techniques they implement, and sessions that sequence them.
- **Singing**: a lyric sung to a melody by speech synthesis and PSOLA,
  with a vibrato on long notes and accents on the beats, and the
  eCantorix engine as a second backend that sings the same score.
- **Input and output**: reading, writing and playing audio, mono and
  stereo, and a sequencer that schedules notes into a timeline.

What makes it precise
---------------------

**Sample-based synthesis.** State is updated at every sample. A note with a
vibrato has a different instantaneous frequency at each of its samples, and the
vibrato pattern is folded into the wavetable lookup rather than applied
afterwards — so the rendered sound is as close as it can be to the mathematical
model that describes it.

**Musical structures**, with an emphasis on symmetry and discourse: permutation
groups, change-ringing peals and plain changes.

Every routine's docstring carries the equation it implements and cites the
article it comes from. If you use this package, please cite:

   Fabbri, Renato, et al. *Musical elements in the discrete-time representation
   of sound.* arXiv preprint `arXiv:1412.6853 <https://arxiv.org/abs/1412.6853>`_
   (2017).

Install
-------

.. code-block:: console

   pip install music

Or from a checkout, which is convenient for hacking and debugging:

.. code-block:: console

   git clone https://github.com/ttm/music.git
   pip install -e music

Requires Python 3.10 or newer.

:meth:`PrimaryTables.draw_tables <music.PrimaryTables.draw_tables>` plots the
waveform tables and is the one thing that needs matplotlib, which is an extra:

.. code-block:: console

   pip install 'music[plot]'

Nothing else in the package uses it, and leaving it out makes ``import music``
about 40% faster.

:func:`music.sing` sings with espeak-ng, a system program, and Praat,
through praat-parselmouth, which the singing extra installs:

.. code-block:: console

   sudo apt install espeak-ng        # or: brew install espeak-ng
   pip install 'music[singing]'

Its second backend, eCantorix, is a Perl program the package clones; the
:doc:`api` says what it needs.

Where things live
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Module
     - What it holds
   * - :doc:`core.synths <api>`
     - Notes with vibrato, glissando, FM and Doppler; envelopes; noises
   * - :doc:`core.filters <api>`
     - ADSR, fades, FIR/IIR, reverb, loudness, stereo localization
   * - :doc:`core.io <api>`
     - Reading, writing and playing audio, mono and stereo
   * - :doc:`structures <api>`
     - Permutations, algebraic groups, change-ringing peals
   * - :doc:`sequencer <api>`
     - Scheduling notes into a timeline and rendering them
   * - :doc:`utils <api>`
     - Conversions, mixing, stacking, rhythm
   * - :doc:`theory <api>`
     - Scales, modes, chords, intervals and the harmonic series
   * - :doc:`bonds <api>`
     - Vibrato and tremolo tied to a note's pitch
   * - :doc:`hrtf <api>`
     - The KEMAR measurements, fetched and read by direction
   * - :doc:`stimulation <api>`
     - Binaural, monaural and isochronic beats, modulations, sessions
   * - :doc:`singing <api>`
     - ``sing``, with the PSOLA singer and the eCantorix engine
   * - :doc:`tables <api>`
     - Waveform lookup tables
   * - :doc:`legacy <api>`
     - The ``Being`` and ``IteratorSynth`` synthesizer classes

The whole public API is re-exported flat from the top level, so
``music.note(...)`` works regardless of which submodule defines it.

.. toctree::
   :maxdepth: 2
   :hidden:

   tutorial
   choosing_an_api
   sstim_interop
   api

.. toctree::
   :caption: Project
   :hidden:

   GitHub <https://github.com/ttm/music>
   Issues <https://github.com/ttm/music/issues>
   Sponsor <https://github.com/sponsors/ttm>
   MASS framework <https://github.com/ttm/mass/>
