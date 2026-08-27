music
=====

**Extreme-fidelity synthesis of musical elements.**

``music`` generates and manipulates music and sound in LPCM audio. It
implements the `MASS <https://github.com/ttm/mass/>`_ framework — *Music and
Audio in Sample Sequences* — a collection of psychophysical descriptions of
musical elements expressed as equations and corresponding Python routines.

.. code-block:: python

   import music

   scale = [music.note(440 * 2 ** (i / 12), duration=0.4) for i in range(12)]
   music.write_wav_mono(music.horizontal_stack(*scale), "chromatic.wav")

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

The whole public API is re-exported flat from the top level, so
``music.note(...)`` works regardless of which submodule defines it.

.. toctree::
   :maxdepth: 2
   :hidden:

   api

.. toctree::
   :caption: Project
   :hidden:

   GitHub <https://github.com/ttm/music>
   Issues <https://github.com/ttm/music/issues>
   MASS framework <https://github.com/ttm/mass/>
