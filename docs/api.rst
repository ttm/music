API reference
=============

Every name below is re-exported from the top level: ``music.note(...)`` works
regardless of which submodule defines it.

.. currentmodule:: music

Synthesis
---------

Notes
~~~~~

Each returns a numpy array of PCM samples. ``note`` is the plain wavetable
lookup; the rest layer vibrato, pitch transitions, frequency modulation and
movement onto it.

.. autosummary::
   :toctree: generated
   :nosignatures:

   note
   note_with_phase
   note_with_vibrato
   note_with_two_vibratos
   note_with_glissando
   note_with_glissando_vibrato
   note_with_two_vibratos_glissando
   note_with_vibratos_glissandos
   note_with_vibrato_seq_localization
   note_with_fm
   note_with_doppler
   trill

Envelopes and noise
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   am
   tremolo
   tremolos
   noise
   gaussian_noise
   silence

Filters
-------

Amplitude
~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   adsr
   adsr_stereo
   adsr_vibrato
   fade
   cross_fade
   loud
   louds

Spectral and spatial
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   fir
   iir
   reverb
   localize
   localize2
   localize_linear
   stretches

Input and output
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   read_wav
   write_wav_mono
   write_wav_stereo
   play_audio
   normalize_mono
   normalize_stereo

Sequencing
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Sequencer

Musical structures
------------------

Permutations, algebraic groups and change-ringing peals.

.. autosummary::
   :toctree: generated
   :nosignatures:

   InterestingPermutations
   transpose_permutation
   dist
   GenericPeal
   Peals
   PlainChanges
   print_peal

Utilities
---------

Conversions
~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   hz_to_midi
   midi_to_hz
   midi_to_hz_interval
   pitch_to_freq
   db_to_amp
   amp_to_db
   rhythm_to_durations

Combining sonic vectors
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   horizontal_stack
   mix
   mix_many
   mix_many_with_offsets
   mix_stereo
   mix_with_offset
   convert_to_stereo
   resolve_stereo
   pan_transitions
   profile

Wavetables
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   PrimaryTables

The module-level constants ``WAVEFORM_SINE``, ``WAVEFORM_TRIANGULAR``,
``WAVEFORM_SQUARE`` and ``WAVEFORM_SAWTOOTH`` in :mod:`music.utils` are the
lookup tables used by default throughout the package.

Singing
-------

Text-to-speech built on the external `eCantorix
<https://github.com/ttm/ecantorix>`_ engine. Run :func:`setup_engine` once to
clone it; it also needs ``git``, ``make``, ``perl`` and ``espeak`` on the
system.

.. autosummary::
   :toctree: generated
   :nosignatures:

   setup_engine
   get_engine
   make_test_song

Legacy
------

Synthesizer classes kept for backwards compatibility, and as material for
making more music.

.. autosummary::
   :toctree: generated
   :nosignatures:

   Being
   CanonicalSynth
   IteratorSynth
