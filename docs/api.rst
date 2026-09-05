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

Sensory stimulation
~~~~~~~~~~~~~~~~~~~

Stimuli for sensory-stimulation work, each rendering one technique
catalogued in `SSTIM <https://w3id.org/sstim>`_, the Sensory Stimulation
Vocabulary. Every docstring names the SSTIM term it implements and states
whether the modulation is physically present in the signal or constructed
by the listener.

.. autosummary::
   :toctree: generated
   :nosignatures:

   binaural_beats
   monaural_beats
   isochronic_tones
   amplitude_modulation
   frequency_modulation
   modulated_noise
   spatial_motion

A protocol is a sequence of those rather than one of them.
:class:`~music.StimulationSession` holds that sequence and renders it as
one sound, joining the phases with crossfades centred on their boundaries
so that the session lasts exactly the sum of the durations it was given.

.. autosummary::
   :toctree: generated
   :nosignatures:

   StimulationSession
   StimulusPhase

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

Filter design
~~~~~~~~~~~~~

``iir`` applies coefficients; these compute them, from the four designs the
MASS article specifies. Cutoff, centre and bandwidth are fractions of the
sample rate, which ``fraction_of`` converts a frequency in Hertz into.

.. autosummary::
   :toctree: generated
   :nosignatures:

   low_pass
   high_pass
   band_pass
   band_reject
   fraction_of

Input and output
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   read_audio
   read_wav
   write_audio
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

Music theory
------------

Scales, chords and the harmonic series, counted in semitones from a tonic or
root of zero -- which is what ``pitch_to_freq`` takes, so any of these
becomes frequencies and then sound in two steps.

.. autosummary::
   :toctree: generated
   :nosignatures:

   scale
   mode_by_rotation
   harmonic_series
   chord
   add_seventh
   invert

The tables these read from are exported too: ``SCALES``, ``MODES``,
``MINOR_SCALES`` and ``DIATONIC_STEPS`` for the scales, ``CHORDS``,
``TRIADS`` and ``SEVENTHS`` for the chords, and
``HARMONIC_SERIES_AS_PRINTED`` for the article's own table of partials.

Bonds
-----

Relations tying a note's vibrato and tremolo to its frequency, so that a
piece decides once how its notes behave rather than note by note.

.. autosummary::
   :toctree: generated
   :nosignatures:

   Bonds
   proportional
   inversely_proportional
   stepped

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
   mix2
   mix_many
   mix_many_with_offsets
   mix_stereo
   mix_with_offset
   mix_with_offset_
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
   waveform_table

``WAVEFORM_SINE``, ``WAVEFORM_TRIANGULAR``, ``WAVEFORM_SQUARE`` and
``WAVEFORM_SAWTOOTH`` are the lookup tables used by default throughout the
package, and are the names that appear in the synthesis signatures above.
They are re-exported from the top level like everything else, so
``music.note(waveform_table=music.WAVEFORM_SINE)`` works;
``music.WAVEFORMS`` names the shapes :func:`waveform_table` can build.

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
