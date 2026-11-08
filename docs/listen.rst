:og:description: Sounds computed by music, a Python package for sound synthesis, each under the code that computes it: a scale, vibrato, aliasing, noise, filters, harmony, change ringing, spatial audio, binaural beats and singing.

Listen
======

Each sound on this page is computed by the code above it, when the
documentation is built, and played as that code returns it: written
losslessly, as FLAC, and brought to one loudness, so that no clip is
louder than the next only because of what it is. Each block leaves its
sound in ``sound``, which :func:`music.play_audio` plays and
:func:`music.write_audio` writes, in one channel or two.

In a notebook, ``IPython.display.Audio(sound,
rate=music.get_sample_rate())`` plays it in the page. Every block here is
in `a notebook that runs in Google Colab
<https://colab.research.google.com/github/ttm/music/blob/master/examples/listen.ipynb>`_,
with nothing to install.

Two of these are for headphones, and say so; one is high, and says that.

A scale, joined without clicks
------------------------------

Thirteen notes of the chromatic scale. Each note ends wherever its phase
happens to be, so notes joined raw would click; the envelope takes each
to silence at both ends first.

.. sound:: chromatic-scale

   import music

   notes = [music.note(440 * 2 ** (i / 12), duration=0.25)
            for i in range(13)]
   sound = music.horizontal_stack(*[music.adsr(sonic_vector=note)
                                    for note in notes])

A vibrato, and a glide under it
-------------------------------

The pitch moves at every sample, inside the wavetable lookup, rather than
being bent afterwards: half a semitone each way, six times a second, on a
held note and then on a glide up an octave.

.. sound:: vibrato

   import music

   held = music.note_with_vibrato(220, duration=2, vibrato_freq=6,
                                  max_pitch_dev=0.5)
   glide = music.note_with_glissando_vibrato(220, 440, duration=2,
                                             vibrato_freq=6,
                                             max_pitch_dev=0.5)
   sound = music.horizontal_stack(music.adsr(sonic_vector=held),
                                  music.adsr(sonic_vector=glide))

A sawtooth that folds, and one that does not
--------------------------------------------

*High: turn the volume down first.* A sawtooth at 10 kHz read from a
table, as MASS reads it, and then the same note band-limited. The first
has partials above the Nyquist frequency, folded back into the band
between its own harmonics, where they sound as tones the note does not
have; the second has none. :doc:`bandlimited_synthesis` measures both.

.. sound:: folding

   import music

   folded = music.note(10000, duration=1.5,
                       waveform_table=music.waveform_table("sawtooth"))
   clean = music.bandlimited_note(10000, duration=1.5, waveform="sawtooth")
   sound = music.horizontal_stack(music.adsr(sonic_vector=folded),
                                  music.silence(duration=0.5),
                                  music.adsr(sonic_vector=clean))

Noise, by colour
----------------

Brown, pink, white, blue and violet: each colour is a slope, in decibels
per octave, from -6 for brown to +6 for violet. A ``seed`` draws the same
noise again.

.. sound:: noise

   import music

   colours = ("brown", "pink", "white", "blue", "violet")
   sound = music.horizontal_stack(*[
       music.adsr(sonic_vector=music.noise(colour, duration=0.6, seed=1))
       for colour in colours])

A low pass, opening
-------------------

White noise through a low pass whose cutoff doubles six times, from
125 Hz to 8 kHz. A cutoff is a fraction of the sample rate, which
:func:`music.fraction_of` converts hertz to, so the same coefficients
describe the same filter at any rate.

.. sound:: low-pass

   import music

   noise = music.noise("white", duration=0.5, seed=2)
   steps = [music.iir(noise, *music.low_pass(music.fraction_of(cutoff)))
            for cutoff in (125, 250, 500, 1000, 2000, 4000, 8000)]
   sound = music.horizontal_stack(*[music.adsr(sonic_vector=step)
                                    for step in steps])

A mode, and a cadence
---------------------

The theory is counted in semitones from a tonic of zero, so a scale or a
chord becomes frequencies in one step. The dorian mode up from A, and
then I, vi, IV and V7 in A major.

.. sound:: mode-and-cadence

   import music

   dorian = music.pitch_to_freq(220.0, music.scale("dorian") + (12,))
   line = [music.adsr(sonic_vector=music.note(freq, duration=0.3))
           for freq in dorian]
   chords = [music.chord("major"), music.chord("minor", root=9),
             music.chord("major", root=5),
             music.chord("dominant seventh", root=7)]
   cadence = [music.adsr(sonic_vector=music.mix_many(
                  [music.note(freq, duration=1)
                   for freq in music.pitch_to_freq(220.0, chord)]))
              for chord in chords]
   sound = music.horizontal_stack(*line, music.silence(duration=0.4),
                                  *cadence)

A vibrato bound to the pitch
----------------------------

:class:`music.Bonds` decides once how a line's notes behave, rather than
note by note: here the vibrato quickens as the line rises, and its depth
narrows.

.. sound:: bonds

   import music

   voice = music.Bonds(vibrato_freq=music.proportional(1 / 40),
                       max_pitch_dev=music.inversely_proportional(400))
   sound = voice.render([220, 277, 330, 440, 330, 277, 220], duration=0.5)

Plain changes on four bells
---------------------------

Four bells, tuned to the first four notes of E major and rung from the
highest, in every order once: the permutations act on the bells'
frequencies, so the peal is the melody. It needs ``pip install
'music[structures]'``.

.. sound:: plain-changes
   :needs: structures

   import music

   bells = [440.0, 415.3, 369.99, 329.63]
   rows = music.PlainChanges(4).act(bells)
   sound = music.horizontal_stack(*[
       music.adsr(sonic_vector=music.note(freq, duration=0.16))
       for row in rows for freq in row])

A source passing by
-------------------

*Headphones.* Pink noise moving from one side to the other, its
interaural time and intensity differences computed at every sample from
where it is.

.. sound:: passing

   import music

   source = music.adsr(sonic_vector=music.noise("pink", duration=3, seed=3))
   sound = music.localize_linear(source, theta1=150, theta2=30, dist=0.6)

A binaural beat
---------------

*Headphones.* 197 Hz in one ear and 203 Hz in the other. Neither ear
hears a beat; the 6 Hz one arises where their signals meet, in the
listener. Each of the package's stimuli is named for the SSTIM technique
it implements, and :doc:`sstim` writes them as descriptions.

.. sound:: binaural-beat

   import music

   beat = music.binaural_beats(carrier_freq=200, beat_freq=6, duration=8)
   sound = music.adsr(sonic_vector=beat, attack_ms=500, release_ms=500)

A session, crossfaded
---------------------

Stimuli are delivered as a protocol, one phase after another. A
:class:`music.StimulationSession` plays its phases in order, crossfaded
rather than cut, and lasts exactly as long as the durations written down:
isochronic tones, and then pink noise modulated at the same rate.

.. sound:: session

   import music

   session = music.StimulationSession(end_ramp=0.5)
   session.add(music.isochronic_tones, duration=5, carrier_freq=300,
               pulse_rate=4, ramp_duration=0.01)
   session.add(music.modulated_noise, duration=5, ramp=1,
               noise_type="pink", modulation_freq=4, seed=1)
   sound = session.render()

A line, sung
------------

One syllable a note: espeak-ng says each syllable, and Praat's
pitch-synchronous overlap-add holds it at its note's pitch and length,
with a vibrato on the long note. It needs espeak-ng and ``pip install
'music[singing]'``.

.. sound:: singing
   :needs: singing

   import music

   sound = music.sing(text="Twin-kle, twin-kle, lit-tle star",
                      notes=(0, 0, 7, 7, 9, 9, 7),
                      durs=(1, 1, 1, 1, 1, 1, 2))
