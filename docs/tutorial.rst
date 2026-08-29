Tutorial
========

There is one idea to absorb, and the rest follows from it: **every routine in
this package returns a numpy array of PCM samples.** A note is an array. An
envelope is an array. A whole piece is an array. Nothing is a handle to an
audio engine, nothing has to be started or torn down, and there is no
graph to wire up. You make arrays, you combine them with arithmetic, and at
the end you write one to a file.

This page walks from a single note to a short stereo piece. Every block
below runs as written.

A first sound
-------------

.. code-block:: python

   import music

   a = music.note(freq=440, duration=1.5)
   music.write_wav_mono(a, "first.wav")

``a`` is 66,150 float64 samples — 1.5 seconds at 44.1 kHz — in the range
[-1, 1]. Frequency is in hertz, duration in seconds. To hear it without
writing a file, use :func:`music.play_audio`.

Should you prefer to think in samples rather than seconds, every synthesis
routine takes ``number_of_samples`` instead, and it wins over ``duration``:

.. code-block:: python

   exactly_one_period = music.note(440, number_of_samples=round(44100 / 440))

It is all numpy
---------------

Because the return value is an ordinary array, the operations you already
know are the mixing desk:

.. code-block:: python

   quiet = music.note(440, 1) * 0.5           # halve the amplitude
   inverted = -music.note(440, 1)             # flip the phase
   both = music.note(440, 1) + music.note(660, 1)   # sum two tones

The package's own helpers exist for the cases where plain arithmetic is
not enough — where lengths differ, or where a stereo array has to be kept
distinct from a mono one. :func:`music.mix` sums two vectors, padding the
shorter; :func:`music.horizontal_stack` concatenates them end to end.

Notes in sequence
-----------------

A melody is notes concatenated:

.. code-block:: python

   freqs = [261.63, 293.66, 329.63, 349.23, 392.0]   # C D E F G
   melody = music.horizontal_stack(*[music.note(f, 0.35) for f in freqs])

If you would rather write rhythm in note values than in seconds,
:func:`music.rhythm_to_durations` converts them. Its ``durations`` are
denominators of a whole note, so ``4`` is a quarter note:

.. code-block:: python

   >>> music.rhythm_to_durations([4, 2, 2, 4], duration=0.25)
   [1.0, 0.5, 0.5, 1.0]

Pass ``bpm`` instead of ``duration`` to give the tempo directly.

Shaping a note
--------------

A bare note starts and stops abruptly, which is audible as a click. An
ADSR envelope fixes that and is most of what makes a sound feel like an
instrument:

.. code-block:: python

   raw = music.note(440, 1)
   shaped = music.adsr(sonic_vector=raw, attack_duration=50,
                       decay_duration=100, sustain_level=-9,
                       release_duration=300)

``shaped`` is the same length as ``raw``: the envelope is fitted to the
sound it is given, not appended to it. Called *without* a ``sonic_vector``,
the same function returns the bare envelope, which you can inspect, plot,
or multiply into something yourself.

Note the units. They are not uniform, and deliberately so — each parameter
is expressed in the unit that parameter is normally thought about in:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Kind of quantity
     - Unit
     - Examples
   * - Duration of a sound
     - seconds
     - ``duration``, ``envelope_duration``
   * - Duration of an envelope stage
     - milliseconds
     - ``attack_duration``, ``release_duration``
   * - Level
     - decibels
     - ``sustain_level``, ``max_db_dev``, ``trans_dev``
   * - Pitch deviation
     - semitones
     - ``max_pitch_dev``
   * - Frequency
     - hertz
     - ``freq``, ``vibrato_freq``, ``fm``
   * - Angle
     - degrees
     - ``theta``, ``theta1``, ``theta2``

The other shapers follow the same convention — pass ``sonic_vector`` to
apply, omit it to get the envelope. :func:`music.fade` for a fade in or
out, :func:`music.tremolo` for periodic loudness, :func:`music.am` for
amplitude modulation, :func:`music.loud` for a single loudness transition,
:func:`music.reverb` for a room.

What "sample by sample" buys you
--------------------------------

This is the claim the package is built on, and it is worth seeing rather
than taking on trust. A vibrato is a periodic deviation in *pitch*. The
common way to build one is to render a steady tone and modulate it
afterwards; this package instead folds the deviation into the wavetable
lookup, accumulating phase sample by sample, so the note genuinely has a
different instantaneous frequency at every one of its samples.

You can measure that from the output alone. Render two semitones of
vibrato around 440 Hz, then recover the frequency cycle by cycle from the
interpolated zero crossings:

.. code-block:: python

   import numpy as np

   v = music.note_with_vibrato(freq=440, duration=2, vibrato_freq=5,
                               max_pitch_dev=2)

   negative = np.signbit(v)
   crossings = np.where(~negative[1:] & negative[:-1])[0]
   exact = crossings + v[crossings] / (v[crossings] - v[crossings + 1])
   inst = 44100 / np.diff(exact)

   print(f"{inst.min():.1f} .. {inst.max():.1f} Hz")

which prints ``392.0 .. 493.9 Hz``. The model says the extremes should be
440·2\ :sup:`±2/12`, which is 392.0 and 493.9 Hz. The rendered audio
agrees to the last digit shown, and takes 617 distinct values along the
way — it is not stepping between a handful of pitches, it is a continuous
curve sampled at the audio rate.

The same is true of a glissando. An exponential sweep from 220 Hz to
880 Hz should pass through the geometric mean at its midpoint:

.. code-block:: python

   g = music.note_with_glissando(220, 880, duration=2, method="exp")

Measured at the halfway sample, it is at 441 Hz; the geometric mean of 220
and 880 is 440.

:func:`music.note_with_fm` (frequency modulation) and
:func:`music.note_with_doppler` (a source moving past the listener, which
produces the pitch shift *and* the stereo image) are built the same way.

Layering
--------

:func:`music.mix` sums two vectors and pads the shorter one, so a chord is
a fold:

.. code-block:: python

   chord = music.note(261.63, 2)
   for f in (329.63, 392.0):
       chord = music.mix(chord, music.note(f, 2))

:func:`music.mix_with_offset` starts the second sound a given number of
seconds after the first, which is enough for an arpeggio or a canon:

.. code-block:: python

   canon = music.mix_with_offset(melody, melody, duration=0.7)

A negative ``duration`` places the second sound that many seconds *before*
the first one ends.

The sequencer
-------------

Once there are more than a handful of overlapping events, give them
absolute start times instead:

.. code-block:: python

   seq = music.Sequencer()
   for i, freq in enumerate([440, 550, 660]):
       seq.add_note(freq, start=i * 0.25, duration=1.0,
                    adsr_params={"attack_duration": 20,
                                 "release_duration": 400})
   seq.write("chord.wav")

``seq.render()`` returns the array instead of writing it, so the result
goes back into everything above.

Stereo, and where a sound is
----------------------------

A stereo signal is a ``(2, n)`` array — channel first. The localization
routines take a mono vector and return one of these, computing the
interaural time and intensity differences from a geometry you describe.

:func:`music.localize` fixes the source at an angle and distance:

.. code-block:: python

   fixed = music.localize(music.note(440, 1), theta=45, distance=2)

:func:`music.localize_linear` moves it in a straight line, recomputing
the position — and from it the two delays and the two gains — at every
sample:

.. code-block:: python

   passing = music.localize_linear(music.note(330, 3),
                                   theta1=150, theta2=30, dist=0.6)
   music.write_wav_stereo(passing, "passing.wav", fades=(100, 500))

Angles are in degrees, and *x is the lateral axis*: 0° is the right ear,
90° is straight ahead, 180° is the left ear. So the call above sweeps from
left to right.

The delays are fractional — a source 0.3 samples further from one ear than
the other is rendered as 0.3 of a sample, by cubic interpolation, not
rounded to the nearest one. At 44.1 kHz a whole sample is already 8 mm of
head, so rounding would be audible as a coarse, stepping image.

``fades`` above is a shorthand every writer takes: ``(in, out)`` in
milliseconds, applied on the way out to the file.

Noise
-----

Six colours, each defined by its gain per octave — brown at −6 dB, pink at
−3, white at 0, blue at +3, violet at +6, black at −12:

.. code-block:: python

   colours = music.horizontal_stack(
       *[music.noise(kind, 0.5)
         for kind in ("brown", "pink", "white", "blue", "violet")])

Any number works in place of a name, and means exactly that many decibels
per octave, so ``music.noise(-1.5, 0.5)`` is halfway between pink and
white. Noise is also what :func:`music.reverb` convolves with to make a
tail.

For a response you design yourself, :func:`music.fir` takes a magnitude
spectrum — one absolute value per frequency bin, the last being Nyquist —
and applies it:

.. code-block:: python

   dull = music.fir(np.linspace(1, 0, 64), music.noise("white", 0.5))

Structure: change ringing as melody
-----------------------------------

The part of the package with the least competition elsewhere is the
:mod:`music.structures` module: permutation groups, and the peals of
English change ringing. A peal is a sequence of permutations in which each
step swaps only adjacent pairs, and which visits every permutation exactly
once. Applied to bells it is campanology; applied to *frequencies* the
peal is the melody, and its symmetry is audible:

.. code-block:: python

   peal = music.PlainChanges(4)
   rows = peal.act([220, 275, 330, 440])
   notes = [music.note(freq, 0.2) for row in rows for freq in row]
   music.write_wav_mono(music.horizontal_stack(*notes), "campanology.wav")

``rows`` holds all 24 permutations of the four frequencies, in ringing
order — ``[220, 275, 330, 440]``, ``[275, 220, 330, 440]``,
``[275, 330, 220, 440]``, and so on. ``act`` will permute any sequence you
hand it, so the same structure can order durations, amplitudes, or stereo
positions rather than pitches.

:class:`music.InterestingPermutations` gives the other groups — rotations,
reflections, alternating and dihedral subgroups — with the same interface.

Putting it together
-------------------

Nothing here is new; it is the pieces above composed:

.. code-block:: python

   import music
   import numpy as np

   # a melodic line, each note given a vibrato and an envelope
   voice = music.horizontal_stack(*[
       music.adsr(sonic_vector=music.note_with_vibrato(
                      freq, 0.5, vibrato_freq=6, max_pitch_dev=0.3),
                  sustain_level=-6)
       for freq in (392, 440, 493.88, 587.33)])

   # a quiet bed of pink noise underneath it, of exactly the same length
   bed = music.loud(sonic_vector=music.noise("pink",
                                             len(voice) / 44100)) * 0.05

   # mix them, then sweep the result across the stereo field
   piece = music.localize_linear(music.mix(voice, bed),
                                 theta1=120, theta2=60)
   music.write_wav_stereo(piece, "piece.wav", fades=(100, 500))

Where to go next
----------------

* The :doc:`api` reference. Every routine's docstring carries the equation
  it implements, an example, and the article section it comes from.
* `The examples folder
  <https://github.com/ttm/music/tree/master/examples>`_, which holds each
  of the above as a runnable script, plus the ``Being`` and
  ``IteratorSynth`` classes for generating material algorithmically.
* `Musical elements in the discrete-time representation of sound
  <https://arxiv.org/abs/1412.6853>`_, the article the whole package
  implements. If you use this package, please cite it.
