"""Render one file per auditory stimulation technique.

Each of these implements a technique catalogued in SSTIM, the Sensory
Stimulation Vocabulary (https://w3id.org/sstim), and the docstrings in
`music.stimulation` name the term each one corresponds to.

The important difference between them is not how they sound but where
the modulation lives. Four of the five put a real modulation into the
air, and it is present in the rendered file. The binaural beat does
not: each channel holds a steady tone, and the beat exists only for a
listener hearing both, which is why it alone must be heard over
headphones.
"""

import music

DURATION = 8.0
CARRIER = 220.0
RATE = 7.0  # the entrainment rate these stimuli share, in Hz

# Perceptually constructed: stereo, and only over headphones.
music.write_wav_stereo(
    music.binaural_beats(carrier_freq=CARRIER, beat_freq=RATE,
                         duration=DURATION),
    "stim_binaural.wav")

# Physically present: the same two tones summed into one channel.
music.write_wav_mono(
    music.monaural_beats(carrier_freq=CARRIER, beat_freq=RATE,
                         duration=DURATION),
    "stim_monaural.wav")

# A gated pulse train. The ramp is not cosmetic: gating a tone abruptly
# is a step discontinuity twice per pulse, heard as a click and visible
# in a recording as broadband energy that is not part of the stimulus.
music.write_wav_mono(
    music.isochronic_tones(carrier_freq=CARRIER, pulse_rate=RATE,
                           duty_cycle=0.5, ramp_duration=0.005,
                           duration=DURATION),
    "stim_isochronic.wav")

# The canonical steady-state-response stimulus. The modulation rate,
# not the carrier, sets the frequency of the response.
music.write_wav_mono(
    music.amplitude_modulation(carrier_freq=CARRIER, modulation_freq=RATE,
                               modulation_depth=1.0, duration=DURATION),
    "stim_am.wav")

# Slow frequency modulation, in Hertz rather than in semitones.
music.write_wav_mono(
    music.frequency_modulation(carrier_freq=CARRIER, modulation_freq=0.25,
                               frequency_deviation=20, duration=DURATION),
    "stim_fm.wav")

print("wrote five stimuli, one per SSTIM technique")
