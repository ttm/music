"""Render one file per auditory stimulation technique.

Each of these implements a technique catalogued in SSTIM, the Sensory
Stimulation Vocabulary (https://w3id.org/sstim), and the docstrings in
`music.stimulation` name the term each one corresponds to.

The important difference between them is not how they sound but where
the modulation lives. Six of the seven put a real modulation into the
air, and it is present in the rendered file. The binaural beat does
not: each channel holds a steady tone, and the beat exists only for a
listener hearing both, which is why it alone must be heard over
headphones.

The last file is not a stimulus but a protocol: a session of three
phases, crossfaded rather than cut, which is the form these stimuli are
actually delivered in.
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

# Noise rather than a tone as the carrier. With no rate this is the
# plain broadband stimulus SSTIM types as a non-entrainment technique;
# with one it is amplitude modulation, whose definition names a carrier
# tone *or noise*.
music.write_wav_mono(
    music.modulated_noise(noise_type="pink", modulation_freq=RATE,
                          modulation_depth=0.8, duration=DURATION),
    "stim_noise.wav")

# A source orbiting the head. The angle is measured from the ear axis:
# 180 is the left ear's side and 0 the right, so this crosses the head
# and returns, four times over the eight seconds.
music.write_wav_stereo(
    music.spatial_motion(carrier_freq=CARRIER, motion_rate=0.5,
                         theta1=180, theta2=0, duration=DURATION),
    "stim_spatial.wav")

# A protocol, which is what these are delivered as: settle at 7 Hz,
# descend to 4, and rest under a noise bed. The ramps are crossfades
# centred on each boundary, so the session lasts exactly the 24 seconds
# its phases add up to -- a protocol that says eight minutes should not
# render 7:58 because it spent time fading.
session = music.StimulationSession(end_ramp=2.0)
session.add(music.binaural_beats, duration=DURATION, ramp=1.0,
            label="settle", carrier_freq=CARRIER, beat_freq=RATE)
session.add(music.isochronic_tones, duration=DURATION, ramp=2.0,
            label="descend", carrier_freq=CARRIER, pulse_rate=4.0,
            ramp_duration=0.005)
session.add(music.modulated_noise, duration=DURATION, ramp=2.0,
            gain=0.6, label="rest", noise_type="pink", modulation_freq=4.0)
session.write("stim_session.wav")

print("wrote seven stimuli, one per SSTIM technique, and a session")
print(f"session: {session!r}")
