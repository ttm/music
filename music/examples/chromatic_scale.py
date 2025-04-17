""" Simple script that writes a chromatic scale on a WAV file. """

import music

scale = [
    261.63,  # C4
    277.18,  # C#4
    293.66,  # D4
    311.13,  # D#4
    329.63,  # E4
    349.23,  # F4
    369.99,  # F#4
    392.00,  # G4
    415.30,  # G#4
    440.00,  # A4
    466.16,  # A#4
    493.88   # B4
]

sonic_vector = []

for note in scale:
    sound = music.core.synths.note(freq=note,
                                   duration=0.4)
    sonic_vector.append(sound)

stack = music.utils.horizontal_stack(*sonic_vector)

music.core.io.write_wav_mono(sonic_vector=stack,
                             filename='chromatic_scale.wav')
