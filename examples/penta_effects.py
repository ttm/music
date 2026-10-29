""" Simple script that writes a pentatonic scale on a WAV file
    with different effects: plain, gliding up 30 Hz on each note, with a
    vibrato, with a Doppler shift and with FM.

    Each note is shaped with an ADSR envelope before the notes are
    joined. A note ends wherever its phase lands and the next opens at the
    bottom of its table, so raw notes joined end to end click.
"""

import music

scale = [
    261.63,  # C4
    293.66,  # D4
    329.63,  # E4
    392.00,  # G4
    440.00   # A4
]

sonic_vector = []
for note in scale:
    sound = music.core.synths.note(freq=note,
                                   duration=0.4)
    sonic_vector.append(music.adsr(sonic_vector=sound))

sonic_vector.append(music.core.synths.silence())

for note in scale:
    sound = music.core.synths.note_with_glissando(start_freq=note,
                                                  end_freq=note+30,
                                                  duration=0.4)
    sonic_vector.append(music.adsr(sonic_vector=sound))

sonic_vector.append(music.core.synths.silence())

for note in scale:
    sound = music.core.synths.note_with_vibrato(freq=note,
                                                duration=0.4)
    sonic_vector.append(music.adsr(sonic_vector=sound))

sonic_vector.append(music.core.synths.silence())

for note in scale:
    sound = music.core.synths.note_with_doppler(freq=note,
                                                duration=0.4)
    sonic_vector.append(music.adsr(sonic_vector=sound))

sonic_vector.append(music.core.synths.silence())

for note in scale:
    sound = music.core.synths.note_with_fm(freq=note,
                                           duration=0.4)
    sonic_vector.append(music.adsr(sonic_vector=sound))

stack = music.utils.horizontal_stack(*sonic_vector)
music.core.io.write_wav_stereo(sonic_vector=stack,
                               filename='penta_effects.wav')
