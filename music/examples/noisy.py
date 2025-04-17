""" Simple script that writes a pentatonic scale on a WAV file
    with different effects.
"""

import music

noises = ['brown', 'pink', 'white', 'blue', 'violet', 'black']
sonic_vector = []
silence = music.core.synths.silence(duration=0.4)
beep = music.core.synths.note(duration=0.1)

for noise in noises:
    sonic_vector.append(music.core.synths.noises.noise(noise_type=noise))
    sonic_vector.append(silence)
    sonic_vector.append(beep)
    sonic_vector.append(silence)

sonic_vector.append(music.core.synths.noises.gaussian_noise())

stack = music.utils.horizontal_stack(*sonic_vector)
music.core.io.write_wav_stereo(sonic_vector=stack,
                               filename='noisy.wav')
