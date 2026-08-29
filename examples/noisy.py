""" Simple script that writes a sequence of noises into a WAV file,
    each separated by a short beep.
"""

import music

noises = ['brown', 'pink', 'white', 'blue', 'violet', 'black']
sonic_vector = []
silence = music.core.synths.silence(duration=0.4)

# A note starts and stops abruptly, which clicks. An ADSR envelope fades it
# in and out, so it makes a cleaner separator between the noises.
beep = music.adsr(sonic_vector=music.core.synths.note(duration=0.1),
                  attack_duration=10, release_duration=20)

for noise in noises:
    sonic_vector.append(music.core.synths.noises.noise(noise_type=noise))
    sonic_vector.append(silence)
    sonic_vector.append(beep)
    sonic_vector.append(silence)

sonic_vector.append(music.core.synths.noises.gaussian_noise())

stack = music.utils.horizontal_stack(*sonic_vector)
music.core.io.write_wav_mono(sonic_vector=stack,
                             filename='noisy.wav')

# The same note with and without the envelope, to hear what it does on its
# own: the plain one clicks at both ends, the shaped one does not.
plain = music.core.synths.note(duration=1)
shaped = music.adsr(sonic_vector=plain)
music.core.io.write_wav_mono(sonic_vector=plain, filename='beep_plain.wav')
music.core.io.write_wav_mono(sonic_vector=shaped, filename='beep_shaped.wav')
