"""Sing a short phrase, and write it to a WAV file.

``music.sing`` needs espeak-ng (``sudo apt install espeak-ng``, or
``brew install espeak-ng``) and ``pip install 'music[singing]'``. With
``backend="ecantorix"`` it sings the same score with the eCantorix engine,
which ``music.singing.setup_engine()`` clones, and which needs Perl,
abc2midi and sox.
"""

import music

sound = music.sing(text="Mar-ry had a litt-le lamb",
                   notes=(4, 2, 0, 2, 4, 4, 4), durs=(1, 1, 1, 1, 1, 1, 2))
music.write_wav_mono(sonic_vector=sound, filename="singing_demo.wav")
print(f"singing_demo.wav: Mary had a little lamb, {len(sound) / 44100:.1f} s")
