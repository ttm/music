"""Sing eCantorix's effects with both of the package's singers.

eCantorix's extra voices give it three effects, and ``music.sing`` sings
each with either backend: ``tremolo``, a nine-hertz tremolo on each note
and a room after it; ``melt``, the same with espeak's female voice, its
formants a quarter higher, sinking with the pitch below 216 Hz; and
``flite``, flite's voice in place of espeak's, named by ``lang``. The
psola backend makes them with the package's own tremolo, reverberation
and resampling, set to what eCantorix's sox gives, as measured.

Each is written once a backend, ``mary.<effect>.<backend>.wav``: in
stereo for the tremolo and the melted voice, which ring a second past
the line, as eCantorix's do. The flite effect needs ``flite``
(``sudo apt install flite``, or ``brew install flite``); see
``singing_backends.py`` for what each backend needs. Where one cannot
sing an effect, this says what it lacks and goes on.
"""

import music

MARY = dict(text="Mar-ry had a litt-le lamb", notes=(4, 2, 0, 2, 4, 4, 4),
            durs=(1, 1, 1, 1, 1, 1, 2))

# flite names its voices rather than its languages: slt is a woman's
# voice, and sings in tune with both backends.
EFFECTS = {"tremolo": {}, "melt": {}, "flite": {"lang": "slt"}}

for effect, voice in EFFECTS.items():
    for backend in ("psola", "ecantorix"):
        try:
            sound = music.sing(backend=backend, effect=effect,
                               **MARY, **voice)
        except (RuntimeError, ValueError) as missing:
            print(f"{effect} with {backend} is left out: {missing}")
            continue
        filename = f"mary.{effect}.{backend}.wav"
        if sound.ndim == 2:
            music.write_wav_stereo(sound, filename)
        else:
            music.write_wav_mono(sound, filename)
        print(f"{filename}: {sound.shape[-1] / 44100:.1f} s, "
              f"{'stereo' if sound.ndim == 2 else 'mono'}")
