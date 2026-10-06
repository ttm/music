"""Sing the same songs with both of the package's singers, side by side.

``music.sing`` has two backends, and they sing the same score at the same
pitches, so the difference between them is the singing itself. The
default, ``psola``, is the package's own: espeak-ng says each syllable,
again and slower when its note is longer, and Praat's pitch-synchronous
overlap-add holds it at the note's pitch, lengthening only the middle of
its vowel and singing a long note with a vibrato, louder on the beats.
``ecantorix`` is the engine it replaced, kept as the reference.

Each song is written once a backend, ``<song>.psola.wav`` and
``<song>.ecantorix.wav``, to be listened to in turn. The slow song is
where they differ most: on a long note eCantorix holds what espeak said
at its slowest, and PSOLA holds a vowel.

The psola backend needs espeak-ng (``sudo apt install espeak-ng``, or
``brew install espeak-ng``) and ``pip install 'music[singing]'``.
eCantorix needs its engine, which ``music.singing.setup_engine()``
clones, and Perl, abc2midi and sox; where a backend cannot sing, this
says what it lacks and sings with the other.
"""

import music

SONGS = {
    "twinkle": dict(
        text="Twin-kle, twin-kle, lit-tle star, how I won-der what you are",
        notes=(0, 0, 7, 7, 9, 9, 7, 5, 5, 4, 4, 2, 2, 0),
        durs=(1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2)),
    "frere_jacques": dict(
        text="Frè-re Jac-ques, frè-re Jac-ques, dor-mez vous? dor-mez vous?",
        notes=(0, 2, 4, 0, 0, 2, 4, 0, 4, 5, 7, 4, 5, 7),
        durs=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2), lang="fr"),
    "alle_meine_entchen": dict(
        text="Al-le mei-ne Ent-chen schwim-men auf dem See",
        notes=(0, 2, 4, 5, 7, 7, 9, 9, 9, 9, 7),
        durs=(1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 4), lang="de"),
    # A quarter note a second: notes of one and two seconds.
    "twinkle_slow": dict(
        text="Twin-kle, twin-kle, lit-tle star", Q=60,
        notes=(0, 0, 7, 7, 9, 9, 7), durs=(1, 1, 1, 1, 1, 1, 2)),
}

singers = ["psola", "ecantorix"]
for name, song in SONGS.items():
    for backend in list(singers):
        try:
            sound = music.sing(backend=backend, **song)
        except RuntimeError as missing:
            print(f"{backend} cannot sing here, and is left out: {missing}")
            singers.remove(backend)
            continue
        filename = f"{name}.{backend}.wav"
        music.write_wav_mono(sound, filename)
        print(f"{filename}: {len(sound) / 44100:.1f} s")
