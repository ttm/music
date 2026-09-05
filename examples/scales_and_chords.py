"""Scales and chords, from the theory the MASS companion paper states.

`music.theory` counts semitones from a tonic of zero, which is what
`pitch_to_freq` takes, so getting from a named scale to a sound is two
steps: name the degrees, turn them into frequencies, render each.

Writes three files: the seven modes one after another, a cadence of four
chords, and the first sixteen partials of the harmonic series played over
their own fundamental so the tuning can be heard against it.
"""

import music

TONIC = 220.0          # A3
NOTE_DURATION = 0.32
SAMPLE_RATE = 44100


def render_scale(name, start_freq=TONIC, duration=NOTE_DURATION):
    """One scale, ascending, as a single sonic vector."""
    freqs = music.pitch_to_freq(start_freq=start_freq,
                                semitones=music.scale(name))
    # Close it on the octave, so a mode ends where it started.
    freqs = list(freqs) + [start_freq * 2]
    notes = [music.note(freq=freq, duration=duration,
                        sample_rate=SAMPLE_RATE) for freq in freqs]
    return music.horizontal_stack(*notes)


def render_chord(name, root=0, duration=1.0):
    """One chord, all of its notes sounding together."""
    freqs = music.pitch_to_freq(start_freq=TONIC,
                                semitones=music.chord(name, root=root))
    notes = [music.note(freq=freq, duration=duration,
                        sample_rate=SAMPLE_RATE) for freq in freqs]
    # mix_many sums them sample by sample, which is what a chord is.
    return music.adsr(sonic_vector=music.mix_many(notes),
                      envelope_duration=duration, sample_rate=SAMPLE_RATE)


# --- The seven modes -------------------------------------------------------
# Every one of them is a rotation of a single step pattern, so this list is
# a convenience: music.mode_by_rotation(kappa) reaches the same seven from
# that pattern directly, for kappa in range(7).
modes = ("ionian", "dorian", "phrygian", "lydian", "mixolydian", "aeolian",
         "locrian")

all_modes = music.horizontal_stack(*[render_scale(name) for name in modes])
music.write_wav_mono(sonic_vector=music.normalize_mono(all_modes),
                     filename="modes.wav")
print(f"modes.wav: {', '.join(modes)}")

# --- A cadence -------------------------------------------------------------
# I - vi - IV - V in A: the roots are scale degrees of the major scale, and
# the qualities are what the harmony of that scale gives them.
major = music.scale("major")
cadence = music.horizontal_stack(
    render_chord("major", root=major[0]),
    render_chord("minor", root=major[5]),
    render_chord("major", root=major[3]),
    render_chord("dominant seventh", root=major[4]),
)
music.write_wav_mono(sonic_vector=music.normalize_mono(cadence),
                     filename="cadence.wav")
print("cadence.wav: I - vi - IV - V7 in A")

# --- The harmonic series ---------------------------------------------------
# Sixteen partials over a held fundamental. The octaves land on the tempered
# scale exactly and nothing else does: the third partial is 2 cents sharp of
# a fifth, the seventh 31 cents flat of a minor seventh.
partials = music.harmonic_series(16)
fundamental = music.note(freq=TONIC, duration=len(partials) * NOTE_DURATION,
                         sample_rate=SAMPLE_RATE)
over_it = music.horizontal_stack(
    *[music.note(freq=TONIC * music.midi_to_hz_interval(semitones),
                 duration=NOTE_DURATION, sample_rate=SAMPLE_RATE)
      for semitones in partials])
series = music.mix(fundamental * 0.5, over_it)
music.write_wav_mono(sonic_vector=music.normalize_mono(series),
                     filename="harmonic_series.wav")
print(f"harmonic_series.wav: {len(partials)} partials over {TONIC} Hz")
