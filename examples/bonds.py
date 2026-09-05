"""Bonds: a piece deciding once how its notes behave.

The MASS article's equation `eq:vinculos` says the vibrato rate, the
tremolo rate and their depths may each be a function of the note's
frequency, and that those functions "are arbitrary and dependent on musical
intentions". `music.Bonds` is where such a function goes.

The point is not that any one set of bonds is right. It is that binding the
characteristics to the pitch, rather than setting them per note, is what
makes a run of notes sound like one instrument instead of a list -- which
is what the article means by using this to build a musical language.

Writes three passages over the same notes, so the difference is the bonds
and nothing else.
"""

import music

TONIC = 220.0
SAMPLE_RATE = 44100

# One line, played by each of the three below: a minor scale up and back.
degrees = list(music.scale("minor")) + [12] + list(reversed(
    music.scale("minor")))
line = music.pitch_to_freq(start_freq=TONIC, semitones=degrees)


def write(name, bonds, duration=0.28):
    sound = bonds.render(line, duration=duration, sample_rate=SAMPLE_RATE)
    music.write_wav_mono(sonic_vector=music.normalize_mono(sound),
                         filename=f"bonds_{name}.wav")
    low, high = min(line), max(line)
    print(f"bonds_{name}.wav: "
          f"{bonds} from {low:.0f} Hz to {high:.0f} Hz")


# --- Nothing bound ---------------------------------------------------------
# The line as plain notes, for comparison.
write("plain", music.Bonds())

# --- The article's own two examples ----------------------------------------
# "a vibrato frequency proportional to note pitch", so the vibrato speeds up
# as the line rises; and a depth inversely proportional to it, so the higher
# notes waver less. Together they read as one voice getting tighter as it
# climbs.
write("proportional", music.Bonds(
    vibrato_freq=music.proportional(1 / 40),
    max_pitch_dev=music.inversely_proportional(400),
))

# --- A decision by register ------------------------------------------------
# Bonds need not be continuous. Below middle C this voice is slow and deep;
# above it, fast and shallow, with a tremolo that only appears up there.
middle_c = 261.63
write("registers", music.Bonds(
    vibrato_freq=music.stepped([(middle_c, 3.0)], otherwise=9.0),
    max_pitch_dev=music.stepped([(middle_c, 1.5)], otherwise=0.4),
    tremolo_freq=music.stepped([(middle_c, 0.0)], otherwise=7.0),
    max_db_dev=music.stepped([(middle_c, 0.0)], otherwise=6.0),
))
