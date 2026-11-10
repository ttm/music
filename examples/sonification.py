"""Data sonification: global temperature since 1880, heard three ways.

A sonification decides what a number becomes. Here a year's temperature
becomes a pitch, and the decisions are the constants below: the pitch of
the value the data are measured from, the interval a degree is worth and
the time a year takes. They are what the axes are to a chart, and as much
a choice. The interval is counted in semitones because pitch is heard in
ratios: equal differences in temperature are then equal intervals,
wherever in the record they fall.

Writes three files, the same data under the same mapping:

- sonification_notes.wav, a note for each year;
- sonification_scale.wav, each note moved to the nearest degree of a
  scale, which sounds more like music and is less like the data, by an
  amount the script prints;
- sonification_line.wav, one line through the years, as a chart would
  draw it, with what a chart would draw around it: a drone for the
  reference and a tick for each decade. Without them a listener has
  only the line to compare the line to.

To hear another series, replace the numbers and choose the constants
again.
"""

import numpy as np

import music
from music.core.synths.mipmap import mipmap_frequency_path

# --- The data --------------------------------------------------------------
# Each year's global mean surface temperature, over land and ocean, as its
# difference from the mean of 1951-1980 in hundredths of a degree Celsius,
# a decade to a row. They are the annual means, the column J-D, of
# GLB.Ts+dSST.csv, from:
#
#   GISTEMP Team, 2026: GISS Surface Temperature Analysis (GISTEMP),
#   version 4. NASA Goddard Institute for Space Studies. Dataset accessed
#   2026-11-10 at https://data.giss.nasa.gov/gistemp/.
#
#   Lenssen, N., G.A. Schmidt, M. Hendrickson, P. Jacobs, M. Menne, and
#   R. Ruedy, 2024: A GISTEMPv4 observational uncertainty ensemble.
#   J. Geophys. Res. Atmos., 129, no. 17, e2023JD040179,
#   doi:10.1029/2023JD040179.
FIRST_YEAR = 1880
HUNDREDTHS = """
    -17  -9 -11 -18 -28 -33 -31 -36 -18 -11
    -36 -22 -27 -31 -30 -23 -11 -11 -28 -18
     -8 -16 -28 -37 -47 -26 -22 -38 -42 -48
    -44 -44 -37 -34 -15 -14 -36 -46 -30 -27
    -27 -18 -28 -26 -27 -22 -11 -22 -20 -36
    -15  -9 -16 -28 -13 -20 -15  -3   0  -2
     12  18   6   9  20   9  -7  -3 -11 -11
    -17  -7   1   8 -13 -14 -19   5   6   3
     -2   6   3   5 -20 -11  -6  -2  -8   5
      3  -8   1  16  -7  -1 -10  18   7  16
     26  32  14  31  16  12  18  32  39  27
     45  41  22  23  31  45  33  47  61  38
     39  53  63  62  53  68  64  66  54  66
     72  61  65  67  75  89 101  91  85  98
    101  85  89 117 128 118
"""

# --- The mapping -----------------------------------------------------------
TONIC = 440.0              # Hz: the pitch of zero, the mean of 1951-1980
SEMITONES_PER_DEGREE = 12  # a degree Celsius is an octave, and up is warmer
YEAR = 0.1                 # seconds: a decade lasts one
SCALE = "dorian"           # the second file's: any name in music.SCALES

celsius = np.array(HUNDREDTHS.split(), dtype=float) / 100
years = FIRST_YEAR + np.arange(len(celsius))
semitones = SEMITONES_PER_DEGREE * celsius


def melody(pitches):
    """A note for each pitch, in semitones from the tonic, end to end."""
    # Each note is shaped before the notes are joined: raw ones click
    # where one stops and the next starts.
    return music.horizontal_stack(*[
        music.adsr(sonic_vector=music.note(freq, duration=YEAR))
        for freq in music.pitch_to_freq(TONIC, pitches)])


# --- A note for each year --------------------------------------------------
notes = melody(semitones)
music.write_wav_mono(notes, "sonification_notes.wav")
low, high = music.pitch_to_freq(TONIC, (semitones.min(), semitones.max()))
print(f"sonification_notes.wav: {len(years)} years from {years[0]}, "
      f"{1 / YEAR:g} a second, from {low:.0f} Hz in "
      f"{years[celsius.argmin()]} to {high:.0f} Hz in "
      f"{years[celsius.argmax()]}")

# --- The same notes, on a scale --------------------------------------------
# The degrees of the scale in every octave the data reach, and for each
# year the nearest of them. Which scale is a choice the data have no say
# in, and it lends them a mood.
degrees = np.array(music.scale(SCALE))
octaves = range(int(semitones.min() // 12), int(semitones.max() // 12) + 2)
ruler = np.concatenate([degrees + 12 * octave for octave in octaves])
on_scale = ruler[np.abs(semitones[:, None] - ruler).argmin(axis=1)]
music.write_wav_mono(melody(on_scale), "sonification_scale.wav")

# What the scale costs, in the data's own unit.
moved = np.abs(on_scale - semitones) / SEMITONES_PER_DEGREE
changes = np.diff(celsius) != 0
merged = changes & (np.diff(on_scale) == 0)
print(f"sonification_scale.wav: the {SCALE} scale moves a year by "
      f"{moved.mean():.2f} of a degree on average and {moved.max():.2f} "
      f"at most, and plays {merged.sum()} of the {changes.sum()} changes "
      f"from one year to the next as one note repeated")

# --- One line, with axes ---------------------------------------------------
# A frequency for every sample the notes last. Each year's pitch is
# reached in the middle of the year's time, and the pitch moves straight
# from one year's to the next, as the line of a chart does: what lies
# between two years is the interpolation's, not a measurement. The
# oscillator keeps its phase through every change, so nothing is joined
# and nothing clicks.
seconds = np.arange(len(notes)) / music.get_sample_rate()
middles = (np.arange(len(years)) + 0.5) * YEAR
path = np.interp(seconds, middles, semitones)
line = mipmap_frequency_path(TONIC * 2 ** (path / 12), waveform="triangle")

# The axes. The drone sounds an octave below the reference, so a year an
# octave above the drone is at the mean of 1951-1980; on the reference
# itself, the line would beat against it in crossing. A tick of noise
# opens each decade, louder at the turn of a century; it has no pitch, so
# it cannot be taken for data.
drone = music.note(TONIC / 2, number_of_samples=len(line))
tick = music.fade(sonic_vector=music.noise("white", duration=0.05, seed=0))
decades = np.flatnonzero(years % 10 == 0)
clock = music.mix_many(
    [tick * (2 if years[index] % 100 == 0 else 1) for index in decades],
    offset=list(decades * YEAR))

# A sustain of 0 dB leaves the sound as it is but for its two ends.
tones = music.adsr(sonic_vector=line + music.db_to_amp(-12) * drone,
                   sustain_level=0)
music.write_wav_mono(music.mix_many([tones, music.db_to_amp(-6) * clock]),
                     "sonification_line.wav")
print(f"sonification_line.wav: one line through the same pitches, over "
      f"a drone at {TONIC / 2:.0f} Hz, an octave under the reference, "
      f"and a tick each decade")
