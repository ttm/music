"""The four IIR filter designs the MASS article specifies, applied to noise.

`music.iir` applies coefficients; `low_pass`, `high_pass`, `band_pass` and
`band_reject` compute them. Noise is the useful thing to hear them on,
because it starts with every frequency in it and what comes out is the
shape of the filter.

Cutoffs and centres are fractions of the sample rate -- `fraction_of` turns
a frequency in Hertz into one -- so the same coefficients describe the same
filter whatever rate the sound is at.
"""

import music

SAMPLE_RATE = 44100
DURATION = 1.2


def through(design, *args, **kwargs):
    """White noise, filtered, and normalised so the files are comparable."""
    a, b = design(*args, **kwargs)
    noise = music.noise(noise_type="white", duration=DURATION,
                        sample_rate=SAMPLE_RATE)
    return music.normalize_mono(music.iir(sonic_vector=noise, a=a, b=b))


# --- One filter at a time --------------------------------------------------
cutoff = music.fraction_of(1000, sample_rate=SAMPLE_RATE)
centre = music.fraction_of(1500, sample_rate=SAMPLE_RATE)
width = music.fraction_of(400, sample_rate=SAMPLE_RATE)

pieces = {
    "low_pass": through(music.low_pass, cutoff),
    "high_pass": through(music.high_pass, cutoff),
    "band_pass": through(music.band_pass, centre, width),
    "band_reject": through(music.band_reject, centre, width),
}
for name, sound in pieces.items():
    music.write_wav_mono(sonic_vector=sound,
                         filename=f"filter_{name}.wav")
    print(f"filter_{name}.wav")

# --- A sweep ---------------------------------------------------------------
# The same low pass, opening from 200 Hz to 8 kHz over sixteen steps. One
# pole, so it is 6 dB per octave and gentle; the article says as much, and
# names biquad recipes for when that is not enough.
steps = 16
segment = DURATION / steps
sweep = []
for step in range(steps):
    hertz = 200 * (8000 / 200) ** (step / (steps - 1))
    a, b = music.low_pass(music.fraction_of(hertz, sample_rate=SAMPLE_RATE))
    noise = music.noise(noise_type="white", duration=segment,
                        sample_rate=SAMPLE_RATE)
    sweep.append(music.normalize_mono(
        music.iir(sonic_vector=noise, a=a, b=b)))

music.write_wav_mono(sonic_vector=music.horizontal_stack(*sweep),
                     filename="filter_sweep.wav")
print(f"filter_sweep.wav: a low pass opening from 200 Hz to 8 kHz "
      f"in {steps} steps")
