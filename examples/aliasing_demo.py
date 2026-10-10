"""Render a direct A/B of high-frequency wavetable aliasing.

Run: python examples/aliasing_demo.py
The two WAVs use one shared gain factor, retaining their relative levels.
This is an acoustic example, not evidence about human hearing outcomes.
"""

import numpy as np
import soundfile as sf

import music

RATE = 44100
FREQ = 10000.0

legacy = music.note(FREQ, duration=1,
                    waveform_table=music.waveform_table("sawtooth"))
filtered = music.bandlimited_note(FREQ, duration=1, waveform="sawtooth")

# The truncated Fourier series can exceed +/-1 near discontinuities.
# Use one gain for both so their relative amplitude is not normalized away.
gain = 0.95 / max(np.max(np.abs(legacy)), np.max(np.abs(filtered)))
sf.write("sawtooth_mass_lut.wav", legacy * gain, RATE, subtype="PCM_24")
sf.write("sawtooth_bandlimited.wav", filtered * gain, RATE, subtype="PCM_24")
print("Rendered 10 kHz sawtooth in two variants at a common safe peak. "
      "For measurements run tools/benchmark_spectral_aliasing.py.")
