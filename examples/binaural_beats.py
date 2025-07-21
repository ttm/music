"""Generate a binaural beat using Music's synthesis primitives.

This script creates two sine waves with a small frequency difference and
applies a gentle tremolo to each channel. Listening to the resulting
stereo file can help create a calm environment for relaxation or focus.
"""

import numpy as np
import music

BASE_FREQ = 440.0  # central frequency in Hz
BEAT_FREQ = 4.0    # difference between left and right in Hz
DURATION = 10.0    # seconds
TREMOLO_FREQ = 0.5  # Hz, slow amplitude modulation

left = music.note_with_phase(
    freq=BASE_FREQ - BEAT_FREQ / 2,
    duration=DURATION,
    waveform_table=music.tables.PrimaryTables().sine,
)
right = music.note_with_phase(
    freq=BASE_FREQ + BEAT_FREQ / 2,
    duration=DURATION,
    waveform_table=music.tables.PrimaryTables().sine,
)

left = music.tremolo(
    duration=DURATION,
    tremolo_freq=TREMOLO_FREQ,
    max_db_dev=3,
    sonic_vector=left,
)
right = music.tremolo(
    duration=DURATION,
    tremolo_freq=TREMOLO_FREQ,
    max_db_dev=3,
    sonic_vector=right,
)

stereo = np.vstack((left, right))

music.write_wav_stereo(stereo, "binaural_beats.wav")
