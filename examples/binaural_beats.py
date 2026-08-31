"""Generate a binaural beat, with a slow tremolo over it.

The stimulus itself is one call: `music.binaural_beats` renders the
technique SSTIM catalogues as `sstim-v:techBinauralBeats`, two sine
tones split symmetrically about a centre frequency, one per ear.

Play this over headphones. The beat is not in either channel -- it is
constructed by the listener from the two of them -- so anything that
downmixes the file to mono destroys the stimulus and leaves a monaural
beat, which is a different technique.
"""

import numpy as np

import music

BASE_FREQ = 440.0   # central frequency in Hz
BEAT_FREQ = 4.0     # difference between left and right in Hz
DURATION = 10.0     # seconds
TREMOLO_FREQ = 0.5  # Hz, slow amplitude modulation

stereo = music.binaural_beats(
    carrier_freq=BASE_FREQ,
    beat_freq=BEAT_FREQ,
    duration=DURATION,
)

# A gentle tremolo, applied equally to both channels so that it does not
# disturb the frequency difference the beat depends on.
left, right = (
    music.tremolo(duration=DURATION, tremolo_freq=TREMOLO_FREQ,
                  max_db_dev=3, sonic_vector=channel)
    for channel in stereo
)

music.write_wav_stereo(np.vstack((left, right)), "binaural_beats.wav")
