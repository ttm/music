from music.legacy import Being

# 1) start a synth
being = Being()

# 2) set its parameters using sequences to be iterated through
being.d_ = [1/2, 1/4, 1/4]  # durations in seconds
being.fv_ = [0, 1, 5, 15, 150, 1500, 15000]  # vibrato frequency
being.nu_ = [5]  # vibrato depth in semitones (maximum deviation of pitch)
being.f_ = [220, 330]  # frequencies for the notes

# 3) render the wavfile with 30 notes iterating though the lists above
being.render(30, 'thirty_notes.wav')
