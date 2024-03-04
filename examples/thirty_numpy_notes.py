import music

# 1) start a ѕynth
being = music.legacy.Being()

# 2) set its parameters using sequences to be iterated through
being.d_ = [1/2, 1/4, 1/4]  # durations in seconds
being.fv_ = [0, 1,5,15,150,1500,15000]  # vibrato frequency
being.nu_ = [5]  # vibrato depth in semitones (maximum deviation of pitch)
being.f_ = [220, 330]  # frequencies for the notes

# 3) Use numpy arrays directly and use them to concatenate and/or mix sounds:
s1 = being.render(30)
being.f_ += [440]
being.fv_ = [1,2,3,4,5]
s2 = being.render(30)

# s1 then s2 then s1 and s2 at the same time, then at the same time but one in each LR channel,
# then s1 times s2 reversed, then s1+s2 but jumping 6 samples before using one:
s3 = music.utils.horizontal_stack(s1, s2, s1 + s2, (s1, s2),
       s1*s2[::-1],
       s1[::7] + s2[::7])
music.core.io.write_wav_stereo(s3, 'thirty_numpy_notes.wav')
