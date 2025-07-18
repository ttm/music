from music.legacy import Being
from music.utils import horizontal_stack
from music.core.io import write_wav_stereo

# 1) start a ѕynth
being = Being()

# 3) Use numpy arrays directly and use them to concatenate and/or mix sounds:
s1 = being.render(30)
being.f_ += [440]
being.fv_ = [1, 2, 3, 4, 5]
s2 = being.render(30)

# s1 then s2 then s1 and s2 at the same time, then at the same time but one in
# each LR channel, then s1 times s2 reversed, then s1+s2 but jumping 6 samples
# before using one:
s3 = horizontal_stack(s1, s2, s1 + s2, (s1, s2), s1*s2[::-1], s1[::7] +
                      s2[::7])
write_wav_stereo(s3, 'thirty_numpy_notes.wav')
