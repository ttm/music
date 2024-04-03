import music

##############
# Notice that you might relate a peal or any set of permutations
# to a sonic characteristic (frequency, duration, vibrato depth, vibrato frequency,
# attack duration, etc) through at least 3 methods:
# 1) initiate a Being(), set its permutations to the permutation sequence,
# its domain to the values to be permuted, and its curseq to
# the name of the Being sequence to be yielded by the permutation of the domain.
#
# 2) Achieve the sequence of values through peal.act() or just using permutation(domain)
# for all the permutations at hand.
# Then render the notes directly (e.g. using M.core.V_) or passing the sequence of values
# to a synth, such as Being()

pe3 = music.structures.peals.PlainChanges.PlainChanges(3)
music.structures.symmetry.print_peal(pe3.act(), [0])
freqs = sum(pe3.act([220,440,330]), [])

nnotes = len(freqs)

being = music.legacy.Being()
being.f_ = freqs
being.render(nnotes, 'campanology_1.wav')

### OR
being = music.legacy.Being()
being.domain = [220, 440, 330]
being.perms = pe3.peal_direct
being.f_ = []
being.curseq = 'f_'
being.stay(nnotes)
being.render(nnotes, 'campanology_2.wav')
