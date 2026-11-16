"""Plain changes on three bells, rung on frequencies, two ways.

A peal, or any sequence of permutations, can be tied to a characteristic
of the sound -- its frequency, its duration, the depth or rate of its
vibrato, its attack -- in either of two ways:

1) Act the peal on the values with ``peal.act(values)``, which gives the
   rows, and hand them to a synth such as ``Being``, or render each note
   directly with ``music.note`` or ``music.note_with_vibrato``.
2) Give a ``Being`` the permutations, the values they permute as its
   ``domain``, and the name of the sequence they make as its ``curseq``,
   and let it permute the values as it plays.

This writes the first as campanology_1.wav and the second as
campanology_2.wav; they ring the same changes.
"""
import music

pe3 = music.structures.peals.PlainChanges(3)
music.structures.symmetry.print_peal(pe3.act(), [0])
freqs = sum(pe3.act([220, 440, 330]), [])

nnotes = len(freqs)

being = music.legacy.Being()
being.f_ = freqs
being.render(nnotes, 'campanology_1.wav')

# OR
being = music.legacy.Being()
being.domain = [220, 440, 330]
being.perms = pe3.peal_direct
being.f_ = []
being.curseq = 'f_'
being.stay(nnotes)
being.render(nnotes, 'campanology_2.wav')
