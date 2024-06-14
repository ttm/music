import music

tables = music.tables.PrimaryTables()
pe3 = music.structures.PlainChanges(3)
music.structures.symmetry.print_peal(pe3.act(), [0])
freqs = sum(pe3.act([220, 440, 330]), [])

isynth = music.legacy.IteratorSynth()
isynth.fundamental_frequency_sequence = freqs
isynth.tab_sequence = [tables.sine, tables.triangle, tables.square, tables.saw]

pcm_samples = music.utils.horizontal_stack(*[isynth.renderIterate()
                                             for i in range(len(freqs))])

music.core.io.write_wav_mono(pcm_samples, 'isynth.wav')
