"""Auditory stimuli for sensory-stimulation work.

Each routine here renders one technique catalogued in **SSTIM**, the
Sensory Stimulation Vocabulary developed in the W3C Sensory Stimulation
Vocabulary Community Group. Every function names the SSTIM term it
implements and links its IRI, so a rendered stimulus can be described in
the same words a protocol, a dataset or a device uses:

===============================  ==========================================
Function                         SSTIM technique
===============================  ==========================================
:func:`binaural_beats`           ``sstim-v:techBinauralBeats``
:func:`monaural_beats`           ``sstim-v:techMonauralBeats``
:func:`isochronic_tones`         ``sstim-v:techIsochronicTones``
:func:`amplitude_modulation`     ``sstim-v:techAmplitudeModulation``
:func:`frequency_modulation`     ``sstim-v:techFrequencyModulation``
:func:`modulated_noise`          ``sstim-v:techBroadbandNoise``
:func:`spatial_motion`           ``sstim-v:techSpatialAuditory``
===============================  ==========================================

with ``sstim-v:`` standing for ``https://w3id.org/sstim/vocab#``. All
seven are members of ``sstim-v:TechniqueScheme``; the last two are typed
there as non-entrainment techniques, which is a claim about what they do
rather than about how they are built.

One distinction SSTIM draws is worth carrying into the code, because it
decides what a recording of the output contains. SSTIM records on each
rendering mechanism whether it *puts a signal physically into the world*
or whether the signal is *constructed by the nervous system*. Six of
these seven put a real modulation into the air, and a spectrum of the
rendered audio shows it. :func:`binaural_beats` does not: each ear
receives a steady tone, neither channel contains the beat, and the beat
exists only for a listener hearing both. Measure the file and it is not
there. Each docstring states which case it is.

A protocol is a sequence of these rather than one of them, and
:class:`StimulationSession` is that sequence: phases in order, each with
its own stimulus and duration, joined by ramps rather than by cuts.

These routines build on the package's own primitives and follow its
sample-by-sample model: a modulator is folded into the wavetable lookup
rather than applied to a finished sound.

References
----------
.. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
       representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)
.. [2] SSTIM, the Sensory Stimulation Vocabulary.
       https://w3id.org/sstim
"""

from .session import StimulationSession, StimulusPhase
from .stimuli import (
    amplitude_modulation,
    binaural_beats,
    frequency_modulation,
    isochronic_tones,
    modulated_noise,
    monaural_beats,
    spatial_motion,
)

__all__ = [
    'StimulationSession',
    'StimulusPhase',
    'amplitude_modulation',
    'binaural_beats',
    'frequency_modulation',
    'isochronic_tones',
    'modulated_noise',
    'monaural_beats',
    'spatial_motion',
]
