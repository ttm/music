"""Top-level package for basic audio synthesis utilities."""

from importlib.metadata import (PackageNotFoundError as _NotFound,
                                version as _version)
from typing import TYPE_CHECKING as _TYPE_CHECKING, Any as _Any

try:
    __version__ = _version("music")
except _NotFound:  # pragma: no cover - running from a checkout
    __version__ = "0.0.0.dev0"

from .utils import (
    WAVEFORM_SAWTOOTH,
    WAVEFORM_SINE,
    WAVEFORM_SQUARE,
    WAVEFORM_TRIANGULAR,
    WAVEFORMS,
    amp_to_db,
    convert_to_stereo,
    db_to_amp,
    horizontal_stack,
    hz_to_midi,
    midi_to_hz_interval,
    midi_to_hz,
    mix_many_with_offsets,
    mix_many,
    mix_stereo,
    mix_with_offset_,
    mix_with_offset,
    mix,
    mix2,
    pan_transitions,
    pitch_to_freq,
    profile,
    resolve_stereo,
    rhythm_to_durations,
    waveform_table
)
from .core import (
    adsr_stereo,
    adsr_vibrato,
    adsr,
    am,
    cross_fade,
    fade,
    fir,
    gaussian_noise,
    iir,
    localize,
    localize2,
    localize_linear,
    loud,
    louds,
    noise,
    normalize_mono,
    normalize_stereo,
    note_with_doppler,
    note_with_fm,
    note_with_glissando_vibrato,
    note_with_glissando,
    note_with_phase,
    note_with_two_vibratos_glissando,
    note_with_two_vibratos,
    note_with_vibrato_seq_localization,
    note_with_vibrato,
    note_with_vibratos_glissandos,
    note,
    read_audio,
    read_wav,
    reverb,
    silence,
    stretches,
    tremolo,
    tremolos,
    trill,
    write_audio,
    write_wav_mono,
    write_wav_stereo,
    play_audio,
)
from .tables import PrimaryTables
from .stimulation import (
    StimulationSession,
    StimulusPhase,
    amplitude_modulation,
    binaural_beats,
    frequency_modulation,
    isochronic_tones,
    modulated_noise,
    monaural_beats,
    spatial_motion,
)
from .singing import get_engine, make_test_song, setup_engine
from .legacy import Being, CanonicalSynth, IteratorSynth
from .sequencer import Sequencer

# The permutation and change-ringing structures are reached through
# ``__getattr__`` rather than imported here, because importing them means
# importing sympy, and importing sympy means importing the computer algebra
# system it sits on -- ``sympy.polys`` and the rest. That cost about half of
# ``import music`` (roughly 550-830 ms down to 225-320 ms measured warm on
# 3.12) and was paid by everyone, including the majority of callers who only
# synthesize sound and never touch a peal.
#
# Nothing about the API changes: ``music.Peals``, ``from music import Peals``
# and ``help(music.Peals)`` all work, and sympy loads on the first of them.
# ``music.structures`` can still be imported directly for the eager path.
#
# This is the same bargain matplotlib already gets here, one layer up: an
# expensive dependency that a minority of the package needs should be paid
# for by the callers who need it.
_LAZY_STRUCTURES = frozenset({
    'dist',
    'GenericPeal',
    'InterestingPermutations',
    'Peals',
    'PlainChanges',
    'print_peal',
    'transpose_permutation',
})

if _TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from .structures import (  # noqa: F401
        dist,
        GenericPeal,
        InterestingPermutations,
        Peals,
        PlainChanges,
        print_peal,
        transpose_permutation,
    )


def __getattr__(name: str) -> _Any:
    """Resolve the structures exports on first use.

    PEP 562. Anything not deferred raises ``AttributeError`` as it would
    have without this hook, so a typo still fails the way a reader expects
    rather than reporting an import error from somewhere else.
    """
    if name in _LAZY_STRUCTURES:
        from . import structures
        return getattr(structures, name)
    if name == 'structures':
        # `music.structures` used to be bound as a side effect of
        # importing the names above. Deferring that import took the
        # submodule attribute with it, and `music.structures.peals`
        # stopped resolving -- which the examples use and the tests did
        # not cover.
        import importlib
        return importlib.import_module('.structures', __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Keep the deferred names visible to ``dir()`` and to tab completion."""
    return sorted(set(globals()) | _LAZY_STRUCTURES | {'structures'})


__all__ = [
    'WAVEFORMS',
    'WAVEFORM_SAWTOOTH',
    'WAVEFORM_SINE',
    'WAVEFORM_SQUARE',
    'WAVEFORM_TRIANGULAR',
    'waveform_table',
    'StimulationSession',
    'StimulusPhase',
    'amplitude_modulation',
    'binaural_beats',
    'frequency_modulation',
    'isochronic_tones',
    'modulated_noise',
    'monaural_beats',
    'spatial_motion',
    '__version__',
    'adsr_stereo',
    'adsr_vibrato',
    'adsr',
    'am',
    'amp_to_db',
    'Being',
    'CanonicalSynth',
    'convert_to_stereo',
    'cross_fade',
    'db_to_amp',
    'dist',
    'fade',
    'fir',
    'gaussian_noise',
    'GenericPeal',
    'get_engine',
    'setup_engine',
    'horizontal_stack',
    'hz_to_midi',
    'iir',
    'InterestingPermutations',
    'IteratorSynth',
    'localize',
    'localize2',
    'localize_linear',
    'loud',
    'louds',
    'make_test_song',
    'midi_to_hz_interval',
    'midi_to_hz',
    'mix_many_with_offsets',
    'mix_many',
    'mix_stereo',
    'mix_with_offset_',
    'mix_with_offset',
    'mix',
    'mix2',
    'noise',
    'normalize_mono',
    'normalize_stereo',
    'note_with_doppler',
    'note_with_fm',
    'note_with_glissando_vibrato',
    'note_with_glissando',
    'note_with_phase',
    'note_with_two_vibratos_glissando',
    'note_with_two_vibratos',
    'note_with_vibrato_seq_localization',
    'note_with_vibrato',
    'note_with_vibratos_glissandos',
    'note',
    'pan_transitions',
    'Peals',
    'pitch_to_freq',
    'PlainChanges',
    'PrimaryTables',
    'print_peal',
    'profile',
    'read_audio',
    'read_wav',
    'resolve_stereo',
    'reverb',
    'rhythm_to_durations',
    'silence',
    'stretches',
    'transpose_permutation',
    'tremolo',
    'tremolos',
    'trill',
    'write_audio',
    'write_wav_mono',
    'write_wav_stereo',
    'play_audio',
    'Sequencer'
]
