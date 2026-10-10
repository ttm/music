"""Synthesis primitives for envelopes, notes, and noises."""

from .bandlimited import bandlimited_note
from .envelopes import am, tremolo, tremolos
from .notes import (
    note, note_with_doppler, note_with_fm, note_with_glissando,
    note_with_glissando_vibrato, note_with_phase, note_with_two_vibratos,
    note_with_two_vibratos_glissando, note_with_vibrato,
    note_with_vibrato_seq_localization, note_with_vibratos_glissandos, trill
)
from .noises import gaussian_noise, noise, silence
from .oversampling import render_oversampled

__all__ = [
    'am',
    'bandlimited_note',
    'gaussian_noise',
    'note',
    'note_with_doppler',
    'note_with_fm',
    'note_with_glissando',
    'note_with_glissando_vibrato',
    'note_with_phase',
    'note_with_vibrato',
    'note_with_two_vibratos',
    'note_with_two_vibratos_glissando',
    'note_with_vibratos_glissandos',
    'note_with_vibrato_seq_localization',
    'noise',
    'render_oversampled',
    'silence',
    'tremolo',
    'tremolos',
    'trill',
]
