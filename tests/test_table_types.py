"""Every oscillator returns float64, whatever type its tables are.

Most read a waveform table as it was given and returned what it held, so
a table of float32 gave a note of float32 and a table of integers a note
of integers, against the float64 each is annotated to return. These hold
every routine with a table to float64, and to the very samples the same
table gives as float64.
"""

import inspect

import numpy as np
import pytest

import music

RATE = 8000
SHORT = 0.05

#: The routines that read a table, found rather than listed, so that a new
#: one is held to this too.
ROUTINES = {}
for _name in music.__all__:
    _routine = getattr(music, _name)
    if callable(_routine) and not inspect.isclass(_routine):
        _tables = [parameter for parameter in
                   inspect.signature(_routine).parameters
                   if parameter.endswith(("table", "tables"))]
        if _tables:
            ROUTINES[_name] = _tables


def _as(table, dtype):
    if dtype is np.int16:
        # Whole values, so the integers hold the table exactly: a square.
        return np.sign(table).astype(np.int16)
    return table.astype(dtype)


def _laid_out(default, table):
    """`table` in the layout of `default`: one, or rows of them."""
    if isinstance(default, np.ndarray):
        return table
    return tuple(tuple(table for _ in row) for row in default)


def _render(name, tables):
    routine = getattr(music, name)
    parameters = inspect.signature(routine).parameters
    kwargs = {parameter: _laid_out(parameters[parameter].default, table)
              for parameter, table in tables.items()}
    if "duration" in parameters:
        kwargs["duration"] = SHORT
    if "seed" in parameters:
        kwargs["seed"] = 1
    with music.set_sample_rate(RATE):
        return routine(**kwargs)


def test_the_routines_with_tables_are_the_ones_held_to_this():
    assert set(ROUTINES) == {
        "am", "amplitude_modulation", "binaural_beats",
        "frequency_modulation", "isochronic_tones", "modulated_noise",
        "monaural_beats", "note", "note_with_doppler", "note_with_fm",
        "note_with_glissando", "note_with_glissando_vibrato",
        "note_with_phase", "note_with_two_vibratos",
        "note_with_two_vibratos_glissando", "note_with_vibrato",
        "note_with_vibrato_seq_localization",
        "note_with_vibratos_glissandos", "spatial_motion", "tremolo",
        "tremolos"}


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
@pytest.mark.parametrize("name", sorted(ROUTINES))
def test_an_oscillator_returns_float64_for_a_table_of_any_type(name, dtype):
    table = music.waveform_table("sine", 256)
    given = {parameter: _as(table, dtype) for parameter in ROUTINES[name]}
    widened = {parameter: _as(table, dtype).astype(np.float64)
               for parameter in ROUTINES[name]}
    rendered = _render(name, given)
    assert rendered.dtype == np.float64
    np.testing.assert_array_equal(rendered, _render(name, widened))
