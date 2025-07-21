"""Simple note sequencer built on Music primitives."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np

from .core import synths
from .core.filters import adsr
from .core.filters.localization import localize
from .utils import convert_to_stereo
from .core.io import write_wav_mono, write_wav_stereo


@dataclass
class NoteEvent:
    """Represents a scheduled note."""

    freq: float
    start: float
    duration: float
    vibrato_freq: float = 0.0
    max_pitch_dev: float = 0.0
    adsr_params: Optional[Dict[str, Any]] = None
    spatial: Optional[Dict[str, Any]] = None


@dataclass
class Sequencer:
    """Schedules notes and renders them as audio."""

    sample_rate: int = 44100
    events: List[NoteEvent] = field(default_factory=list)

    def _mix_with_offset(self, base: np.ndarray, new: np.ndarray, start: float) -> np.ndarray:
        """Mix two sonic vectors with an offset in seconds."""
        offset = int(round(start * self.sample_rate))
        if base.ndim != new.ndim:
            if base.ndim == 1:
                base = convert_to_stereo(base)
            else:
                new = convert_to_stereo(new)

        if base.ndim == 1:
            final_len = max(len(base), offset + len(new))
            out = np.zeros(final_len)
            out[: len(base)] += base
            out[offset : offset + len(new)] += new
        else:
            final_len = max(base.shape[1], offset + new.shape[1])
            out = np.zeros((2, final_len))
            out[:, : base.shape[1]] += base
            out[:, offset : offset + new.shape[1]] += new
        return out

    def add_note(
        self,
        freq: float,
        start: float,
        duration: float,
        vibrato_freq: float = 0.0,
        max_pitch_dev: float = 0.0,
        adsr_params: Optional[Dict[str, Any]] = None,
        spatial: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a note event to the sequencer."""
        self.events.append(
            NoteEvent(
                freq=freq,
                start=start,
                duration=duration,
                vibrato_freq=vibrato_freq,
                max_pitch_dev=max_pitch_dev,
                adsr_params=adsr_params,
                spatial=spatial,
            )
        )

    # internal synthesize
    def _render_event(self, event: NoteEvent) -> np.ndarray:
        if event.vibrato_freq and event.max_pitch_dev:
            note = synths.note_with_vibrato(
                freq=event.freq,
                duration=event.duration,
                vibrato_freq=event.vibrato_freq,
                max_pitch_dev=event.max_pitch_dev,
                sample_rate=self.sample_rate,
            )
        else:
            note = synths.note(
                freq=event.freq,
                duration=event.duration,
                sample_rate=self.sample_rate,
            )
        if event.adsr_params:
            note = adsr(
                sonic_vector=note,
                sample_rate=self.sample_rate,
                **event.adsr_params,
            )
        if event.spatial:
            note = localize(
                sonic_vector=note, sample_rate=self.sample_rate, **event.spatial
            )
        return note

    def render(self) -> np.ndarray:
        """Render all scheduled events and return the audio array."""
        result: np.ndarray | None = None
        for event in sorted(self.events, key=lambda e: e.start):
            sound = self._render_event(event)
            if result is None:
                base = np.zeros((2, 0)) if sound.ndim == 2 else np.zeros(0)
                result = self._mix_with_offset(base, sound, event.start)
            else:
                result = self._mix_with_offset(result, sound, event.start)
        return np.array([]) if result is None else result

    def write(self, filename: str, bit_depth: int = 16) -> None:
        """Write the rendered audio to a WAV file."""
        data = self.render()
        if data.ndim == 1:
            write_wav_mono(data, filename=filename, sample_rate=self.sample_rate, bit_depth=bit_depth)
        else:
            write_wav_stereo(
                data, filename=filename, sample_rate=self.sample_rate, bit_depth=bit_depth
            )


__all__ = ["Sequencer", "NoteEvent"]

