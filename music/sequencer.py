"""Simple note sequencer built on Music primitives."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Dict, Any

import numpy as np

from .core import synths
from .core.filters.adsr import adsr
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
    """Schedules notes and renders them as audio.

    Attributes
    ----------
    sample_rate : integer
        The sampling frequency in Hertz, used for every note rendered
        and for turning each event's start time into a sample offset.
    events : list of NoteEvent
        The scheduled notes. Normally built with :meth:`add_note` rather
        than passed in, and rendered in order of their start times
        rather than the order they were added.

    Raises
    ------
    ValueError
        If ``sample_rate`` is not positive. At zero every note rendered
        as no samples, and the sequence as silence of no length.

    See Also
    --------
    music.StimulationSession : phases in sequence rather than notes at
                               offsets, for stimulation protocols.

    Examples
    --------
    >>> seq = Sequencer()
    >>> for i, freq in enumerate([440, 550, 660]):
    ...     seq.add_note(freq, start=i * 0.25, duration=1.0)
    >>> len(seq.events)
    3
    >>> seq.write("chord.wav")
    """

    sample_rate: int = 44100
    events: List[NoteEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be positive; got {self.sample_rate}")

    def _mix_with_offset(self, base: np.ndarray, new: np.ndarray,
                         start: float) -> np.ndarray:
        """Mix two sonic vectors with an offset in seconds."""
        offset = int(round(start * self.sample_rate))
        if base.ndim != new.ndim:
            if base.ndim == 1:
                base = convert_to_stereo(base)
            else:
                new = convert_to_stereo(new)

        out: np.ndarray  # 1-D or (2, n) depending on the branch below
        if base.ndim == 1:
            final_len = max(len(base), offset + len(new))
            out = np.zeros(final_len)
            out[: len(base)] = base
            out[offset : offset + len(new)] += new
        else:
            final_len = max(base.shape[1], offset + new.shape[1])
            out = np.zeros((2, final_len))
            out[:, : base.shape[1]] = base
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
        """Add a note event to the sequencer.

        Parameters
        ----------
        freq : scalar
            The note's frequency in Hertz.
        start : scalar
            When it starts, in seconds from the start of the sequence.
        duration : scalar
            How long it lasts, in seconds.
        vibrato_freq : scalar
            The rate of a vibrato in Hertz. With ``max_pitch_dev``, both
            nonzero, the note is rendered by :func:`note_with_vibrato`.
        max_pitch_dev : scalar
            The depth of that vibrato.
        adsr_params : dict, optional
            Arguments for :func:`adsr`, applied to the note.
        spatial : dict, optional
            Arguments for :func:`localize`, which makes the note stereo.

        Raises
        ------
        ValueError
            If ``start`` is negative or not finite, or ``adsr_params`` or
            ``spatial`` names ``sonic_vector`` or ``sample_rate``, which
            the sequencer supplies itself. A negative start was placed at
            a negative sample offset and failed with a broadcasting
            error, NaN and infinity failed in ``round()`` when rendering,
            and a second ``sample_rate`` as a ``TypeError`` there.
        """
        if not math.isfinite(start):
            raise ValueError(
                f"a note starts a finite number of seconds into the "
                f"sequence; got start={start}")
        if start < 0:
            raise ValueError(
                f"a note cannot start before the sequence does; got "
                f"start={start}")
        for name, params in (("adsr_params", adsr_params),
                             ("spatial", spatial)):
            supplied = sorted({"sonic_vector", "sample_rate"}
                              & set(params or {}))
            if supplied:
                raise ValueError(
                    f"{name} cannot set {', '.join(supplied)}: the "
                    "sequencer passes the note and its own sample rate")
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
                sonic_vector=note, sample_rate=self.sample_rate,
                **event.spatial,
            )
        return note

    def render(self) -> np.ndarray:
        """Render all scheduled events and return the audio array."""
        # From no samples, which the first stereo note makes stereo.
        result = np.zeros(0)
        for event in sorted(self.events, key=lambda e: e.start):
            result = self._mix_with_offset(result, self._render_event(event),
                                           event.start)
        return result

    def write(self, filename: str, bit_depth: int = 16) -> None:
        """Write the rendered audio to a WAV file.

        Raises
        ------
        ValueError
            If there are no notes to write. The empty render reached the
            normalization, which blamed a duration computed as zero.
        """
        if not self.events:
            raise ValueError(
                "there are no notes to write; add some with add_note")
        data = self.render()
        if data.ndim == 1:
            write_wav_mono(
                data, filename=filename, sample_rate=self.sample_rate,
                bit_depth=bit_depth,
            )
        else:
            write_wav_stereo(
                data, filename=filename, sample_rate=self.sample_rate,
                bit_depth=bit_depth,
            )


__all__ = ["Sequencer", "NoteEvent"]

