"""Simple note sequencer built on Music primitives."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Dict, Any

import numpy as np

from .core import synths
from .core.filters.adsr import adsr
from .core.filters.localization import localize
from .core.io import write_audio
from ._rate import get_sample_rate


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
        and for turning each event's start time into a sample offset. It
        defaults to :func:`music.get_sample_rate`'s rate when the
        sequencer is made.
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

    sample_rate: int = field(default_factory=get_sample_rate)
    events: List[NoteEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be positive; got {self.sample_rate}")

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
            If ``start`` or ``duration`` is negative or not finite, or
            ``adsr_params`` or ``spatial`` names ``sonic_vector`` or
            ``sample_rate``, which the sequencer supplies itself. A
            negative start was placed at a negative sample offset and
            failed with a broadcasting error, NaN and infinity failed in
            ``round()`` when rendering, and a second ``sample_rate`` as a
            ``TypeError`` there. A negative duration rendered no note,
            silently, and one that was not finite failed when rendering.

            If ``freq``, ``vibrato_freq`` or ``max_pitch_dev`` is NaN or
            infinite, which the note's oscillator refuses only when the
            sequence renders, without saying which note it was.
        """
        if not math.isfinite(start):
            raise ValueError(
                f"a note starts a finite number of seconds into the "
                f"sequence; got start={start}")
        if start < 0:
            raise ValueError(
                f"a note cannot start before the sequence does; got "
                f"start={start}")
        if not (math.isfinite(duration) and duration >= 0):
            raise ValueError(
                f"a note lasts a finite number of seconds, none of them "
                f"negative; got duration={duration}")
        # Refused where the note is written. The oscillator refuses them
        # too, but only when the sequence renders, and cannot say which
        # note it was rendering.
        for name, value in (("freq", freq), ("vibrato_freq", vibrato_freq),
                            ("max_pitch_dev", max_pitch_dev)):
            if not math.isfinite(value):
                raise ValueError(
                    f"a note's {name} is a finite number; got "
                    f"{name}={value}")
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
        """Render all scheduled events and return the audio array.

        Returns
        -------
        ndarray
            The notes summed, each from its start, as long as the last
            one to end: mono, or ``(2, nsamples)`` once any note is placed
            with ``spatial``, the mono ones then in both channels. No
            notes render no samples.

        Notes
        -----
        Each note starts at the sample nearest its start. The mix is
        made once, as long as the note that ends last, and each note is
        added where it starts, so a render costs the length of the piece
        and of its notes. It used to grow the mix by a copy for every
        note, which cost each one the length of everything before it:
        2,000 notes in a row, 200 s of audio, took half a minute to
        render, a tenth of a second of it synthesis.
        """
        events = sorted(self.events, key=lambda e: e.start)
        notes = [self._render_event(event) for event in events]
        offsets = [int(round(event.start * self.sample_rate))
                   for event in events]
        length = max((offset + note.shape[-1]
                      for offset, note in zip(offsets, notes)), default=0)
        # Stereo once any note is, a mono note then in both channels; the
        # notes are added in the order of their starts, as they were.
        stereo = any(note.ndim == 2 for note in notes)
        result = np.zeros((2, length) if stereo else length)
        for offset, note in zip(offsets, notes):
            result[..., offset:offset + note.shape[-1]] += note
        return result

    def write(self, filename: str, bit_depth: int = 16,
              normalize: bool = True) -> None:
        """Write the rendered audio to a file, as :func:`write_audio` does.

        Parameters
        ----------
        filename : string
            The path to write: ``.wav`` or ``.flac``.
        bit_depth : integer
            The bit depth of the samples written.
        normalize : boolean
            Normalize to full scale, as by default, or write the samples
            as they are, refusing a sound that reaches past [-1, 1], which
            keeps the level of a sequence written beside others.

        Raises
        ------
        ValueError
            If there are no notes to write: the empty render reached the
            normalization, which blamed a duration computed as zero. Or
            as :func:`write_audio` raises it.
        """
        if not self.events:
            raise ValueError(
                "there are no notes to write; add some with add_note")
        write_audio(self.render(), filename=filename,
                    sample_rate=self.sample_rate, bit_depth=bit_depth,
                    normalize=normalize)


__all__ = ["Sequencer", "NoteEvent"]

