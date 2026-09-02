"""Protocols: sequences of stimuli, joined by ramps rather than by cuts.

A stimulation protocol is rarely one stimulus. It is a few of them in
order -- settle at 10 Hz, descend to 6, rest -- and the interesting part
is the joins, because a cut between two stimuli is a step discontinuity
and a step is a click. :class:`StimulationSession` holds that sequence
and renders it as one sound.

SSTIM calls the reproducible description of an intended run a
``sstim:SessionSpecification``: "the unit of scientific reproducibility:
given a specification and a conforming engine, the acoustic output is
fully determined." That is the object this class is the rendering half
of. The phases correspond to what SSTIM models as the audible layers of
a configuration, ``sstim:AudioTrack`` -- "an isochronic tone, a binaural
carrier pair, a plain carrier, noise" -- taken in sequence rather than
in parallel.

Teaching this class to *read and write* those descriptions is separate
work, tracked as issue #75; the vocabulary is adopted here so that the
two do not have to be reconciled later.

References
----------
.. [1] SSTIM, ``SessionSpecification``.
       https://w3id.org/sstim/session
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.io import write_wav_mono, write_wav_stereo
from ..utils import convert_to_stereo

__all__ = ['StimulationSession', 'StimulusPhase']

Stimulus = Union[Callable[..., ArrayLike], ArrayLike]


@dataclass
class StimulusPhase:
    """One phase of a session: a stimulus, for a while, ramped into.

    Attributes
    ----------
    stimulus : callable or array_like
        What to render. A callable is called with ``number_of_samples``
        and ``sample_rate`` alongside ``parameters``, which is what
        every generator in :mod:`music.stimulation` accepts, so the
        session can render it at exactly the length the arrangement
        needs. An array is taken as it stands.
    duration : scalar
        How long the phase lasts, in seconds, not counting its share of
        the ramps at either end. Ignored when ``stimulus`` is an array,
        which brings its own length.
    ramp : scalar
        The length in seconds of the transition *into* this phase: a
        crossfade with the phase before it, or a fade in from silence
        for the first phase.
    gain : scalar
        A factor applied to the phase before it is mixed, for balancing
        stimuli of different loudness against each other.
    label : string
        A name for the phase, carried for the reader's sake and shown
        in the session's ``repr``.
    parameters : dict
        Keyword arguments passed to ``stimulus`` when it is a callable.
    """

    stimulus: Stimulus
    duration: float = 0.0
    ramp: float = 0.0
    gain: float = 1.0
    label: str = ''
    parameters: Dict[str, Any] = field(default_factory=dict)


def _ramp_shape(count, shape, rising):
    """A gain curve `count` samples long, rising or falling.

    Two stimuli of a protocol are different sounds rather than the same
    sound twice, so their crossfade is between uncorrelated signals,
    where amplitudes add in quadrature. A linear pair of ramps then dips
    about 3 dB in the middle -- audible as a hole at every transition --
    and the sine/cosine pair does not, which is why it is the default.
    ``'linear'`` is offered for the correlated case, where it is the
    pair that holds amplitude constant instead.
    """
    if count <= 0:
        return np.ones(0)
    progress = np.arange(count) / count
    if not rising:
        progress = 1 - progress
    if shape == 'linear':
        return progress
    return np.sin(progress * np.pi / 2)


@dataclass
class StimulationSession:
    """A sequence of stimuli rendered as one sound.

    Phases are added in order and rendered end to end. Each transition
    is a crossfade centred on the boundary between the phases it joins,
    so **the session lasts exactly the sum of its phase durations**: a
    ramp is taken half from the phase before it and half from the phase
    after, rather than being inserted between them and stretching the
    protocol past the length its author wrote down. A protocol that says
    ten minutes lasts ten minutes.

    Attributes
    ----------
    sample_rate : integer
        The sampling frequency in Hertz, passed to every callable
        stimulus so the whole session is rendered at one rate.
    end_ramp : scalar
        The fade to silence at the end of the session, in seconds. Like
        the fade in at the start -- the first phase's ``ramp`` -- it is
        taken from within the session rather than added to it, because
        there is no neighbouring phase for it to overlap.
    ramp_shape : string
        ``'equal_power'``, the default, or ``'linear'``. See
        :func:`_ramp_shape` for which to want when.
    phases : list of StimulusPhase
        The phases, in order. Normally built with :meth:`add`.

    Examples
    --------
    >>> import music
    >>> session = music.StimulationSession(end_ramp=0.05)
    >>> session.add(music.binaural_beats, duration=0.2, beat_freq=10)
    >>> session.add(music.isochronic_tones, duration=0.2, ramp=0.05,
    ...             pulse_rate=6)
    >>> session.duration
    0.4
    >>> session.render().shape
    (2, 17640)

    See Also
    --------
    music.Sequencer : notes scheduled at arbitrary offsets, which is the
                      musical version of the same idea.
    """

    sample_rate: int = 44100
    end_ramp: float = 0.0
    ramp_shape: str = 'equal_power'
    phases: List[StimulusPhase] = field(default_factory=list)

    def add(self, stimulus: Stimulus, duration: float = 0.0,
            ramp: float = 0.0, gain: float = 1.0, label: str = '',
            **parameters: Any) -> None:
        """Append a phase.

        Parameters
        ----------
        stimulus : callable or array_like
            The generator to render, or an already rendered sound.
        duration : scalar
            How long the phase lasts, in seconds. Required for a
            callable; ignored for an array, which brings its own length.
        ramp : scalar
            The transition into this phase, in seconds.
        gain : scalar
            A factor applied to the phase before mixing.
        label : string
            A name for the phase.
        **parameters
            Passed to ``stimulus`` when it is a callable, so a phase is
            written as the generator plus the arguments it would have
            been called with.

        Raises
        ------
        ValueError
            If ``ramp`` or ``duration`` is negative, or if a callable
            stimulus is given no duration -- it would render nothing,
            and silently dropping a phase from a protocol is worse than
            refusing it.
        """
        if duration < 0:
            raise ValueError(f"duration cannot be negative, got {duration}")
        if ramp < 0:
            raise ValueError(f"ramp cannot be negative, got {ramp}")
        if callable(stimulus) and not duration:
            raise ValueError(
                "a callable stimulus needs a duration; pass an already "
                "rendered array to use its own length instead")
        self.phases.append(
            StimulusPhase(stimulus=stimulus, duration=duration, ramp=ramp,
                          gain=gain, label=label, parameters=parameters))

    def _layout(self):
        """Where each phase starts and how long it is rendered, in samples.

        Returns one ``(start, length, rise, fall)`` tuple per phase.
        ``rise`` and ``fall`` are the crossfade lengths at its two
        edges; consecutive phases overlap by exactly the shared ramp, so
        one phase's fall lines up sample for sample with the next one's
        rise.
        """
        count = len(self.phases)
        ramps = [int(round(phase.ramp * self.sample_rate))
                 for phase in self.phases]
        ramps.append(int(round(self.end_ramp * self.sample_rate)))

        layout = []
        # Each boundary is rounded from the elapsed time rather than
        # summed from per-phase roundings, which would drift by a sample
        # every few phases and leave a long protocol the wrong length --
        # the same error, in the same package, that phase integration
        # made until 1.3.0. Phases given as arrays contribute exact
        # sample counts, so they are carried separately.
        elapsed = 0.0
        from_arrays = 0
        for index, phase in enumerate(self.phases):
            # The opening and closing ramps overlap nothing, so they are
            # carved out of the session rather than added to it, and the
            # phases at the two ends are not extended.
            half_in = 0 if index == 0 else ramps[index] // 2
            half_out = (0 if index == count - 1
                        else ramps[index + 1] - ramps[index + 1] // 2)
            boundary = int(round(elapsed * self.sample_rate)) + from_arrays
            if callable(phase.stimulus):
                elapsed += phase.duration
                span = (int(round(elapsed * self.sample_rate))
                        + from_arrays - boundary)
            else:
                span = (len(np.atleast_2d(phase.stimulus)[0])
                        - half_in - half_out)
                from_arrays += span
            layout.append((boundary - half_in, span + half_in + half_out,
                           ramps[index], ramps[index + 1]))
        return layout

    def _render_phase(self, phase, length):
        """One phase as an array of exactly `length` samples.

        A callable is asked for that length directly. An array is
        already the right length by construction -- :meth:`_layout`
        derived the phase's span from it -- so it is only converted.
        """
        if callable(phase.stimulus):
            rendered = phase.stimulus(number_of_samples=length,
                                      sample_rate=self.sample_rate,
                                      **phase.parameters)
        else:
            rendered = phase.stimulus
        return np.asarray(rendered, dtype=np.float64) * phase.gain

    @property
    def duration(self) -> float:
        """How long the session lasts, in seconds.

        Measured from the layout rather than from the durations that
        were asked for, so it reports what will actually be rendered.
        """
        layout = self._layout()
        if not layout:
            return 0.0
        start, length, _, _ = layout[-1]
        return (start + length) / self.sample_rate

    def render(self) -> NDArray[np.float64]:
        """Render every phase and mix them into one sound.

        Returns
        -------
        ndarray
            A one-dimensional array, or a ``(2, nsamples)`` one if any
            phase is stereo -- a session that mixes binaural beats with
            isochronic tones is stereo throughout, because the two
            cannot share a channel layout and the binaural phase is the
            one that would be destroyed by flattening.

        Raises
        ------
        ValueError
            If a callable stimulus returns a length other than the one
            it was asked for. Every generator in this package honours
            ``number_of_samples``; one that does not would silently
            shift every phase after it.
        """
        layout = self._layout()
        if not layout:
            return np.zeros(0)

        rendered = [self._render_phase(phase, length)
                    for phase, (_, length, _, _) in zip(self.phases, layout)]
        for sound, (_, length, _, _) in zip(rendered, layout):
            if np.atleast_2d(sound).shape[1] != length:
                raise ValueError(
                    f"a stimulus rendered "
                    f"{np.atleast_2d(sound).shape[1]} samples where "
                    f"{length} were asked for")

        stereo = any(sound.ndim == 2 for sound in rendered)
        start, length, _, _ = layout[-1]
        total = start + length
        mix = np.zeros((2, total)) if stereo else np.zeros(total)

        for sound, (start, length, rise, fall) in zip(rendered, layout):
            envelope = np.ones(length)
            rise = min(rise, length)
            fall = min(fall, length - rise)
            envelope[:rise] = _ramp_shape(rise, self.ramp_shape, True)
            if fall:
                envelope[length - fall:] = _ramp_shape(
                    fall, self.ramp_shape, False)
            if stereo and sound.ndim == 1:
                sound = convert_to_stereo(sound)
            mix[..., start:start + length] += sound * envelope
        return mix

    def write(self, filename: str, bit_depth: int = 16) -> None:
        """Render the session and write it to a WAV file.

        Parameters
        ----------
        filename : string
            The path to write to.
        bit_depth : integer
            The bit depth of the samples written.
        """
        data = self.render()
        if data.ndim == 1:
            write_wav_mono(data, filename=filename,
                           sample_rate=self.sample_rate,
                           bit_depth=bit_depth)
        else:
            write_wav_stereo(data, filename=filename,
                             sample_rate=self.sample_rate,
                             bit_depth=bit_depth)

    def __repr__(self) -> str:
        if not self.phases:
            return 'StimulationSession(empty)'
        names = ', '.join(
            phase.label or getattr(phase.stimulus, '__name__', 'array')
            for phase in self.phases)
        return (f'StimulationSession({len(self.phases)} phases, '
                f'{self.duration:g} s: {names})')
