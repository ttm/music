"""Behavioral regressions from the non-envelope filter audit."""

from importlib import import_module

import numpy as np
import pytest

import music


def test_louds_pads_a_short_signal_with_silence():
    """Samples the envelope extends past the sound must stay silent."""
    signal = np.array([1.0, -1.0])
    result = music.louds(
        number_of_samples=(2, 2),
        trans_devs=(6, -6),
        alpha=(1, 1),
        method=("exp", "exp"),
        sonic_vector=signal,
    )

    assert result.shape == (4,)
    np.testing.assert_array_equal(result[2:], [0.0, 0.0])


def test_loud_rejects_an_unknown_method_with_a_value_error():
    with pytest.raises(ValueError, match="method"):
        music.loud(number_of_samples=8, method="bogus")


@pytest.mark.parametrize(("method", "canonical"), [
    ("lin", "linear"),
    ("exponential", "exp"),
])
def test_loud_method_aliases_match_their_canonical_names(method, canonical):
    aliases = music.loud(number_of_samples=8, method=method, trans_dev=6)
    expected = music.loud(number_of_samples=8, method=canonical,
                          trans_dev=6)
    np.testing.assert_array_equal(aliases, expected)


def test_loud_decreasing_exponential_is_checked_sample_by_sample():
    count = 5
    produced = music.loud(number_of_samples=count, trans_dev=6, alpha=2,
                          to=False, method="exp")
    samples = np.arange(count)
    expected = 10 ** (((count - 1 - samples) / (count - 1)) ** 2 * 6 / 20)
    np.testing.assert_allclose(produced, expected, rtol=0, atol=1e-12)


def test_loud_linear_transition_can_start_at_the_deviation():
    np.testing.assert_array_equal(
        music.loud(number_of_samples=5, trans_dev=2, to=False,
                   method="linear"),
        [2.0, 1.75, 1.5, 1.25, 1.0],
    )


def test_loud_applies_its_envelope_to_the_signal():
    signal = np.array([-2.0, 1.0, 3.0, -4.0])
    produced = music.loud(number_of_samples=len(signal), trans_dev=6,
                          sonic_vector=signal)
    envelope = 10 ** (np.arange(len(signal)) / (len(signal) - 1) * 6 / 20)
    np.testing.assert_allclose(produced, signal * envelope)


def test_louds_accepts_a_one_sample_transition():
    np.testing.assert_array_equal(
        music.louds(number_of_samples=(1,), trans_devs=(6,), alpha=(1,),
                    method=("exp",)),
        [1.0],
    )


def test_louds_honors_each_transition_method():
    produced = music.louds(number_of_samples=(3, 3), trans_devs=(6, -6),
                           alpha=(1, 1), method=("linear", "exp"))
    first = np.array([1.0, 3.5, 6.0])
    second = 10 ** (np.arange(3) / 2 * -6 / 20) * first[-1]
    np.testing.assert_allclose(produced, np.concatenate((first, second)))


def test_louds_holds_the_final_gain_when_the_signal_is_longer():
    signal = np.arange(1.0, 8.0)
    produced = music.louds(number_of_samples=(3, 2), trans_devs=(6, -12),
                           alpha=(1, 1), method=("exp", "exp"),
                           sonic_vector=signal)
    first = 10 ** (np.arange(3) / 2 * 6 / 20)
    second = first[-1] * 10 ** (np.arange(2) * -12 / 20)
    envelope = np.concatenate((first, second, np.repeat(second[-1], 2)))
    np.testing.assert_allclose(produced, signal * envelope)


def test_filter_defaults_keep_their_documented_sample_spans():
    assert len(music.loud()) == 2 * 44100
    assert music.loud()[-1] == pytest.approx(10 ** (10 / 20))
    assert len(music.louds()) == 8 * 44100
    assert len(music.reverb()) == int(1.9 * 44100)
    assert len(music.stretches(np.array([0.0, 1.0]))) == 25 * 44100


@pytest.mark.parametrize("sample_rate", [0, -1])
def test_fraction_of_requires_a_positive_sample_rate(sample_rate):
    with pytest.raises(ValueError, match="sample_rate"):
        music.fraction_of(1, sample_rate=sample_rate)


def test_fraction_of_keeps_sub_hertz_values_and_its_default_rate():
    assert music.fraction_of(0.5) == pytest.approx(0.5 / 44100)
    assert music.fraction_of(0.5, sample_rate=1) == pytest.approx(0.5)


def test_filter_design_defaults_match_their_coefficient_equations():
    cutoff, centre, bandwidth = 0.1, 0.1, 0.05
    x = np.exp(-2 * np.pi * cutoff)
    a, b = music.low_pass()
    np.testing.assert_allclose(a, [1 - x])
    np.testing.assert_allclose(b, [1.0, x])

    a, b = music.high_pass()
    half = (x + 1) / 2
    np.testing.assert_allclose(a, [half, -half])
    np.testing.assert_allclose(b, [1.0, x])

    r = 1 - 3 * bandwidth
    cosine = np.cos(2 * np.pi * centre)
    k = (1 - 2 * r * cosine + r ** 2) / (2 - 2 * cosine)
    a, b = music.band_pass()
    np.testing.assert_allclose(
        a, [1 - k, 2 * (k - r) * cosine, r ** 2 - k])
    np.testing.assert_allclose(b, [1.0, 2 * r * cosine, -r ** 2])
    a, b = music.band_reject()
    np.testing.assert_allclose(a, [k, -2 * k * cosine, k])
    np.testing.assert_allclose(b, [1.0, 2 * r * cosine, -r ** 2])


@pytest.mark.parametrize("max_freq", [True, False])
def test_fir_defaults_to_magnitudes_with_the_selected_endpoint(max_freq):
    magnitudes = np.array([1.0, 0.6, 0.2, 0.0])
    signal = np.array([0.0, 1.0, -2.0, 0.5, 3.0, -1.0])
    mirrored = magnitudes[1:-1] if max_freq else magnitudes[1:]
    spectrum = np.hstack((magnitudes, mirrored[::-1]))
    kernel = np.fft.fftshift(np.fft.ifft(spectrum).real)
    expected = np.convolve(kernel, signal)

    np.testing.assert_allclose(
        music.fir(magnitudes, signal, max_freq=max_freq), expected)


def test_fir_defaults_to_nyquist_as_the_last_magnitude():
    magnitudes = np.array([1.0, 0.6, 0.2, 0.0])
    signal = np.array([0.0, 1.0, -2.0, 0.5, 3.0, -1.0])
    spectrum = np.hstack((magnitudes, magnitudes[1:-1][::-1]))
    kernel = np.fft.fftshift(np.fft.ifft(spectrum).real)

    np.testing.assert_allclose(music.fir(magnitudes, signal),
                               np.convolve(kernel, signal))


def test_reverb_rejects_a_negative_first_phase_duration():
    with pytest.raises(ValueError,
                       match="^first_phase_duration must be non-negative"):
        music.reverb(duration=0.1, first_phase_duration=-0.01,
                     sample_rate=1000)


@pytest.mark.parametrize("sample_rate", [0, -1])
def test_reverb_requires_a_positive_sample_rate(sample_rate):
    with pytest.raises(ValueError, match="sample_rate"):
        music.reverb(duration=0.1, first_phase_duration=0,
                     sample_rate=sample_rate)


def test_reverb_accepts_a_one_hertz_sample_rate(monkeypatch):
    module = import_module("music.core.filters.reverb")
    monkeypatch.setattr(
        module, "noise",
        lambda _kind, **kwargs: np.ones(kwargs["number_of_samples"]))
    assert len(module.reverb(duration=2, first_phase_duration=0,
                             sample_rate=1)) == 2


def test_reverb_asks_for_noise_through_its_sample_rate_nyquist(monkeypatch):
    module = import_module("music.core.filters.reverb")
    calls = []

    def fake_noise(noise_type, **kwargs):
        calls.append((noise_type, kwargs))
        return np.ones(kwargs["number_of_samples"])

    monkeypatch.setattr(module, "noise", fake_noise)
    module.reverb(duration=0.01, first_phase_duration=0.002,
                  noise_type="white", sample_rate=8000)

    assert calls == [("white", {
        "max_freq": 4000,
        "number_of_samples": 64,
        "sample_rate": 8000,
    })]


def test_reverb_uses_the_quadratic_probability_for_first_period_hits(
        monkeypatch):
    module = import_module("music.core.filters.reverb")
    draws = np.array([0.0, 0.02, (2 / 5) ** 2, 0.2, 0.7])
    monkeypatch.setattr(module.np.random, "random", lambda _size: draws)
    monkeypatch.setattr(
        module, "noise",
        lambda _kind, **kwargs: np.ones(kwargs["number_of_samples"]))

    response = module.reverb(duration=1, first_phase_duration=0.5,
                             decay=0, sample_rate=10)
    np.testing.assert_array_equal(response[1:5], [1.0, 0.0, 1.0, 0.0])


def test_reverb_with_no_first_phase_leaves_the_noise_random_stream_alone():
    sample_rate = 8000
    duration = 0.01
    count = int(duration * sample_rate)
    np.random.seed(911)
    tail = music.noise("white", max_freq=sample_rate / 2,
                       number_of_samples=count, sample_rate=sample_rate)
    expected = tail * 10 ** (
        -6 / 20 * np.arange(count) / (count - 1))
    expected[0] = 1.0

    np.random.seed(911)
    response = music.reverb(duration=duration, first_phase_duration=0,
                            decay=-6, noise_type="white",
                            sample_rate=sample_rate)

    np.testing.assert_allclose(response, expected, rtol=1e-15, atol=0)


def test_two_sample_reverb_reaches_its_requested_final_decay(monkeypatch):
    module = import_module("music.core.filters.reverb")
    monkeypatch.setattr(
        module, "noise",
        lambda _kind, **kwargs: np.ones(kwargs["number_of_samples"]))
    response = module.reverb(duration=0.2, first_phase_duration=0,
                             decay=-20, sample_rate=10)
    np.testing.assert_allclose(response, [1.0, 0.1])


def test_reverb_defaults_to_a_point_one_five_second_first_phase():
    np.random.seed(19)
    default = music.reverb(duration=0.4, sample_rate=1000)
    np.random.seed(19)
    explicit = music.reverb(duration=0.4, first_phase_duration=0.15,
                            sample_rate=1000)
    np.testing.assert_array_equal(default, explicit)


def test_fir_returns_float_samples_for_integer_impulse_responses():
    response = music.fir(np.array([1, 2]), np.array([3, 4]), freq=False)
    assert response.dtype == np.float64


def test_reverb_applies_the_response_by_convolution():
    signal = np.array([1.0, -2.0, 0.5])
    settings = dict(duration=0.02, first_phase_duration=0.01,
                    sample_rate=1000, noise_type="white")
    np.random.seed(123)
    response = music.reverb(**settings)
    np.random.seed(123)
    produced = music.reverb(sonic_vector=signal, **settings)
    np.testing.assert_array_equal(produced, np.convolve(signal, response))


def test_stretches_accepts_duration_generators():
    fragment = np.array([0.0, 1.0, 2.0, 3.0])
    generated = music.stretches(fragment,
                                durations=(duration for duration in (1, 2)),
                                sample_rate=2)
    expected = music.stretches(fragment, durations=(1, 2), sample_rate=2)
    np.testing.assert_array_equal(generated, expected)


def test_stretches_with_no_durations_returns_no_samples():
    mono = music.stretches(np.arange(4.0), durations=())
    stereo = music.stretches(np.ones((2, 4)), durations=())
    assert mono.shape == (0,)
    assert stereo.shape == (2, 0)


def test_stretches_rejects_nonpositive_sample_rates():
    with pytest.raises(ValueError, match="sample_rate"):
        music.stretches(np.array([1.0]), sample_rate=0)


def test_stretches_accepts_a_one_hertz_sample_rate():
    assert music.stretches(np.array([1.0, 2.0]), durations=(1,),
                           sample_rate=1).shape == (1,)


@pytest.mark.parametrize("shape", [(), (2, 3, 4), (3, 4)])
def test_stretches_refuses_unsupported_audio_shapes(shape):
    with pytest.raises(ValueError,
                       match="^stretches accepts mono or stereo"):
        music.stretches(np.zeros(shape))
