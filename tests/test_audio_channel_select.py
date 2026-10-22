"""Tests for selecting one input channel from a multichannel audio stream."""

import numpy as np
import pytest
from pydantic import ValidationError

from ledfx.configuration.models import AudioConfig
from ledfx.effects.audio import AudioInputSource

# Two frames of 4.1 audio (FL, FR, LFE, RL, RR), interleaved
INTERLEAVED_4_1 = np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float32
)


def make_ais(audio_channel, stream_channels):
    ais = object.__new__(AudioInputSource)
    ais._config = AudioConfig.model_validate({"audio_channel": audio_channel})
    ais._stream_channels = stream_channels
    return ais


@pytest.mark.parametrize(
    ("channel", "expected"),
    [(0, [0.1, 1.1]), (2, [0.3, 1.3]), (4, [0.5, 1.5])],
)
def test_selects_configured_channel(channel, expected):
    out = make_ais(channel, 5)._select_channel(INTERLEAVED_4_1)

    np.testing.assert_allclose(out, expected)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("channel", [-1, 5])
def test_mixes_when_channel_is_mix_or_absent(channel):
    out = make_ais(channel, 5)._select_channel(INTERLEAVED_4_1)

    np.testing.assert_allclose(out, [0.3, 1.3], rtol=1e-6)
    assert out.dtype == np.float32


def test_mono_stream_is_passed_through():
    sample = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    assert make_ais(0, 1)._select_channel(sample) is sample


def test_default_is_mix():
    assert AudioConfig.model_validate({}).audio_channel == -1


@pytest.mark.parametrize("value", [-2, 32])
def test_rejects_out_of_range_channel(value):
    with pytest.raises(ValidationError):
        AudioConfig.model_validate({"audio_channel": value})
