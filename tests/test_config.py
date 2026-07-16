import pytest
from pathlib import Path
from eeg_simulator.config import SimulationConfig, ChannelConfig, LogFormat

def test_config_validation():
    ch = ChannelConfig(name="O1", frequency_hz=10.0, amplitude_uv=1.2)
    config = SimulationConfig(channels=[ch])
    assert config.sample_rate_hz == 250
    assert len(config.channels) == 1

def test_invalid_sample_rate():
    ch = ChannelConfig(name="O1", frequency_hz=10.0, amplitude_uv=1.2)
    with pytest.raises(ValueError):
        SimulationConfig(channels=[ch], sample_rate_hz=0)