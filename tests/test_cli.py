import pytest
import argparse
from pathlib import Path
from eeg_simulator.cli import parse_channel_string, main
from eeg_simulator.config import LogFormat


def test_parse_channel_string_success():
    """Verify correct format strings map accurately to ChannelConfig objects."""
    channel = parse_channel_string("Fp1,10.5,55.3")
    assert channel.name == "Fp1"
    assert channel.frequency_hz == 10.5
    assert channel.amplitude_uv == 55.3


def test_parse_channel_string_failures():
    """Ensure malformed configuration strings throw expected parser error types."""
    # Test missing amplitude component
    with pytest.raises(argparse.ArgumentTypeError):
        parse_channel_string("Fp1,10.5")
    
    # Test string parsing conversion error
    with pytest.raises(argparse.ArgumentTypeError):
        parse_channel_string("Fp1,not_a_float,50")