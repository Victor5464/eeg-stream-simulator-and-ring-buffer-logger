import time
import csv
import pytest
from pathlib import Path
from eeg_simulator.config import SimulationConfig, ChannelConfig, LogFormat
from eeg_simulator.simulator import EEGSimulator


def test_simulator_integration(tmp_path: Path):
    # Setup standard dual-channel array configuration using amplitude_uv
    channels = [
        ChannelConfig(name="Fp1", frequency_hz=10.0, amplitude_uv=1.0),
        ChannelConfig(name="O2", frequency_hz=12.0, amplitude_uv=1.5),
    ]
    config = SimulationConfig(
        channels=channels,
        sample_rate_hz=100,
        buffer_size_seconds=5,
        log_format=LogFormat.CSV,
        output_directory=tmp_path
    )
    
    simulator = EEGSimulator(config)
    assert not simulator.is_running
    
    simulator.start()
    assert simulator.is_running
    
    # Run long enough for one full writing sweep to process
    time.sleep(1.2)
    
    simulator.stop()
    assert not simulator.is_running
    
    # Assert logs populated correctly
    log_file = simulator.logger.file_path
    assert log_file.exists()
    
    with open(log_file, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
        
    assert header == ["Timestamp", "Fp1", "O2"]
    assert len(rows) > 0
    assert float(rows[0][0]) > 0.0  # Unix timestamp evaluated successfully