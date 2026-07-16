import struct
import csv
import json
import numpy as np
import pytest
from pathlib import Path
from eeg_simulator.config import SimulationConfig, ChannelConfig, LogFormat
from eeg_simulator.logger import EEGLogger


def _make_channels():
    return [
        ChannelConfig(name="Fp1", frequency_hz=10.0, amplitude_uv=1.0),
        ChannelConfig(name="O2", frequency_hz=12.0, amplitude_uv=1.5),
    ]


def test_logger_csv_header(tmp_path):
    config = SimulationConfig(
        channels=_make_channels(),
        output_directory=tmp_path,
        log_format=LogFormat.CSV,
    )
    logger = EEGLogger(config)
    
    with open(logger.file_path, mode='r', encoding='utf-8') as f:
        header = next(csv.reader(f))
    
    assert header == ["Timestamp", "Fp1", "O2"]


def test_logger_csv_write(tmp_path):
    config = SimulationConfig(
        channels=_make_channels(),
        output_directory=tmp_path,
        log_format=LogFormat.CSV,
    )
    logger = EEGLogger(config)
    
    timestamps = np.array([0.0, 0.004])
    data = np.array([[1.0, 2.0], [1.1, 2.1]])
    logger.write_window(timestamps, data)
    
    with open(logger.file_path, mode='r', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    
    assert rows[0] == ["Timestamp", "Fp1", "O2"]
    assert rows[1] == ["0.0", "1.0", "2.0"]
    assert rows[2] == ["0.004", "1.1", "2.1"]


def test_logger_binary_write(tmp_path):
    config = SimulationConfig(
        channels=_make_channels(),
        output_directory=tmp_path,
        log_format=LogFormat.BINARY,
    )
    logger = EEGLogger(config)
    
    timestamps = np.array([0.1])
    data = np.array([[3.14, 2.71]])
    logger.write_window(timestamps, data)
    
    with open(logger.file_path, mode='rb') as f:
        raw = f.read()
    
    # Payload Layout: 1 double (8B) + 2 float32s (2 * 4B = 8B) = 16 bytes
    assert len(raw) == 16
    
    # Unpack format: '<dff' (one little-endian double, two little-endian floats)
    unpacked = struct.unpack("<dff", raw)
    assert pytest.approx(unpacked[0]) == 0.1
    assert pytest.approx(unpacked[1], rel=1e-5) == 3.14
    assert pytest.approx(unpacked[2], rel=1e-5) == 2.71


def test_logger_json_write(tmp_path):
    config = SimulationConfig(
        channels=_make_channels(),
        output_directory=tmp_path,
        log_format=LogFormat.JSON,
    )
    logger = EEGLogger(config)
    
    timestamps = np.array([0.0])
    data = np.array([[5.0, 6.0]])
    logger.write_window(timestamps, data)
    
    with open(logger.file_path, mode='r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
    
    assert len(records) == 1
    assert records[0]["timestamp"] == 0.0
    assert records[0]["readings"] == {"Fp1": 5.0, "O2": 6.0}


def test_logger_empty_window_no_write(tmp_path):
    config = SimulationConfig(
        channels=_make_channels(),
        output_directory=tmp_path,
        log_format=LogFormat.CSV,
    )
    logger = EEGLogger(config)
    logger.write_window(np.array([]), np.empty((0, 2)))
    
    # File should only have the CSV header
    with open(logger.file_path, mode='r', encoding='utf-8') as f:
        content = f.read().strip()
    assert content == "Timestamp,Fp1,O2"


def test_logger_creates_output_directory(tmp_path):
    nested = tmp_path / "sub" / "deep"
    config = SimulationConfig(
        channels=_make_channels(),
        output_directory=nested,
        log_format=LogFormat.CSV,
    )
    EEGLogger(config)
    assert nested.exists()