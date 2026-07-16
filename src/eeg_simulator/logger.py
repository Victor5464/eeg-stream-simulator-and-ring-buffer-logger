import csv
import json
import struct
from pathlib import Path
import numpy as np
from eeg_simulator.config import SimulationConfig, LogFormat


class EEGLogger:
    """Thread-safe file writer for EEG data supporting CSV, JSON, and Binary formats."""
    
    def __init__(self, config: SimulationConfig) -> None:
        self._config = config
        self._config.output_directory.mkdir(parents=True, exist_ok=True)
        self._file_path = self._config.output_directory / f"eeg_data.{self._config.log_format.value}"
        
        if self._config.log_format == LogFormat.CSV:
            self._write_csv_header()
    
    @property
    def file_path(self) -> Path:
        return self._file_path
    
    def _write_csv_header(self) -> None:
        with open(self._file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ["Timestamp"] + [ch.name for ch in self._config.channels]
            writer.writerow(header)
    
    def write_window(self, timestamps: np.ndarray, data_matrix: np.ndarray) -> None:
        """Persist a window of EEG samples to disk.
        
        Args:
            timestamps: 1D array of shape (N,) with UNIX timestamps.
            data_matrix: 2D array of shape (N, num_channels) with sample values.
        """
        n_samples = len(timestamps)
        if n_samples == 0:
            return
        
        fmt = self._config.log_format
        
        if fmt == LogFormat.CSV:
            self._write_csv(timestamps, data_matrix)
        elif fmt == LogFormat.BINARY:
            self._write_binary(timestamps, data_matrix)
        elif fmt == LogFormat.JSON:
            self._write_json(timestamps, data_matrix)
    
    def _write_csv(self, timestamps: np.ndarray, data: np.ndarray) -> None:
        with open(self._file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Combine timestamps and data matrix for a single vectorized write iteration
            combined = np.column_stack((timestamps, data))
            writer.writerows(combined)
    
    def _write_binary(self, timestamps: np.ndarray, data: np.ndarray) -> None:
        """Writes binary data with zero loop-overhead.
        
        Structure per sample:
        - 1 Double (8 bytes) for Unix timestamp
        - N Float32s (4 bytes each) for the channel readings
        """
        num_channels = data.shape[1]
        
        # Build structured array layout to pack on the CPU vector engine
        # 'd' = float64 (double), 'f4' = float32 (single precision)
        dtype = np.dtype([
            ('timestamp', '<f8'), 
            ('readings', f'<f4', (num_channels,))
        ])
        
        # Initialize structured container
        packed_records = np.empty(len(timestamps), dtype=dtype)
        packed_records['timestamp'] = timestamps
        packed_records['readings'] = data.astype(np.float32, copy=False)
        
        # Write entire memory block to disk in a single system write call
        with open(self._file_path, mode='ab') as f:
            f.write(packed_records.tobytes())
    
    def _write_json(self, timestamps: np.ndarray, data: np.ndarray) -> None:
        channel_names = [ch.name for ch in self._config.channels]
        with open(self._file_path, mode='a', encoding='utf-8') as f:
            lines = []
            for i in range(len(timestamps)):
                record = {
                    "timestamp": float(timestamps[i]),
                    "readings": {
                        channel_names[c]: float(data[i, c])
                        for c in range(len(channel_names))
                    }
                }
                lines.append(json.dumps(record) + "\n")
            f.writelines(lines)