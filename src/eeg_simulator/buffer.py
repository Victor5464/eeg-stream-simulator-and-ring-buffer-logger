import numpy as np
from threading import Lock

class RingBuffer:
    """Thread-safe pre-allocated circular buffer for real-time multi-channel signal data.
    
    Optimized to store values in single-precision float32 (4 bytes per sample), 
    providing efficient C-contiguous memory layout. Supports direct binary byte serialization.
    On overflow, oldest samples are silently overwritten.
    """
    
    def __init__(self, max_samples: int, num_channels: int) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive.")
        if num_channels <= 0:
            raise ValueError("num_channels must be positive.")
            
        self._max_samples = max_samples
        self._num_channels = num_channels
        
        # Using float32 instead of float64 halves cache overhead and matches BCI hardware standards
        self._data = np.zeros((max_samples, num_channels), dtype=np.float32)
        
        # Pre‑computed index ramp for vectorized modulo window retrieval
        self._index_ramp = np.arange(max_samples, dtype=np.int32)
        
        self._write_index: int = 0
        self._total_written: int = 0
        self._lock = Lock()
    
    @property
    def max_samples(self) -> int:
        return self._max_samples
    
    @property
    def num_channels(self) -> int:
        return self._num_channels
    
    @property
    def total_samples_written(self) -> int:
        with self._lock:
            return self._total_written
    
    def push(self, sample: np.ndarray) -> None:
        """Thread‑safe insertion of sample frames. Downcasts to float32 natively.
        
        Args:
            sample: 1D array of shape (num_channels,) containing one time step.
            
        Raises:
            ValueError: If sample shape doesn't match configured num_channels.
        """
        if sample.shape != (self._num_channels,):
            raise ValueError(
                f"Sample shape {sample.shape} must be ({self._num_channels},)"
            )
        
        with self._lock:
            # Native cast to single-precision float32 during insertion
            self._data[self._write_index] = sample.astype(np.float32, copy=False)
            self._write_index = (self._write_index + 1) % self._max_samples
            self._total_written += 1
    
    def get_latest_window(self, window_size: int) -> np.ndarray:
        """Thread‑safe, non‑destructive retrieval of the most recent samples as floats.
        
        Returns:
            2D array of shape (min(window_size, total_written), num_channels) in float32.
        """
        if window_size <= 0:
            raise ValueError("Window size must be positive.")
        if window_size > self._max_samples:
            raise ValueError(
                f"Window size {window_size} exceeds buffer capacity {self._max_samples}."
            )
        
        with self._lock:
            available = min(window_size, self._total_written)
            
            if available == 0:
                return np.empty((0, self._num_channels), dtype=np.float32)
            
            start_idx = (self._write_index - available) % self._max_samples
            indices = (self._index_ramp[:available] + start_idx) % self._max_samples
            return self._data[indices].copy()

    def get_latest_window_bytes(self, window_size: int) -> bytes:
        """Thread‑safe retrieval of the most recent samples as raw, contiguous bytes.
        
        Provides extremely fast disk write mechanics by avoiding float conversions.
        
        Returns:
            Raw bytes containing packed float32 channel variables.
        """
        if window_size <= 0:
            raise ValueError("Window size must be positive.")
        if window_size > self._max_samples:
            raise ValueError(
                f"Window size {window_size} exceeds buffer capacity {self._max_samples}."
            )

        with self._lock:
            available = min(window_size, self._total_written)
            
            if available == 0:
                return b""
            
            start_idx = (self._write_index - available) % self._max_samples
            indices = (self._index_ramp[:available] + start_idx) % self._max_samples
            
            # Extract consecutive, ordered memory layout slice and export directly to bytes
            return self._data[indices].tobytes()