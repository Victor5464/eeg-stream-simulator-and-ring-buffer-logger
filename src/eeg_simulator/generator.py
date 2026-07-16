import numpy as np
from typing import List, Optional
from eeg_simulator.config import ChannelConfig

class SignalGenerator:
    """Stateful multichannel signal generator producing values frame-by-frame."""
    
    def __init__(
        self, 
        channels: List[ChannelConfig], 
        noise_amplitude: float = 0.05, 
        sampling_rate_hz: int = 250,
        seed: Optional[int] = None
    ) -> None:
        self.channels = channels
        self.noise_amplitude = noise_amplitude
        self.sampling_rate_hz = sampling_rate_hz
        
        self._rng = np.random.default_rng(seed)
        self.current_time = 0
        
        # Vectorized internal matrices
        self._freqs = np.array([ch.frequency_hz for ch in channels], dtype=np.float64)
        self._amps = np.array([ch.amplitude_uv for ch in channels], dtype=np.float64)
        self._phases = np.array([ch.phase for ch in channels], dtype=np.float64)
        self._num_channels = len(channels)

    def reset(self) -> None:
        """Resets the internal time step counter back to zero for testing consistency."""
        self.current_time = 0

    def generate_sample(self) -> np.ndarray:
        """Calculates wave mechanics for all configured channels simultaneously."""
        time_step = self.current_time / self.sampling_rate_hz
        signals = self._amps * np.sin(2 * np.pi * self._freqs * time_step + self._phases)
        
        noise = self._rng.normal(loc=0.0, scale=self.noise_amplitude, size=self._num_channels)
        self.current_time += 1
        
        return signals + noise