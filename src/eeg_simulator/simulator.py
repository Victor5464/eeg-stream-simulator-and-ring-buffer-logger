import time
import threading
import logging
from typing import Optional
import numpy as np

from eeg_simulator.config import SimulationConfig
from eeg_simulator.generator import SignalGenerator
from eeg_simulator.buffer import RingBuffer
from eeg_simulator.logger import EEGLogger

log = logging.getLogger(__name__)


class EEGSimulator:
    """Orchestrates multi-threaded real-time EEG simulation.
    
    Acquires physical signal samples into a pure, optimized float32 RingBuffer, 
    and periodically reconstructs sample timing windows on disk flush events.
    """
    
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.generator = SignalGenerator(
            channels=config.channels,
            sampling_rate_hz=config.sample_rate_hz,
            noise_amplitude=0.05
        )
        self.logger = EEGLogger(config)
        
        self.buffer_capacity = config.sample_rate_hz * config.buffer_size_seconds
        self.num_channels = len(config.channels)
        
        # Pure channel-only configuration storing microvolt streams in float32
        self.buffer = RingBuffer(max_samples=self.buffer_capacity, num_channels=self.num_channels)
        
        # Multi-threading control
        self._stop_event = threading.Event()
        self._producer_thread: Optional[threading.Thread] = None
        self._consumer_thread: Optional[threading.Thread] = None
        self._is_running = False
        
        # Keeps track of the total number of samples committed to disk
        self._last_read_index = 0

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self) -> None:
        """Launches acquisition and logger background loops."""
        if self._is_running:
            log.warning("Simulator is already running.")
            return

        self._is_running = True
        self._stop_event.clear()
        self._last_read_index = 0

        self._producer_thread = threading.Thread(
            target=self._acquisition_loop, 
            name="EEGAcquisitionThread", 
            daemon=True
        )
        self._consumer_thread = threading.Thread(
            target=self._writer_loop, 
            name="EEGWriterThread", 
            daemon=True
        )

        self._producer_thread.start()
        self._consumer_thread.start()
        log.info("Simulation pipeline running.")

    def stop(self) -> None:
        """Signals stop event and joins active loop executions."""
        if not self._is_running:
            return

        log.info("Stopping simulation pipeline...")
        self._stop_event.set()

        if self._producer_thread:
            self._producer_thread.join(timeout=2.0)
        if self._consumer_thread:
            self._consumer_thread.join(timeout=2.0)

        self._is_running = False
        log.info("Simulation pipeline halted.")

    def _acquisition_loop(self) -> None:
        """High-frequency clock-drift self-correcting generation loop."""
        sample_interval = 1.0 / self.config.sample_rate_hz
        start_time = time.time()
        samples_generated = 0

        while not self._stop_event.is_set():
            target_time = start_time + (samples_generated * sample_interval)
            sleep_time = target_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Generate sample and push channel data only (cast as float32 in push)
            sample_data = self.generator.generate_sample()
            self.buffer.push(sample_data)
            
            samples_generated += 1

    def _writer_loop(self) -> None:
        """Checks for new buffer inputs periodically to flush them to disk."""
        write_interval_seconds = 1.0
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=write_interval_seconds)
            self._flush_unread_data()

        # Shutdown flush sweep
        self._flush_unread_data()

    def _flush_unread_data(self) -> None:
        """Calculates precise backwards-looking sampling timestamps and logs data."""
        total_written = self.buffer.total_samples_written
        unread_count = total_written - self._last_read_index

        if unread_count <= 0:
            return

        # Restrict window size to capacity limit on system overflow
        unread_count = min(unread_count, self.buffer.max_samples)
        
        # Retrieve the chronological float32 signal frame matrix
        data_matrix = self.buffer.get_latest_window(unread_count)

        if data_matrix.size > 0:
            # Reconstruct exact time step timelines going backwards from this run
            now = time.time()
            sample_interval = 1.0 / self.config.sample_rate_hz
            timestamps = now - sample_interval * np.arange(unread_count)[::-1]

            # Write data through unified API
            self.logger.write_window(timestamps, data_matrix)
            self._last_read_index += unread_count