import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from eeg_simulator.config import SimulationConfig, ChannelConfig
from eeg_simulator.simulator import EEGSimulator

# 1. Create a fast simulator configuration
channels = [
    ChannelConfig(name="Fp1", frequency_hz=8.0, amplitude_uv=50.0),
    ChannelConfig(name="O2", frequency_hz=12.0, amplitude_uv=30.0),
]
config = SimulationConfig(
    channels=channels,
    sample_rate_hz=250,
    buffer_size_seconds=4  
)

simulator = EEGSimulator(config)
simulator.start()

# 2. Setup Matplotlib Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.suptitle("Real-Time EEG Stream Simulator")

# Retrieve the initial buffer shape
total_samples = simulator.buffer_capacity
time_axis = [i / config.sample_rate_hz for i in range(total_samples)]

line1, = ax1.plot(time_axis, [0.0] * total_samples, color="royalblue", label="Fp1")
line2, = ax2.plot(time_axis, [0.0] * total_samples, color="crimson", label="O2")

ax1.set_ylim(-100, 100)
ax2.set_ylim(-100, 100)
ax1.legend(loc="upper right")
ax2.legend(loc="upper right")
ax2.set_ylabel("Amplitude (uV)")
ax2.set_ylabel("Amplitude (uV)")
ax2.set_xlabel("Seconds (Buffer History)")

def update(frame):
    """Periodically queries the active simulator's buffer and updates lines."""
    if not simulator.is_running:
        sys.exit()
        
    # Grab the current state of the circular buffer
    data = simulator.buffer.get_latest_window(total_samples)
    
    if data.size > 0:
        actual_length = len(data)
        
        # Zero-pad if the buffer hasn't reached maximum capacity yet
        if actual_length < total_samples:
            padding_needed = total_samples - actual_length
            padding = np.zeros((padding_needed, config.channels.__len__()))
            data = np.vstack((padding, data))
            
        line1.set_ydata(data[:, 0])  # Channel 1 (Fp1)
        line2.set_ydata(data[:, 1])  # Channel 2 (O2)
        
    return line1, line2

# Pass cache_frame_data=False to silence warning
ani = FuncAnimation(fig, update, interval=33, blit=True, cache_frame_data=False)

try:
    plt.show()
finally:
    simulator.stop()