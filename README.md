# Real-Time EEG Signal Simulator & Logging Pipeline

A high-performance, multi-threaded real-time EEG (Electroencephalography) signal simulator and data acquisition pipeline written in Python. 

This repository implements a production-grade **Producer-Consumer architecture** designed to mimic clinical-grade biopotential acquisition hardware. By isolating high-frequency signal generation (acquisition) from heavy disk operations (I/O), the pipeline achieves real-time temporal fidelity and zero-overhead binary streaming. It serves as an ideal template for Brain-Computer Interface (BCI) developers looking to prototype algorithms, dashboards, or hardware APIs without requiring a physical headset.

---

## Key Features

* **Multi-Threaded Architecture:** Utilizes Python's `threading` library with dedicated, synchronized background threads for high-speed sampling (Producer) and disk-writing (Consumer).
* **Self-Correcting Clock:** Implements a drift-correcting loop on the acquisition thread to guarantee precise sampling intervals (e.g., 250 Hz, 1000 Hz) even during system scheduling delays.
* **Optimized `float32` Ring Buffer:** A high-speed, thread-safe circular memory buffer that minimizes RAM allocation overhead and handles continuous sliding window queries.
* **Vectorized Binary Logging:** Features an ultra-efficient zero-copy binary writer that bypasses ASCII string formatting. Writes data as contiguous raw memory structures (8-byte Unix timestamp + N x 4-byte `float32` channel voltages).
* **Flexible Format Fallbacks:** Supports legacy `CSV` and `JSON` formats for rapid debugging and manual file inspection.
* **Synthetic Waveform Synthesis:** Generates multi-channel biosignals by modeling native brain rhythms (Alpha, Beta, Theta, etc.) combined with customizable physiological high-frequency noise.

---

## Architecture Overview

    ```text
    eeg_simulator/
    ├── __init__.py
    ├── config.py         # Config schemas for hardware, channels, and log formats
    ├── generator.py      # Mathematical biosignal generators (sine waves + noise)
    ├── buffer.py         # Ultra-fast NumPy circular ring buffer (float32)
    ├── logger.py         # Multi-format vectorized file writer (Binary/CSV/JSON)
    └── simulator.py      # Thread coordinator running the acquisition loops
    tests/
    ├── test_logger.py    # Unit tests for multi-format file operations
    └── test_simulator.py # End-to-end multi-threaded integration assertions
---

## Getting Started

### Prerequisites
* Python 3.8 or higher installed on your system.
* SQLAlchemy installed in your environment.

### Installation
1. Clone the repository to your local machine:
   ```bash
   cd eeg-stream-simulator-and-ring-buffer-logger

2. Navigate into the project directory:
   ```bash
   git clone (https://github.com/yourusername/eeg-stream-simulator-and-ring-buffer-logger.git)


### Running the Application
* Launch the script directly from your terminal:
    ```bash
    python cli.py


## With uv

Install uv (if you do not have it):

* On macOS/Linux
    ```bash
    curl -LsSf https://astral.sh | sh

* On Windows
    ```bash
    powershell -c "irm https://astral.sh | iex"

* Run the project instantly:
    ```bash
    uv run cli.py


### License
    This project is open-source and available under the MIT License.
