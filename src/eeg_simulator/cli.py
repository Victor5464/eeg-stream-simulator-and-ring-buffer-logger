import argparse
import sys
import time
import logging
from pathlib import Path
from eeg_simulator.config import SimulationConfig, ChannelConfig, LogFormat
from eeg_simulator.simulator import EEGSimulator

# Configure basic console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s"
)
log = logging.getLogger("eeg_cli")


def parse_channel_string(channel_str: str) -> ChannelConfig:
    """Parses a comma-separated channel definition from the CLI.
    
    Expected format: name,frequency_hz,amplitude_uv
    Example: "Fp1,10.0,50.0"
    """
    try:
        parts = channel_str.split(",")
        if len(parts) != 3:
            raise ValueError("Must contain exactly 3 components.")
        
        name = parts[0].strip()
        freq = float(parts[1])
        amp = float(parts[2])
        
        return ChannelConfig(name=name, frequency_hz=freq, amplitude_uv=amp)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid channel format '{channel_str}'. "
            f"Expected 'name,frequency,amplitude' (e.g., 'Fp1,10.0,50.0'). Error: {e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time Multi-channel EEG Signal Simulator & Logger CLI"
    )
    
    # Core simulation parameters
    parser.add_argument(
        "--channels",
        type=parse_channel_string,
        nargs="+",
        required=True,
        help="Space-separated list of channel configurations in the format: name,freq,amp (e.g., Fp1,10,50 O2,12,30)"
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=250,
        help="Sampling rate in Hz (default: 250)"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=10,
        help="Circular buffer capacity in seconds (default: 10)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=[f.value for f in LogFormat],
        default="csv",
        help="Output log format: csv, bin, or json (default: csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to write output files (default: ./data)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Simulation run duration in seconds. If 0, runs indefinitely until Ctrl+C (default: 0)"
    )

    args = parser.parse_args()

    # Create the domain configuration from arguments
    config = SimulationConfig(
        channels=args.channels,
        sample_rate_hz=args.rate,
        buffer_size_seconds=args.buffer_size,
        log_format=LogFormat(args.format),
        output_directory=Path(args.output_dir)
    )

    log.info("Initializing EEG Simulator CLI...")
    log.info(f"Target Output Directory: {config.output_directory.resolve()}")
    log.info(f"Configured Channels: {[(ch.name, ch.frequency_hz, ch.amplitude_uv) for ch in config.channels]}")

    simulator = EEGSimulator(config)

    try:
        simulator.start()
        log.info("Simulator running. Press Ctrl+C to terminate.")
        
        if args.duration > 0:
            log.info(f"Running for a fixed duration of {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        log.info("Keyboard interrupt received.")
    finally:
        log.info("Shutting down gracefully...")
        simulator.stop()
        log.info(f"Simulation completed. File stored at: {simulator.logger.file_path}")


if __name__ == "__main__":
    main()