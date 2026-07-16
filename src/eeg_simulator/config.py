from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List

class LogFormat(Enum):
    CSV = "csv"
    BINARY = "bin"
    JSON = "json"

@dataclass(frozen=True)
class ChannelConfig:
    name: str
    frequency_hz: float
    amplitude_uv: float
    phase: float = 0.0

@dataclass(frozen=True)
class SimulationConfig:
    channels: List[ChannelConfig]
    sample_rate_hz: int = 250
    buffer_size_seconds: int = 10
    log_format: LogFormat = LogFormat.CSV
    output_directory: Path = field(default_factory=lambda: Path("./data"))

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be greater than 0.")
        if len(self.channels) == 0:
            raise ValueError("Must configure at least one channel.")