"""Backend protocol and shared data structures for voice conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class TrainResult:
    model_path: Path | None
    index_path: Path | None = None
    config_path: Path | None = None
    train_time_seconds: float = 0.0
    epochs: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class InferResult:
    output_path: Path
    sample_rate: int = 44100
    infer_time_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    sim_to_user: float = 0.0
    sim_to_source: float = 0.0
    sim_baseline: float = 0.0
    f0_correlation: float = 0.0
    f0_mean_error_hz: float = 0.0
    f0_voiced_frames: int = 0


@runtime_checkable
class VCBackend(Protocol):
    name: str
    requires_training: bool

    def is_available(self) -> bool: ...

    def train(
        self,
        profile_dir: Path,
        audio_path: Path,
        *,
        epochs: int = 20,
        batch_size: int = 4,
        **kwargs,
    ) -> TrainResult: ...

    def infer(
        self,
        profile_dir: Path,
        source_vocals: Path,
        output_path: Path,
        *,
        pitch_shift: int = 0,
        **kwargs,
    ) -> InferResult: ...
