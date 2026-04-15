"""Profile metadata schema."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class BackendStatus:
    trained: bool = False
    model_path: str | None = None
    index_path: str | None = None
    trained_at: str | None = None
    epochs: int = 0
    train_time_seconds: float = 0.0


@dataclass
class ProfileMetadata:
    name: str
    created_at: str = ""
    updated_at: str = ""
    source_audio: str = ""
    source_audio_hash: str = ""
    backends: dict[str, BackendStatus] = field(default_factory=dict)

    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
        # Deserialize nested dicts into BackendStatus
        for k, v in self.backends.items():
            if isinstance(v, dict):
                self.backends[k] = BackendStatus(**v)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ProfileMetadata:
        with open(path) as f:
            return cls(**json.load(f))

    def touch(self):
        self.updated_at = datetime.now(timezone.utc).isoformat()
