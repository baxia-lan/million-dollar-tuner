"""Data classes for stem separation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StemResult:
    """Container for separated audio stems."""

    vocals_path: Path | None = None
    drums_path: Path | None = None
    bass_path: Path | None = None
    other_path: Path | None = None
    output_dir: Path | None = None
    model: str = ""
    source_path: Path | None = None

    # Extra stems for 6-source model
    guitar_path: Path | None = None
    piano_path: Path | None = None

    @property
    def instrumental_paths(self) -> list[Path]:
        """Return paths to all non-vocal stems."""
        paths = []
        for p in [self.drums_path, self.bass_path, self.other_path,
                  self.guitar_path, self.piano_path]:
            if p is not None:
                paths.append(p)
        return paths

    @property
    def all_paths(self) -> dict[str, Path]:
        """Return a dict of stem_name -> path for all available stems."""
        result = {}
        for name in ["vocals", "drums", "bass", "other", "guitar", "piano"]:
            p = getattr(self, f"{name}_path")
            if p is not None:
                result[name] = p
        return result
