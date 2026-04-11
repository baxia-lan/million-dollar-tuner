"""User-facing parameter controls for resynthesis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SynthControls:
    """Per-stem synthesis parameters."""

    # General
    volume: float = 1.0            # [0.0, 2.0]
    pan: float = 0.0               # [-1.0, 1.0] (left to right)

    # Effects
    reverb_wet: float = 0.1        # [0.0, 1.0]
    reverb_room: float = 0.3       # [0.0, 1.0]
    eq_low_gain_db: float = 0.0    # Low shelf gain
    eq_mid_gain_db: float = 0.0    # Parametric mid gain
    eq_high_gain_db: float = 0.0   # High shelf gain

    # Synthesis
    soundfont: str | None = None   # Override soundfont for this stem
    program: int | None = None     # Override MIDI program number
    transpose: int = 0             # Semitones to transpose


@dataclass
class ResynthConfig:
    """Configuration for the full resynthesis pipeline."""

    stems: dict[str, SynthControls] = field(default_factory=lambda: {
        "vocals": SynthControls(volume=0.0),  # Muted by default
        "drums": SynthControls(),
        "bass": SynthControls(),
        "other": SynthControls(),
    })
    master_volume: float = 0.9
    sample_rate: int = 44100
    soundfont: str | None = None   # Global default soundfont
