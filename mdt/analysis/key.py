"""Musical key detection using Krumhansl-Schmuckler algorithm."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from mdt.config import ANALYSIS_SR, HOP_LENGTH

# Krumhansl-Kessler pitch-class profiles
_MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
])
_MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
])

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class KeyResult:
    """Musical key detection result."""

    key: str          # e.g. "C major", "A minor"
    root: str         # e.g. "C", "A"
    mode: str         # "major" or "minor"
    confidence: float # correlation coefficient [0, 1]

    def __str__(self) -> str:
        return f"{self.key} (confidence: {self.confidence:.2f})"


def detect_key(
    y: np.ndarray,
    sr: int = ANALYSIS_SR,
    hop_length: int = HOP_LENGTH,
) -> KeyResult:
    """Detect the musical key using chroma features and the K-K algorithm.

    Parameters
    ----------
    y : np.ndarray
        Mono audio signal.
    sr : int
        Sample rate.
    hop_length : int
        Hop size in samples.

    Returns
    -------
    KeyResult
    """
    # Compute chroma features
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    # Sum across time to get pitch-class energy distribution
    pitch_class = np.sum(chroma, axis=1)  # shape: (12,)

    best_corr = -np.inf
    best_root = 0
    best_mode = "major"

    for root in range(12):
        # Rotate profile to match the root note
        major_shifted = np.roll(_MAJOR_PROFILE, root)
        minor_shifted = np.roll(_MINOR_PROFILE, root)

        corr_major = np.corrcoef(pitch_class, major_shifted)[0, 1]
        corr_minor = np.corrcoef(pitch_class, minor_shifted)[0, 1]

        if corr_major > best_corr:
            best_corr = corr_major
            best_root = root
            best_mode = "major"

        if corr_minor > best_corr:
            best_corr = corr_minor
            best_root = root
            best_mode = "minor"

    root_name = _NOTE_NAMES[best_root]

    return KeyResult(
        key=f"{root_name} {best_mode}",
        root=root_name,
        mode=best_mode,
        confidence=max(0.0, best_corr),
    )
