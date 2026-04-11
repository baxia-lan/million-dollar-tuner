"""Audio format conversion and resampling utilities."""

from __future__ import annotations

import librosa
import numpy as np


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to a target sample rate."""
    if orig_sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


def normalize_peak(y: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize audio so peak amplitude matches *target_peak*."""
    peak = np.max(np.abs(y))
    if peak < 1e-8:
        return y
    return y * (target_peak / peak)


def normalize_rms(y: np.ndarray, target_rms: float | None = None) -> np.ndarray:
    """Normalize audio to a target RMS level.

    If *target_rms* is None, normalizes to -20 dBFS (~0.1).
    """
    if target_rms is None:
        target_rms = 10 ** (-20 / 20)  # ~0.1
    current_rms = np.sqrt(np.mean(y ** 2))
    if current_rms < 1e-8:
        return y
    return y * (target_rms / current_rms)


def match_rms(y: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Scale *y* so its RMS matches *reference*."""
    ref_rms = np.sqrt(np.mean(reference ** 2))
    return normalize_rms(y, target_rms=ref_rms)
