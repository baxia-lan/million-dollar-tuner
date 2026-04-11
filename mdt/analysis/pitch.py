"""Pitch detection using PYIN (librosa)."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from mdt.config import ANALYSIS_SR, HOP_LENGTH, PITCH_FMAX, PITCH_FMIN


@dataclass
class PitchCurve:
    """Frame-by-frame pitch detection result."""

    times: np.ndarray        # (n_frames,) timestamps in seconds
    frequencies: np.ndarray  # (n_frames,) F0 in Hz, NaN for unvoiced
    voiced_flag: np.ndarray  # (n_frames,) bool
    voiced_prob: np.ndarray  # (n_frames,) confidence [0, 1]
    sr: int
    hop_length: int

    @property
    def midi_notes(self) -> np.ndarray:
        """Convert frequencies to MIDI note numbers (NaN preserved)."""
        return hz_to_midi(self.frequencies)

    @property
    def voiced_frequencies(self) -> np.ndarray:
        """Return only voiced frequencies (unvoiced replaced with NaN)."""
        result = self.frequencies.copy()
        result[~self.voiced_flag] = np.nan
        return result

    def get_freq_at_time(self, t: float) -> float:
        """Get the interpolated frequency at time *t*."""
        idx = np.searchsorted(self.times, t)
        idx = min(idx, len(self.frequencies) - 1)
        return self.frequencies[idx]


def detect_pitch(
    y: np.ndarray,
    sr: int = ANALYSIS_SR,
    fmin: float = PITCH_FMIN,
    fmax: float = PITCH_FMAX,
    hop_length: int = HOP_LENGTH,
) -> PitchCurve:
    """Detect pitch using PYIN algorithm.

    Parameters
    ----------
    y : np.ndarray
        Mono audio signal.
    sr : int
        Sample rate.
    fmin, fmax : float
        Min/max frequency range for pitch detection.
    hop_length : int
        Hop size in samples.

    Returns
    -------
    PitchCurve
        Detected pitch information.
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
    )

    n_frames = len(f0)
    times = librosa.frames_to_time(
        np.arange(n_frames), sr=sr, hop_length=hop_length
    )

    return PitchCurve(
        times=times,
        frequencies=f0,
        voiced_flag=voiced_flag,
        voiced_prob=voiced_prob,
        sr=sr,
        hop_length=hop_length,
    )


def hz_to_midi(frequencies: np.ndarray) -> np.ndarray:
    """Convert Hz to MIDI note numbers. NaN in, NaN out."""
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69 + 12 * np.log2(frequencies / 440.0)
    return midi


def midi_to_hz(midi_notes: np.ndarray) -> np.ndarray:
    """Convert MIDI note numbers to Hz. NaN in, NaN out."""
    return 440.0 * (2.0 ** ((midi_notes - 69) / 12.0))


def pitch_difference_semitones(
    detected: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Compute pitch difference in semitones between two F0 arrays.

    Returns positive values when *detected* is sharp (above target).
    NaN where either input is NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = 12.0 * np.log2(detected / target)
    return diff
