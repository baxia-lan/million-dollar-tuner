"""Tempo, beat, and onset detection."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from mdt.config import ANALYSIS_SR, HOP_LENGTH


@dataclass
class RhythmProfile:
    """Rhythmic analysis result."""

    tempo: float               # BPM
    beat_times: np.ndarray     # (n_beats,) beat positions in seconds
    onset_times: np.ndarray    # (n_onsets,) onset positions in seconds
    sr: int
    hop_length: int

    @property
    def beat_interval(self) -> float:
        """Average inter-beat interval in seconds."""
        if len(self.beat_times) < 2:
            return 60.0 / self.tempo
        return float(np.mean(np.diff(self.beat_times)))


def analyze_rhythm(
    y: np.ndarray,
    sr: int = ANALYSIS_SR,
    hop_length: int = HOP_LENGTH,
) -> RhythmProfile:
    """Analyze tempo, beats, and onsets.

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
    RhythmProfile
    """
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length
    )
    # librosa may return tempo as array; extract scalar
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    else:
        tempo = float(tempo)

    beat_times = librosa.frames_to_time(
        beat_frames, sr=sr, hop_length=hop_length
    )

    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length
    )
    onset_times = librosa.frames_to_time(
        onset_frames, sr=sr, hop_length=hop_length
    )

    return RhythmProfile(
        tempo=tempo,
        beat_times=beat_times,
        onset_times=onset_times,
        sr=sr,
        hop_length=hop_length,
    )
