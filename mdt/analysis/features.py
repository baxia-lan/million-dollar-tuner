"""Feature extraction for DTW time alignment."""

from __future__ import annotations

import librosa
import numpy as np

from mdt.config import ANALYSIS_SR, HOP_LENGTH


def chroma_features(
    y: np.ndarray,
    sr: int = ANALYSIS_SR,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Compute chroma-CQT features.

    Returns
    -------
    np.ndarray
        Shape ``(12, n_frames)`` chroma feature matrix.
    """
    return librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=hop_length
    )


def mfcc_features(
    y: np.ndarray,
    sr: int = ANALYSIS_SR,
    hop_length: int = HOP_LENGTH,
    n_mfcc: int = 13,
) -> np.ndarray:
    """Compute MFCC features.

    Returns
    -------
    np.ndarray
        Shape ``(n_mfcc, n_frames)`` MFCC feature matrix.
    """
    return librosa.feature.mfcc(
        y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc
    )


def spectral_envelope(
    y: np.ndarray,
    sr: int = ANALYSIS_SR,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Compute spectral centroid, bandwidth, and rolloff.

    Returns
    -------
    np.ndarray
        Shape ``(3, n_frames)`` spectral features.
    """
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, hop_length=hop_length
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, hop_length=hop_length
    )
    return np.vstack([centroid, bandwidth, rolloff])
