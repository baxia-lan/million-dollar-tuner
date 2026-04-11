"""DTW-based time alignment and variable-rate time stretching."""

from __future__ import annotations

import click
import librosa
import numpy as np
import pyrubberband as pyrb

from mdt.analysis.features import chroma_features
from mdt.config import ANALYSIS_SR, HOP_LENGTH, MAX_LENGTH_RATIO_DIFF


def align_vocals(
    user_audio: np.ndarray,
    user_sr: int,
    ref_audio: np.ndarray,
    ref_sr: int,
) -> np.ndarray:
    """Align user vocals to reference vocals using DTW.

    1. Computes chroma features for both signals.
    2. Runs DTW to find the optimal warping path.
    3. Converts the warping path into a time-stretch map.
    4. Applies variable-rate time stretching via pyrubberband.

    Parameters
    ----------
    user_audio : np.ndarray
        User's (already pitch-corrected) mono vocal audio.
    user_sr : int
        Sample rate of user audio.
    ref_audio : np.ndarray
        Reference mono vocal audio (from SUNO).
    ref_sr : int
        Sample rate of reference audio.

    Returns
    -------
    np.ndarray
        Time-aligned user audio, same sample rate as *user_sr*.
    """
    # Check length ratio
    user_dur = len(user_audio) / user_sr
    ref_dur = len(ref_audio) / ref_sr
    ratio_diff = abs(user_dur - ref_dur) / ref_dur

    if ratio_diff > MAX_LENGTH_RATIO_DIFF:
        click.echo(
            f"  WARNING: Recording length differs by {ratio_diff:.0%} "
            f"(user: {user_dur:.1f}s, ref: {ref_dur:.1f}s). "
            f"Results may have artifacts."
        )

    # Resample to analysis SR for feature extraction
    user_analysis = librosa.resample(user_audio, orig_sr=user_sr, target_sr=ANALYSIS_SR)
    ref_analysis = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=ANALYSIS_SR)

    # Compute chroma features
    user_chroma = chroma_features(user_analysis, sr=ANALYSIS_SR, hop_length=HOP_LENGTH)
    ref_chroma = chroma_features(ref_analysis, sr=ANALYSIS_SR, hop_length=HOP_LENGTH)

    # Replace NaN/zero columns to avoid cosine distance producing NaN
    # (silent frames have zero energy -> zero chroma -> NaN cosine)
    for C in (user_chroma, ref_chroma):
        bad = np.isnan(C).any(axis=0) | (np.sum(np.abs(C), axis=0) < 1e-10)
        C[:, bad] = 1e-10

    # Run DTW
    _D, wp = librosa.sequence.dtw(
        X=user_chroma, Y=ref_chroma, metric="cosine"
    )
    # wp is (N, 2) array of (user_frame, ref_frame) in reverse order
    wp = wp[::-1]

    # Convert warping path to a time map
    time_map = _warping_path_to_time_map(
        wp, user_sr, ANALYSIS_SR, HOP_LENGTH, user_audio, ref_audio, ref_sr
    )

    # Apply variable-rate time stretching
    aligned = _apply_time_map(user_audio, user_sr, time_map)

    return aligned


def _warping_path_to_time_map(
    wp: np.ndarray,
    user_sr: int,
    analysis_sr: int,
    hop_length: int,
    user_audio: np.ndarray,
    ref_audio: np.ndarray,
    ref_sr: int,
) -> list[tuple[float, float]]:
    """Convert a DTW warping path to a list of (source_time, target_time) pairs.

    Subsamples the path to keep only key anchor points (every ~0.5s)
    to avoid overly jerky time stretching.
    """
    # Convert frame indices to times
    user_times = librosa.frames_to_time(wp[:, 0], sr=analysis_sr, hop_length=hop_length)
    ref_times = librosa.frames_to_time(wp[:, 1], sr=analysis_sr, hop_length=hop_length)

    # Subsample: keep an anchor roughly every 0.5 seconds
    anchor_interval = 0.5
    time_map = [(0.0, 0.0)]  # Start anchor

    last_anchor_time = 0.0
    for i in range(len(user_times)):
        if ref_times[i] - last_anchor_time >= anchor_interval:
            time_map.append((user_times[i], ref_times[i]))
            last_anchor_time = ref_times[i]

    # End anchor
    user_end = len(user_audio) / user_sr
    ref_end = len(ref_audio) / ref_sr
    time_map.append((user_end, ref_end))

    return time_map


def _apply_time_map(
    y: np.ndarray,
    sr: int,
    time_map: list[tuple[float, float]],
) -> np.ndarray:
    """Apply variable-rate time stretching using pyrubberband.

    Splits the audio into segments based on anchor points and
    stretches each segment individually, then concatenates.
    """
    segments = []

    for i in range(len(time_map) - 1):
        src_start, tgt_start = time_map[i]
        src_end, tgt_end = time_map[i + 1]

        src_start_samp = int(src_start * sr)
        src_end_samp = int(src_end * sr)

        if src_end_samp <= src_start_samp:
            continue

        segment = y[src_start_samp:src_end_samp]
        src_dur = len(segment) / sr
        tgt_dur = tgt_end - tgt_start

        if tgt_dur <= 0 or src_dur <= 0:
            continue

        stretch_ratio = src_dur / tgt_dur

        # pyrubberband expects stretch_ratio > 0
        # ratio > 1 = speed up, ratio < 1 = slow down
        if abs(stretch_ratio - 1.0) < 0.01:
            segments.append(segment)
        else:
            # Clamp extreme ratios
            stretch_ratio = max(0.25, min(4.0, stretch_ratio))
            stretched = pyrb.time_stretch(segment, sr, stretch_ratio)
            segments.append(stretched)

    if not segments:
        return y

    aligned = np.concatenate(segments)
    return aligned.astype(np.float32)


def simple_time_stretch(
    y: np.ndarray,
    sr: int,
    target_duration: float,
) -> np.ndarray:
    """Simple uniform time stretch to match a target duration.

    A simpler alternative to DTW alignment when the user's recording
    is roughly in sync but just a different tempo.
    """
    current_duration = len(y) / sr
    if current_duration <= 0:
        return y

    ratio = current_duration / target_duration
    if abs(ratio - 1.0) < 0.01:
        return y

    ratio = max(0.25, min(4.0, ratio))
    return pyrb.time_stretch(y, sr, ratio).astype(np.float32)
