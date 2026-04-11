"""Auto-tune engine: correct user pitch to match a reference pitch curve."""

from __future__ import annotations

import numpy as np
import psola

from mdt.analysis.pitch import PitchCurve, detect_pitch, midi_to_hz
from mdt.config import (
    ANALYSIS_SR,
    DEFAULT_CORRECTION_STRENGTH,
    MAX_PITCH_SHIFT_SEMITONES,
)


def correct_pitch(
    y: np.ndarray,
    sr: int,
    user_pitch: PitchCurve,
    target_pitch: PitchCurve,
    strength: float = DEFAULT_CORRECTION_STRENGTH,
    max_shift: float = MAX_PITCH_SHIFT_SEMITONES,
) -> np.ndarray:
    """Correct user's vocal pitch to match the target pitch curve.

    Uses PSOLA (Pitch-Synchronous Overlap and Add) for high-quality
    pitch shifting that preserves formants.

    Parameters
    ----------
    y : np.ndarray
        User's mono vocal audio.
    sr : int
        Sample rate.
    user_pitch : PitchCurve
        Detected pitch of the user's vocals.
    target_pitch : PitchCurve
        Target pitch curve (from reference/SUNO vocals).
    strength : float
        Correction strength [0.0, 1.0].
        0.0 = no correction, 1.0 = full correction.
    max_shift : float
        Maximum allowed pitch shift in semitones.

    Returns
    -------
    np.ndarray
        Pitch-corrected audio.
    """
    # Build the target F0 contour for the user's audio frames
    corrected_f0 = _build_corrected_f0(
        user_pitch=user_pitch,
        target_pitch=target_pitch,
        strength=strength,
        max_shift=max_shift,
    )

    # Apply pitch correction using PSOLA
    corrected_audio = psola.vocode(
        audio=y,
        sample_rate=sr,
        target_pitch=corrected_f0,
        fmin=50.0,
        fmax=2200.0,
    )

    return corrected_audio.astype(np.float32)


def _build_corrected_f0(
    user_pitch: PitchCurve,
    target_pitch: PitchCurve,
    strength: float,
    max_shift: float,
) -> np.ndarray:
    """Build a frame-by-frame target F0 array for pitch correction.

    For each voiced frame in the user's audio:
    1. Look up the corresponding target pitch at the same timestamp.
    2. Interpolate between detected and target based on *strength*.
    3. Clamp the shift to *max_shift* semitones.

    Unvoiced frames get NaN (PSOLA will leave them untouched).
    """
    n_frames = len(user_pitch.frequencies)
    corrected = np.full(n_frames, np.nan, dtype=np.float64)

    # Interpolate target pitch at user's timestamps
    target_interp = np.interp(
        user_pitch.times,
        target_pitch.times,
        target_pitch.frequencies,
        left=np.nan,
        right=np.nan,
    )

    for i in range(n_frames):
        user_f0 = user_pitch.frequencies[i]
        tgt_f0 = target_interp[i]

        # Skip unvoiced frames
        if np.isnan(user_f0) or not user_pitch.voiced_flag[i]:
            continue

        # If target is unvoiced, keep user's pitch
        if np.isnan(tgt_f0):
            corrected[i] = user_f0
            continue

        # Calculate shift in semitones
        shift_semitones = 12.0 * np.log2(tgt_f0 / user_f0)

        # Clamp shift
        shift_semitones = np.clip(shift_semitones, -max_shift, max_shift)

        # Apply strength interpolation
        shift_semitones *= strength

        # Convert back to frequency
        corrected[i] = user_f0 * (2.0 ** (shift_semitones / 12.0))

    # Smooth the F0 contour to avoid discontinuities
    corrected = _smooth_f0(corrected, window_size=5)

    return corrected


def _smooth_f0(f0: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply median smoothing to F0, ignoring NaN values."""
    from scipy.ndimage import median_filter

    # Only smooth voiced regions
    voiced = ~np.isnan(f0)
    if np.sum(voiced) < window_size:
        return f0

    smoothed = f0.copy()
    # Apply median filter to voiced segments
    voiced_vals = f0[voiced]
    if len(voiced_vals) > window_size:
        smoothed_vals = median_filter(voiced_vals, size=window_size)
        smoothed[voiced] = smoothed_vals

    return smoothed


def correct_pitch_to_scale(
    y: np.ndarray,
    sr: int,
    scale_notes: list[str] | None = None,
    strength: float = DEFAULT_CORRECTION_STRENGTH,
) -> np.ndarray:
    """Simpler auto-tune: snap pitch to the nearest scale degree.

    This is a traditional auto-tune effect (like T-Pain / Cher)
    rather than matching a reference track.

    Parameters
    ----------
    y : np.ndarray
        Mono vocal audio.
    sr : int
        Sample rate.
    scale_notes : list of str or None
        Notes in the scale (e.g. ["C", "D", "E", "F", "G", "A", "B"]).
        None defaults to chromatic (all 12 notes).
    strength : float
        Correction strength.

    Returns
    -------
    np.ndarray
        Pitch-corrected audio.
    """
    import librosa

    pitch_curve = detect_pitch(y, sr=sr)

    if scale_notes is None:
        # Chromatic – snap to nearest semitone
        scale_midi = np.arange(128)
    else:
        note_to_class = {
            "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
            "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
            "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
        }
        classes = [note_to_class[n] for n in scale_notes]
        scale_midi = []
        for octave in range(11):
            for c in classes:
                midi_note = octave * 12 + c
                if 0 <= midi_note < 128:
                    scale_midi.append(midi_note)
        scale_midi = np.array(sorted(set(scale_midi)))

    # Build corrected F0 by snapping each frame to nearest scale degree
    n_frames = len(pitch_curve.frequencies)
    corrected_f0 = np.full(n_frames, np.nan)

    for i in range(n_frames):
        f0 = pitch_curve.frequencies[i]
        if np.isnan(f0) or not pitch_curve.voiced_flag[i]:
            continue

        midi_val = 69 + 12 * np.log2(f0 / 440.0)
        # Find nearest scale note
        nearest_idx = np.argmin(np.abs(scale_midi - midi_val))
        target_midi = scale_midi[nearest_idx]
        target_f0 = midi_to_hz(np.array([target_midi]))[0]

        # Interpolate with strength
        shift = 12.0 * np.log2(target_f0 / f0) * strength
        corrected_f0[i] = f0 * (2.0 ** (shift / 12.0))

    corrected_audio = psola.vocode(
        audio=y,
        sample_rate=sr,
        target_pitch=corrected_f0,
        fmin=50.0,
        fmax=2200.0,
    )

    return corrected_audio.astype(np.float32)
