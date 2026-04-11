"""Quality evaluation metrics for vocal replacement results.

Defines measurable criteria to assess whether the output is good enough:

1. Pitch Accuracy  — corrected pitch vs target pitch (median deviation in cents)
2. Time Alignment  — onset alignment error (ms) between aligned and reference vocals
3. Vocal Clarity   — spectral smoothness of processed vocals (artifact detection)
4. Mix Balance     — vocal loudness vs instrumental loudness relative to reference

Quality thresholds (GOOD / OK / POOR):
- Pitch: <25 cents / 25-50 / >50
- Timing: <50ms / 50-120ms / >120ms
- Clarity: >0.7 / 0.5-0.7 / <0.5 (correlation with harmonic template)
- Balance: <3dB / 3-8dB / >8dB difference from reference ratio
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

from mdt.analysis.pitch import PitchCurve, detect_pitch, pitch_difference_semitones
from mdt.analysis.rhythm import analyze_rhythm
from mdt.config import ANALYSIS_SR, HOP_LENGTH


# ── Thresholds ──────────────────────────────────────────────────

PITCH_GOOD = 25.0   # cents
PITCH_OK = 50.0

TIMING_GOOD = 50.0   # ms
TIMING_OK = 120.0

CLARITY_GOOD = 0.70  # spectral correlation
CLARITY_OK = 0.50

BALANCE_GOOD = 3.0   # dB
BALANCE_OK = 8.0


@dataclass
class QualityGrade:
    """Grade for a single metric."""

    name: str
    value: float
    unit: str
    grade: str          # "GOOD", "OK", "POOR"
    detail: str = ""

    def __str__(self) -> str:
        symbol = {"GOOD": "+", "OK": "~", "POOR": "!"}[self.grade]
        return f"  [{symbol}] {self.name}: {self.value:.1f} {self.unit} — {self.grade}"


@dataclass
class QualityReport:
    """Full quality assessment of a vocal replacement result."""

    grades: list[QualityGrade] = field(default_factory=list)
    overall_score: float = 0.0

    @property
    def passed(self) -> bool:
        """True if no metric is POOR."""
        return all(g.grade != "POOR" for g in self.grades)

    @property
    def summary(self) -> str:
        grade_map = {
            (90, 101): "A — Excellent",
            (75, 90):  "B — Good",
            (60, 75):  "C — Acceptable",
            (40, 60):  "D — Needs improvement",
            (0, 40):   "F — Poor",
        }
        for (lo, hi), label in grade_map.items():
            if lo <= self.overall_score < hi:
                return label
        return "?"

    def __str__(self) -> str:
        lines = ["\n  Quality Report"]
        lines.append("  " + "─" * 50)
        for g in self.grades:
            lines.append(str(g))
        lines.append("  " + "─" * 50)
        lines.append(f"  Overall Score: {self.overall_score:.0f}/100 — {self.summary}")
        if not self.passed:
            lines.append("  WARNING: Some metrics are below acceptable thresholds.")
        return "\n".join(lines)


# ── Evaluation functions ────────────────────────────────────────

def evaluate_pitch_accuracy(
    corrected_audio: np.ndarray,
    sr: int,
    target_pitch: PitchCurve,
) -> QualityGrade:
    """Measure how close the corrected pitch is to the target.

    Computes median absolute pitch deviation in cents.
    """
    corrected_pitch = detect_pitch(corrected_audio, sr=sr)

    # Interpolate target to corrected timestamps
    target_f0_interp = np.interp(
        corrected_pitch.times,
        target_pitch.times,
        target_pitch.frequencies,
        left=np.nan,
        right=np.nan,
    )

    # Both must be voiced
    both_voiced = (
        corrected_pitch.voiced_flag
        & ~np.isnan(target_f0_interp)
        & ~np.isnan(corrected_pitch.frequencies)
        & (target_f0_interp > 0)
        & (corrected_pitch.frequencies > 0)
    )

    if np.sum(both_voiced) < 10:
        return QualityGrade(
            name="Pitch Accuracy",
            value=0.0,
            unit="cents",
            grade="OK",
            detail="Not enough voiced frames to evaluate",
        )

    # Deviation in cents (100 cents = 1 semitone)
    diff_semitones = pitch_difference_semitones(
        corrected_pitch.frequencies[both_voiced],
        target_f0_interp[both_voiced],
    )
    deviation_cents = np.abs(diff_semitones) * 100
    median_dev = float(np.nanmedian(deviation_cents))

    if median_dev < PITCH_GOOD:
        grade = "GOOD"
    elif median_dev < PITCH_OK:
        grade = "OK"
    else:
        grade = "POOR"

    return QualityGrade(
        name="Pitch Accuracy",
        value=median_dev,
        unit="cents deviation",
        grade=grade,
        detail=f"Median |target - actual| over {np.sum(both_voiced)} voiced frames",
    )


def evaluate_timing_accuracy(
    aligned_audio: np.ndarray,
    aligned_sr: int,
    ref_audio: np.ndarray,
    ref_sr: int,
) -> QualityGrade:
    """Measure timing alignment by comparing onset positions.

    For each onset in the reference, finds the nearest onset in the
    aligned audio and measures the error. Uses beat tracking as
    fallback if onsets are too sparse.
    """
    ref_rhythm = analyze_rhythm(ref_audio, sr=ref_sr)
    aligned_rhythm = analyze_rhythm(aligned_audio, sr=aligned_sr)

    ref_onsets = ref_rhythm.onset_times
    aligned_onsets = aligned_rhythm.onset_times

    if len(ref_onsets) < 3 or len(aligned_onsets) < 3:
        # Fall back to beat alignment
        ref_onsets = ref_rhythm.beat_times
        aligned_onsets = aligned_rhythm.beat_times

    if len(ref_onsets) < 2 or len(aligned_onsets) < 2:
        return QualityGrade(
            name="Timing Accuracy",
            value=0.0,
            unit="ms",
            grade="OK",
            detail="Not enough events to evaluate timing",
        )

    # For each reference onset, find the nearest aligned onset
    errors_ms = []
    for ref_t in ref_onsets:
        nearest_idx = np.argmin(np.abs(aligned_onsets - ref_t))
        error_ms = abs(aligned_onsets[nearest_idx] - ref_t) * 1000
        errors_ms.append(error_ms)

    median_error = float(np.median(errors_ms))

    if median_error < TIMING_GOOD:
        grade = "GOOD"
    elif median_error < TIMING_OK:
        grade = "OK"
    else:
        grade = "POOR"

    return QualityGrade(
        name="Timing Accuracy",
        value=median_error,
        unit="ms onset error",
        grade=grade,
        detail=f"Median nearest-onset error over {len(errors_ms)} events",
    )


def evaluate_vocal_clarity(
    processed_audio: np.ndarray,
    sr: int,
) -> QualityGrade:
    """Measure vocal clarity by detecting processing artifacts.

    Uses spectral flatness as a proxy: natural vocals have low spectral
    flatness (strong harmonic peaks), while artifacts (clicks, metallic
    sounds, phase distortion) increase flatness/noise floor.

    Also checks for energy discontinuities (clicks from bad time-stretching).
    """
    # 1. Spectral flatness (Wiener entropy)
    # Low = tonal/harmonic (good), High = noisy/artifact-heavy (bad)
    flatness = librosa.feature.spectral_flatness(y=processed_audio, hop_length=HOP_LENGTH)
    mean_flatness = float(np.mean(flatness))

    # 2. Check for clicks/discontinuities
    # Compute short-time energy and look for sudden jumps
    frame_length = 1024
    hop = 512
    energy = np.array([
        np.sum(processed_audio[i:i + frame_length] ** 2)
        for i in range(0, len(processed_audio) - frame_length, hop)
    ])
    if len(energy) > 2:
        energy_diff = np.abs(np.diff(energy))
        energy_mean = np.mean(energy) + 1e-10
        # Normalize jumps relative to mean energy
        jump_ratio = np.mean(energy_diff) / energy_mean
    else:
        jump_ratio = 0.0

    # 3. Harmonic-to-noise ratio proxy via autocorrelation
    # Strong autocorrelation peak = clear pitch = good vocal quality
    if len(processed_audio) > sr // 4:
        segment = processed_audio[:sr]  # First second
        autocorr = np.correlate(segment[:4096], segment[:4096], mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
            # Find strongest peak after lag=50 (above 880Hz)
            # and before lag=800 (above 55Hz)
            search_region = autocorr[50:800] if len(autocorr) > 800 else autocorr[50:]
            peak_strength = float(np.max(search_region)) if len(search_region) > 0 else 0.0
        else:
            peak_strength = 0.0
    else:
        peak_strength = 0.5

    # Combine into a clarity score [0, 1]
    # Low flatness = good, low jump_ratio = good, high peak_strength = good
    flatness_score = max(0.0, 1.0 - mean_flatness * 10)  # flatness ~0.01-0.1
    jump_score = max(0.0, 1.0 - jump_ratio * 5)
    harmonic_score = peak_strength

    clarity = 0.4 * flatness_score + 0.3 * jump_score + 0.3 * harmonic_score
    clarity = max(0.0, min(1.0, clarity))

    if clarity > CLARITY_GOOD:
        grade = "GOOD"
    elif clarity > CLARITY_OK:
        grade = "OK"
    else:
        grade = "POOR"

    return QualityGrade(
        name="Vocal Clarity",
        value=clarity,
        unit="(0-1 score)",
        grade=grade,
        detail=f"flatness={mean_flatness:.3f} jumps={jump_ratio:.3f} harmonic={peak_strength:.2f}",
    )


def evaluate_mix_balance(
    vocal_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    ref_vocal_audio: np.ndarray,
    ref_instrumental_audio: np.ndarray | None = None,
) -> QualityGrade:
    """Check if the vocal/instrumental balance matches the original.

    Compares the vocal-to-instrumental RMS ratio against the reference ratio.
    """
    def rms_db(y: np.ndarray) -> float:
        rms = np.sqrt(np.mean(y ** 2))
        if rms < 1e-10:
            return -80.0
        return 20.0 * np.log10(rms)

    vocal_rms = rms_db(vocal_audio)
    inst_rms = rms_db(instrumental_audio)
    user_ratio = vocal_rms - inst_rms

    ref_vocal_rms = rms_db(ref_vocal_audio)
    if ref_instrumental_audio is not None:
        ref_inst_rms = rms_db(ref_instrumental_audio)
    else:
        ref_inst_rms = inst_rms
    ref_ratio = ref_vocal_rms - ref_inst_rms

    balance_diff = abs(user_ratio - ref_ratio)

    if balance_diff < BALANCE_GOOD:
        grade = "GOOD"
    elif balance_diff < BALANCE_OK:
        grade = "OK"
    else:
        grade = "POOR"

    return QualityGrade(
        name="Mix Balance",
        value=balance_diff,
        unit="dB from reference",
        grade=grade,
        detail=f"Vocal/inst ratio: {user_ratio:.1f}dB (ref: {ref_ratio:.1f}dB)",
    )


def evaluate_mix_balance_overall(
    final_mix: np.ndarray,
    original_mix: np.ndarray,
) -> QualityGrade:
    """Compare overall loudness and spectral balance of the final mix vs original.

    This is a simpler, more reliable balance check: the output song
    should have similar overall loudness and spectral distribution
    as the original SUNO song.
    """
    def rms_db(y: np.ndarray) -> float:
        rms = np.sqrt(np.mean(y ** 2))
        if rms < 1e-10:
            return -80.0
        return 20.0 * np.log10(rms)

    min_len = min(len(final_mix), len(original_mix))
    final_rms = rms_db(final_mix[:min_len])
    orig_rms = rms_db(original_mix[:min_len])

    loudness_diff = abs(final_rms - orig_rms)

    if loudness_diff < BALANCE_GOOD:
        grade = "GOOD"
    elif loudness_diff < BALANCE_OK:
        grade = "OK"
    else:
        grade = "POOR"

    return QualityGrade(
        name="Mix Balance",
        value=loudness_diff,
        unit="dB from original",
        grade=grade,
        detail=f"Output: {final_rms:.1f}dB RMS, Original: {orig_rms:.1f}dB RMS",
    )


def evaluate_full(
    corrected_audio: np.ndarray,
    aligned_audio: np.ndarray,
    original_user_audio: np.ndarray,
    ref_vocal_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    target_pitch: PitchCurve,
    sr: int,
    final_mix: np.ndarray | None = None,
    original_mix: np.ndarray | None = None,
) -> QualityReport:
    """Run all quality evaluations and produce a report.

    Parameters
    ----------
    corrected_audio : np.ndarray
        User vocals after pitch correction (before time alignment).
    aligned_audio : np.ndarray
        User vocals after pitch correction AND time alignment.
    original_user_audio : np.ndarray
        Original unprocessed user vocals.
    ref_vocal_audio : np.ndarray
        Reference vocals from SUNO (separated).
    instrumental_audio : np.ndarray
        Instrumental mix from SUNO (separated).
    target_pitch : PitchCurve
        Target pitch curve from reference vocals.
    sr : int
        Sample rate.
    final_mix : np.ndarray or None
        The final output mix (vocals + instrumental), for balance eval.
    original_mix : np.ndarray or None
        The original SUNO song, for balance comparison.

    Returns
    -------
    QualityReport
    """
    report = QualityReport()

    # 1. Pitch accuracy (corrected vs target)
    pitch_grade = evaluate_pitch_accuracy(aligned_audio, sr, target_pitch)
    report.grades.append(pitch_grade)

    # 2. Timing accuracy (aligned vs reference onsets)
    timing_grade = evaluate_timing_accuracy(
        aligned_audio, sr, ref_vocal_audio, sr
    )
    report.grades.append(timing_grade)

    # 3. Vocal clarity (artifact detection on processed audio)
    clarity_grade = evaluate_vocal_clarity(aligned_audio, sr)
    report.grades.append(clarity_grade)

    # 4. Mix balance: compare final mix RMS against original mix RMS
    if final_mix is not None and original_mix is not None:
        balance_grade = evaluate_mix_balance_overall(final_mix, original_mix)
    else:
        balance_grade = evaluate_mix_balance(
            aligned_audio, instrumental_audio, ref_vocal_audio
        )
    report.grades.append(balance_grade)

    # Overall score (weighted average)
    grade_scores = {"GOOD": 100, "OK": 65, "POOR": 25}
    weights = {
        "Pitch Accuracy": 0.35,
        "Timing Accuracy": 0.25,
        "Vocal Clarity": 0.20,
        "Mix Balance": 0.20,
    }

    total = 0.0
    for g in report.grades:
        w = weights.get(g.name, 0.25)
        total += grade_scores[g.grade] * w
    report.overall_score = total

    return report
