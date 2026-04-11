"""Quality evaluation metrics for vocal replacement results.

Defines measurable criteria to assess whether the output is good enough:

1. Pitch Accuracy  — 修正后的音高与目标的偏差 (cents)
2. Time Alignment  — 节奏对齐的精度 (onset误差 ms)
3. Spectral Quality — 频谱失真程度 (信噪比)
4. Mix Balance     — 人声与伴奏的音量平衡 (dB差)
5. Overall Score   — 综合评分 (0-100)

Quality thresholds (通过/警告/失败):
- Pitch: <25 cents = GOOD, 25-50 = OK, >50 = POOR
- Timing: <30ms = GOOD, 30-80ms = OK, >80ms = POOR
- SNR: >15dB = GOOD, 10-15 = OK, <10 = POOR
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mdt.analysis.pitch import PitchCurve, detect_pitch, pitch_difference_semitones
from mdt.analysis.rhythm import analyze_rhythm
from mdt.config import ANALYSIS_SR


# ── Thresholds ──────────────────────────────────────────────────

# Pitch deviation in cents (100 cents = 1 semitone)
PITCH_GOOD = 25.0     # < 25 cents = excellent
PITCH_OK = 50.0        # 25–50 cents = acceptable
                       # > 50 cents = poor

# Onset alignment error in milliseconds
TIMING_GOOD = 30.0     # < 30ms = inaudible
TIMING_OK = 80.0       # 30–80ms = noticeable but ok
                       # > 80ms = poor

# Signal-to-noise ratio in dB
SNR_GOOD = 15.0        # > 15 dB = clean
SNR_OK = 10.0          # 10–15 dB = acceptable
                       # < 10 dB = noisy

# Mix balance: vocal RMS vs instrumental RMS difference in dB
BALANCE_GOOD = 3.0     # < 3 dB difference from reference = good
BALANCE_OK = 6.0       # 3–6 dB = acceptable


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
            grade="POOR",
            detail="Not enough voiced frames to evaluate",
        )

    # Deviation in cents (100 cents = 1 semitone)
    diff_semitones = pitch_difference_semitones(
        corrected_pitch.frequencies[both_voiced],
        target_f0_interp[both_voiced],
    )
    deviation_cents = np.abs(diff_semitones) * 100  # semitones -> cents
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

    For each onset in the reference, finds the nearest onset
    in the aligned audio and measures the error.
    """
    ref_rhythm = analyze_rhythm(ref_audio, sr=ref_sr)
    aligned_rhythm = analyze_rhythm(aligned_audio, sr=aligned_sr)

    ref_onsets = ref_rhythm.onset_times
    aligned_onsets = aligned_rhythm.onset_times

    if len(ref_onsets) < 3 or len(aligned_onsets) < 3:
        return QualityGrade(
            name="Timing Accuracy",
            value=0.0,
            unit="ms",
            grade="OK",
            detail="Not enough onsets to evaluate",
        )

    # For each reference onset, find the nearest aligned onset
    errors_ms = []
    for ref_t in ref_onsets:
        if len(aligned_onsets) == 0:
            break
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
        detail=f"Median nearest-onset error over {len(errors_ms)} onsets",
    )


def evaluate_spectral_quality(
    original_audio: np.ndarray,
    processed_audio: np.ndarray,
) -> QualityGrade:
    """Measure spectral distortion as signal-to-noise ratio.

    Treats the original vocal as the signal and the difference
    (processed - original) as noise/artifacts.
    """
    # Match lengths
    min_len = min(len(original_audio), len(processed_audio))
    orig = original_audio[:min_len]
    proc = processed_audio[:min_len]

    # SNR = 10 * log10(signal_power / noise_power)
    signal_power = np.mean(orig ** 2)
    noise_power = np.mean((proc - orig) ** 2)

    if noise_power < 1e-10:
        snr = 60.0  # Essentially no difference
    elif signal_power < 1e-10:
        snr = 0.0
    else:
        snr = 10.0 * np.log10(signal_power / noise_power)

    if snr > SNR_GOOD:
        grade = "GOOD"
    elif snr > SNR_OK:
        grade = "OK"
    else:
        grade = "POOR"

    return QualityGrade(
        name="Spectral Quality",
        value=snr,
        unit="dB SNR",
        grade=grade,
        detail="Signal-to-noise ratio (original vs processed difference)",
    )


def evaluate_mix_balance(
    vocal_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    ref_vocal_audio: np.ndarray,
    ref_instrumental_audio: np.ndarray | None = None,
) -> QualityGrade:
    """Check if the vocal/instrumental balance matches the original.

    Compares the vocal-to-instrumental RMS ratio against
    the reference ratio.
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


def evaluate_full(
    corrected_audio: np.ndarray,
    aligned_audio: np.ndarray,
    original_user_audio: np.ndarray,
    ref_vocal_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    target_pitch: PitchCurve,
    sr: int,
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

    Returns
    -------
    QualityReport
    """
    report = QualityReport()

    # 1. Pitch accuracy
    pitch_grade = evaluate_pitch_accuracy(aligned_audio, sr, target_pitch)
    report.grades.append(pitch_grade)

    # 2. Timing accuracy
    timing_grade = evaluate_timing_accuracy(
        aligned_audio, sr, ref_vocal_audio, sr
    )
    report.grades.append(timing_grade)

    # 3. Spectral quality (artifacts from pitch/time manipulation)
    spectral_grade = evaluate_spectral_quality(original_user_audio, aligned_audio)
    report.grades.append(spectral_grade)

    # 4. Mix balance
    balance_grade = evaluate_mix_balance(
        aligned_audio, instrumental_audio, ref_vocal_audio
    )
    report.grades.append(balance_grade)

    # Overall score (weighted average)
    grade_scores = {"GOOD": 100, "OK": 65, "POOR": 25}
    weights = {
        "Pitch Accuracy": 0.35,
        "Timing Accuracy": 0.25,
        "Spectral Quality": 0.20,
        "Mix Balance": 0.20,
    }

    total = 0.0
    for g in report.grades:
        w = weights.get(g.name, 0.25)
        total += grade_scores[g.grade] * w
    report.overall_score = total

    return report
