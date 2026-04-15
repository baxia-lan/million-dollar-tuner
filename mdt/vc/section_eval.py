"""Section-level evaluation for voice conversion quality.

Outputs per-section metrics instead of whole-song averages:
- rap_low_similarity: speaker similarity in rap/low-register sections
- melodic_similarity: speaker similarity in melodic sections
- melodic_pitch_corr: F0 correlation in melodic sections
- artifact_score: spectral discontinuity / noise metric
- longform_stability: consistency across the song (std of per-section similarity)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass
class SectionScore:
    label: str
    section_type: str
    start: float
    end: float
    sim_to_user: float = 0.0
    f0_correlation: float = 0.0
    f0_error_hz: float = 0.0
    artifact_score: float = 0.0  # lower = cleaner
    rms_ratio: float = 0.0  # output RMS / source RMS


@dataclass
class SectionReport:
    model_name: str
    train_params: dict = field(default_factory=dict)
    infer_params: dict = field(default_factory=dict)
    sections: list[SectionScore] = field(default_factory=list)
    # Aggregated
    rap_low_similarity: float = 0.0
    melodic_similarity: float = 0.0
    melodic_pitch_corr: float = 0.0
    overall_artifact: float = 0.0
    longform_stability: float = 0.0
    conclusion: str = ""


def load_sections(sections_path: str | Path) -> list[dict]:
    with open(sections_path) as f:
        data = json.load(f)
    return data["sections"]


def extract_segment(y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    s = int(start * sr)
    e = int(end * sr)
    return y[s:e]


def compute_artifact_score(y: np.ndarray, sr: int) -> float:
    """Estimate artifact level from spectral flatness variance and energy jumps.

    Lower = cleaner. Range roughly 0-1.
    """
    hop = 512
    # Spectral flatness: tonal signals have low flatness, noise has high
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    # High variance in flatness = inconsistent quality = artifacts
    flatness_var = float(np.std(flatness))

    # Energy jumps: sudden volume changes indicate glitches
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    if len(rms) > 1:
        jumps = np.abs(np.diff(rms))
        jump_score = float(np.percentile(jumps, 95) / (np.mean(rms) + 1e-10))
    else:
        jump_score = 0.0

    # Combine (both normalized roughly 0-1)
    return min(1.0, flatness_var * 2 + jump_score * 0.5)


def evaluate_sections(
    user_audio_path: str | Path,
    source_vocals_path: str | Path,
    converted_vocals_path: str | Path,
    sections_path: str | Path,
    model_name: str = "",
    train_params: dict | None = None,
    infer_params: dict | None = None,
    sr: int = 44100,
) -> SectionReport:
    """Run section-level evaluation."""
    from mdt.vc.metrics import load_mono, speaker_similarity

    import pyworld as pw

    y_user = load_mono(user_audio_path, sr)
    y_source = load_mono(source_vocals_path, sr)
    y_conv = load_mono(converted_vocals_path, sr)

    sections = load_sections(sections_path)
    scores: list[SectionScore] = []

    for sec in sections:
        start, end = sec["start"], sec["end"]
        label = sec.get("label", "unknown")
        stype = sec.get("type", "unknown")

        seg_conv = extract_segment(y_conv, sr, start, end)
        seg_source = extract_segment(y_source, sr, start, end)

        if len(seg_conv) < sr * 0.5:  # skip very short segments
            continue

        # Speaker similarity (use full user audio as reference, segment for converted)
        sim = speaker_similarity(seg_conv, y_user, sr)

        # F0 correlation for this segment
        f0_src = pw.harvest(seg_source.astype(np.float64), sr)[0]
        f0_conv = pw.harvest(seg_conv.astype(np.float64), sr)[0]
        n = min(len(f0_src), len(f0_conv))
        bv = (f0_src[:n] > 0) & (f0_conv[:n] > 0)
        if np.sum(bv) > 5:
            f0_corr = float(np.corrcoef(f0_src[:n][bv], f0_conv[:n][bv])[0, 1])
            f0_err = float(np.mean(np.abs(f0_src[:n][bv] - f0_conv[:n][bv])))
        else:
            f0_corr, f0_err = 0.0, 0.0

        # Artifact score
        artifact = compute_artifact_score(seg_conv, sr)

        # RMS ratio
        rms_src = np.sqrt(np.mean(seg_source**2)) + 1e-10
        rms_conv = np.sqrt(np.mean(seg_conv**2)) + 1e-10
        rms_ratio = float(rms_conv / rms_src)

        scores.append(SectionScore(
            label=label, section_type=stype,
            start=start, end=end,
            sim_to_user=sim, f0_correlation=f0_corr,
            f0_error_hz=f0_err, artifact_score=artifact,
            rms_ratio=rms_ratio,
        ))

    # Aggregate
    rap_scores = [s for s in scores if s.section_type == "rap"]
    mel_scores = [s for s in scores if s.section_type == "melodic"]
    all_sims = [s.sim_to_user for s in scores]

    report = SectionReport(
        model_name=model_name,
        train_params=train_params or {},
        infer_params=infer_params or {},
        sections=scores,
        rap_low_similarity=np.mean([s.sim_to_user for s in rap_scores]) if rap_scores else 0.0,
        melodic_similarity=np.mean([s.sim_to_user for s in mel_scores]) if mel_scores else 0.0,
        melodic_pitch_corr=np.mean([s.f0_correlation for s in mel_scores]) if mel_scores else 0.0,
        overall_artifact=np.mean([s.artifact_score for s in scores]) if scores else 0.0,
        longform_stability=1.0 - (np.std(all_sims) if len(all_sims) > 1 else 0.0),
    )

    return report


def print_report(report: SectionReport):
    """Print a formatted section-level report."""
    print(f"\n{'='*70}")
    print(f"  Model: {report.model_name}")
    if report.train_params:
        print(f"  Train: {report.train_params}")
    if report.infer_params:
        print(f"  Infer: {report.infer_params}")
    print(f"{'='*70}")

    print(f"\n  {'Section':<20} {'Type':<10} {'→User':>7} {'F0corr':>7} {'F0err':>7} {'Artif':>7}")
    print(f"  {'-'*58}")
    for s in report.sections:
        print(f"  {s.label:<20} {s.section_type:<10} {s.sim_to_user:>7.3f} "
              f"{s.f0_correlation:>7.3f} {s.f0_error_hz:>6.1f}Hz {s.artifact_score:>7.3f}")

    print(f"\n  {'─'*40}")
    print(f"  rap_low_similarity:   {report.rap_low_similarity:.4f}")
    print(f"  melodic_similarity:   {report.melodic_similarity:.4f}")
    print(f"  melodic_pitch_corr:   {report.melodic_pitch_corr:.4f}")
    print(f"  overall_artifact:     {report.overall_artifact:.4f}")
    print(f"  longform_stability:   {report.longform_stability:.4f}")
    if report.conclusion:
        print(f"\n  Conclusion: {report.conclusion}")


def save_report(report: SectionReport, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
