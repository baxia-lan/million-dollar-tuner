"""Vocal replacement pipeline: the core orchestrator with iterative quality refinement.

The pipeline evaluates its own output and automatically adjusts parameters
until the result meets quality thresholds, or max iterations are reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np

from mdt.analysis.key import KeyResult, detect_key
from mdt.analysis.pitch import PitchCurve, detect_pitch
from mdt.analysis.rhythm import RhythmProfile, analyze_rhythm
from mdt.audio.io import load_audio, save_audio
from mdt.config import (
    ANALYSIS_SR,
    DEFAULT_CORRECTION_STRENGTH,
    MAX_PITCH_SHIFT_SEMITONES,
    OUTPUT_SR,
)
from mdt.evaluation import QualityReport, evaluate_full
from mdt.separation.models import StemResult
from mdt.separation.separator import separate
from mdt.tuning.mixer import combine_stems, mix_vocals_with_instrumental
from mdt.tuning.pitch_correct import correct_pitch
from mdt.tuning.time_align import align_vocals

# Maximum number of refinement iterations
MAX_ITERATIONS = 4
# Minimum acceptable overall score to pass
MIN_PASS_SCORE = 60.0


@dataclass
class TuneResult:
    """Result of the vocal replacement pipeline."""

    output_path: Path
    stems: StemResult
    ref_key: KeyResult
    ref_tempo: float
    ref_pitch: PitchCurve
    user_pitch: PitchCurve
    quality_report: QualityReport | None = None
    iterations_used: int = 1
    params_history: list[dict] = field(default_factory=list)


@dataclass
class _TuneParams:
    """Mutable parameters that the iteration loop can adjust."""

    correction_strength: float = 0.8
    max_shift: float = 4.0
    reverb_room: float = 0.3
    reverb_wet: float = 0.15
    vocal_gain_db: float = 0.0
    apply_effects: bool = True

    def to_dict(self) -> dict:
        return {
            "correction_strength": self.correction_strength,
            "max_shift": self.max_shift,
            "reverb_room": self.reverb_room,
            "reverb_wet": self.reverb_wet,
            "vocal_gain_db": self.vocal_gain_db,
        }


def run_vocal_replacement(
    user_vocals_path: str | Path,
    suno_song_path: str | Path,
    output_path: str | Path = "output.wav",
    correction_strength: float = DEFAULT_CORRECTION_STRENGTH,
    max_shift: float = MAX_PITCH_SHIFT_SEMITONES,
    apply_effects: bool = True,
    reverb_room: float = 0.3,
    reverb_wet: float = 0.15,
    vocal_gain_db: float = 0.0,
    demucs_model: str = "htdemucs_ft",
    device: str | None = None,
    stems_dir: str | Path | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> TuneResult:
    """Run the full vocal replacement pipeline with iterative quality refinement.

    The pipeline will:
    1. Separate SUNO song into stems.
    2. Analyze reference vocals.
    3. Tune + align + mix.
    4. **Evaluate quality** against defined thresholds.
    5. If quality is below standard, **automatically adjust parameters and retry**.
    6. Repeat until quality passes or max iterations reached.
    7. Output the best result across all iterations.
    """
    user_vocals_path = Path(user_vocals_path)
    suno_song_path = Path(suno_song_path)
    output_path = Path(output_path)

    if stems_dir is None:
        stems_dir = output_path.parent / "stems"
    stems_dir = Path(stems_dir)

    # ── Step 1: Separate SUNO song (only once) ─────────────────
    click.echo("\n[Step 1/6] Separating SUNO song into stems...")
    stems = separate(
        audio_path=suno_song_path,
        output_dir=stems_dir,
        model_name=demucs_model,
        device=device,
    )
    click.echo("  Done.")

    # ── Step 2: Analyze reference vocals (only once) ───────────
    click.echo("\n[Step 2/6] Analyzing reference vocals...")
    ref_vocals, ref_sr = load_audio(stems.vocals_path, sr=ANALYSIS_SR, mono=True)
    ref_pitch = detect_pitch(ref_vocals, sr=ANALYSIS_SR)
    ref_rhythm = analyze_rhythm(ref_vocals, sr=ANALYSIS_SR)
    ref_key = detect_key(ref_vocals, sr=ANALYSIS_SR)

    voiced_pct = np.sum(ref_pitch.voiced_flag) / len(ref_pitch.voiced_flag) * 100
    click.echo(f"  Key: {ref_key}")
    click.echo(f"  Tempo: {ref_rhythm.tempo:.1f} BPM")
    click.echo(f"  Voiced frames: {voiced_pct:.1f}%")

    # Load audio at various sample rates (only once)
    user_audio, _ = load_audio(user_vocals_path, sr=ANALYSIS_SR, mono=True)
    user_pitch = detect_pitch(user_audio, sr=ANALYSIS_SR)
    user_audio_hq, _ = load_audio(user_vocals_path, sr=OUTPUT_SR, mono=True)
    ref_vocals_hq, _ = load_audio(stems.vocals_path, sr=OUTPUT_SR, mono=True)
    instrumental, inst_sr = combine_stems(stems.instrumental_paths, sr=OUTPUT_SR)

    # ── Iterative refinement loop ──────────────────────────────
    params = _TuneParams(
        correction_strength=correction_strength,
        max_shift=max_shift,
        reverb_room=reverb_room,
        reverb_wet=reverb_wet,
        vocal_gain_db=vocal_gain_db,
        apply_effects=apply_effects,
    )

    best_report: QualityReport | None = None
    best_mix: np.ndarray | None = None
    best_sr: int = OUTPUT_SR
    best_params: dict = params.to_dict()
    params_history: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        if iteration == 1:
            click.echo(f"\n[Step 3/6] Processing vocals (attempt {iteration})...")
        else:
            click.echo(f"\n[Iteration {iteration}/{max_iterations}] "
                       f"Refining with adjusted parameters...")
            click.echo(f"  strength={params.correction_strength:.2f}  "
                       f"max_shift={params.max_shift:.1f}  "
                       f"gain={params.vocal_gain_db:+.1f}dB")

        params_history.append(params.to_dict())

        # Step 3: Pitch correct
        tuned_vocals = correct_pitch(
            y=user_audio_hq,
            sr=OUTPUT_SR,
            user_pitch=user_pitch,
            target_pitch=ref_pitch,
            strength=params.correction_strength,
            max_shift=params.max_shift,
        )

        # Step 4: Time align
        aligned_vocals = align_vocals(
            user_audio=tuned_vocals,
            user_sr=OUTPUT_SR,
            ref_audio=ref_vocals_hq,
            ref_sr=OUTPUT_SR,
        )

        # Step 5: Mix
        final_mix, final_sr = mix_vocals_with_instrumental(
            vocals=aligned_vocals,
            vocals_sr=OUTPUT_SR,
            instrumental=instrumental,
            instrumental_sr=inst_sr,
            reference_vocals=ref_vocals_hq,
            apply_fx=params.apply_effects,
            reverb_room=params.reverb_room,
            reverb_wet=params.reverb_wet,
            vocal_gain_db=params.vocal_gain_db,
        )

        # Step 6: Evaluate quality
        eval_len = min(len(aligned_vocals), len(ref_vocals_hq), len(user_audio_hq))
        instrumental_mono = instrumental
        if instrumental_mono.ndim == 2:
            instrumental_mono = np.mean(instrumental_mono, axis=0)

        report = evaluate_full(
            corrected_audio=tuned_vocals[:eval_len],
            aligned_audio=aligned_vocals[:eval_len],
            original_user_audio=user_audio_hq[:eval_len],
            ref_vocal_audio=ref_vocals_hq[:eval_len],
            instrumental_audio=instrumental_mono[:min(eval_len, len(instrumental_mono))],
            target_pitch=ref_pitch,
            sr=OUTPUT_SR,
        )

        click.echo(str(report))

        # Track best result
        if best_report is None or report.overall_score > best_report.overall_score:
            best_report = report
            best_mix = final_mix
            best_sr = final_sr
            best_params = params.to_dict()

        # Check if quality is acceptable
        if report.passed and report.overall_score >= MIN_PASS_SCORE:
            click.echo(f"\n  Quality PASSED on iteration {iteration}.")
            break

        # If this is the last iteration, don't try to adjust
        if iteration >= max_iterations:
            click.echo(f"\n  Max iterations ({max_iterations}) reached. "
                       f"Using best result (score: {best_report.overall_score:.0f}).")
            break

        # ── Adjust parameters based on what failed ─────────────
        click.echo(f"\n  Quality below target. Auto-adjusting parameters...")
        params = _adjust_params(params, report)

    # ── Save the best result ───────────────────────────────────
    save_audio(output_path, best_mix, best_sr)
    click.echo(f"\n  Final output saved: {output_path}")
    click.echo(f"  Best parameters: {best_params}")
    click.echo("\nDone! Your tuned song is ready.")

    return TuneResult(
        output_path=output_path,
        stems=stems,
        ref_key=ref_key,
        ref_tempo=ref_rhythm.tempo,
        ref_pitch=ref_pitch,
        user_pitch=user_pitch,
        quality_report=best_report,
        iterations_used=min(iteration, max_iterations),
        params_history=params_history,
    )


def _adjust_params(params: _TuneParams, report: QualityReport) -> _TuneParams:
    """Adjust processing parameters based on quality evaluation results.

    Strategy for each failing metric:

    Pitch POOR:
      - Increase correction strength (push closer to target pitch)
      - Increase max allowed shift

    Timing POOR:
      - Not much we can adjust algorithmically; DTW is already best-effort
      - Slight adjustment: keep same params (alignment is deterministic)

    Spectral Quality POOR (too many artifacts):
      - REDUCE correction strength (less processing = fewer artifacts)
      - Reduce max shift (avoid extreme pitch shifts that cause distortion)

    Mix Balance POOR:
      - Adjust vocal gain to compensate

    The key insight: pitch accuracy and spectral quality are in tension.
    More correction fixes pitch but adds artifacts. The loop finds the
    sweet spot automatically.
    """
    new = _TuneParams(
        correction_strength=params.correction_strength,
        max_shift=params.max_shift,
        reverb_room=params.reverb_room,
        reverb_wet=params.reverb_wet,
        vocal_gain_db=params.vocal_gain_db,
        apply_effects=params.apply_effects,
    )

    for grade in report.grades:
        if grade.grade == "POOR":
            if grade.name == "Pitch Accuracy":
                # Pitch is off → push correction harder
                new.correction_strength = min(1.0, params.correction_strength + 0.10)
                new.max_shift = min(6.0, params.max_shift + 1.0)

            elif grade.name == "Spectral Quality":
                # Too many artifacts → ease off processing
                new.correction_strength = max(0.3, params.correction_strength - 0.15)
                new.max_shift = max(2.0, params.max_shift - 0.5)

            elif grade.name == "Mix Balance":
                # Volume mismatch → adjust gain
                # Use the detail field to figure out direction
                if "ratio" in grade.detail:
                    # If user vocal is too quiet relative to reference
                    new.vocal_gain_db = params.vocal_gain_db + 2.0

        elif grade.grade == "OK":
            # "OK" metrics: make small adjustments toward "GOOD"
            if grade.name == "Pitch Accuracy":
                new.correction_strength = min(1.0, params.correction_strength + 0.05)

            elif grade.name == "Spectral Quality":
                new.correction_strength = max(0.3, params.correction_strength - 0.05)

    # Sanity-check: if pitch and spectral are fighting each other,
    # prioritize pitch (the main user complaint is being off-key)
    if any(g.name == "Pitch Accuracy" and g.grade == "POOR" for g in report.grades):
        if any(g.name == "Spectral Quality" and g.grade == "POOR" for g in report.grades):
            # Both are POOR: favor moderate correction
            new.correction_strength = 0.7
            new.max_shift = 3.5

    return new
