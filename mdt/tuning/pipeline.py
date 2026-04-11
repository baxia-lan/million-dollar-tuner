"""Vocal replacement pipeline: extract user timbre, apply to SUNO vocals.

The correct approach: keep SUNO's pitch/timing/rhythm exactly as-is,
only replace the vocal timbre (tone color) with the user's voice.

This avoids all the problems of pitch correction + time alignment
(artifacts, timing mismatches, distortion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np

from mdt.analysis.key import KeyResult, detect_key
from mdt.analysis.pitch import PitchCurve, detect_pitch
from mdt.analysis.rhythm import analyze_rhythm
from mdt.audio.io import load_audio, save_audio
from mdt.config import ANALYSIS_SR, OUTPUT_SR
from mdt.evaluation import QualityReport, evaluate_full
from mdt.separation.models import StemResult
from mdt.separation.separator import separate
from mdt.tuning.mixer import combine_stems, mix_vocals_with_instrumental
from mdt.tuning.voice_convert import convert_voice_timbre

MAX_ITERATIONS = 4
MIN_PASS_SCORE = 60.0


@dataclass
class TuneResult:
    """Result of the vocal replacement pipeline."""

    output_path: Path
    stems: StemResult
    ref_key: KeyResult
    ref_tempo: float
    quality_report: QualityReport | None = None
    iterations_used: int = 1
    params_history: list[dict] = field(default_factory=list)


@dataclass
class _TuneParams:
    """Mutable parameters that the iteration loop can adjust."""

    blend: float = 0.8          # Timbre blend: 0=original, 1=full user timbre
    n_cepstral: int = 60        # Envelope smoothness (lower=more timbre transfer)
    reverb_room: float = 0.3
    reverb_wet: float = 0.15
    vocal_gain_db: float = 0.0
    apply_effects: bool = True

    def to_dict(self) -> dict:
        return {
            "blend": self.blend,
            "n_cepstral": self.n_cepstral,
            "reverb_room": self.reverb_room,
            "reverb_wet": self.reverb_wet,
            "vocal_gain_db": self.vocal_gain_db,
        }


def run_vocal_replacement(
    user_vocals_path: str | Path,
    suno_song_path: str | Path,
    output_path: str | Path = "output.wav",
    blend: float = 0.8,
    apply_effects: bool = True,
    reverb_room: float = 0.3,
    reverb_wet: float = 0.15,
    vocal_gain_db: float = 0.0,
    demucs_model: str = "htdemucs_ft",
    device: str | None = None,
    stems_dir: str | Path | None = None,
    max_iterations: int = MAX_ITERATIONS,
    # Legacy params (ignored, kept for CLI compatibility)
    correction_strength: float = 0.8,
    max_shift: float = 4.0,
) -> TuneResult:
    """Run the vocal replacement pipeline with voice timbre transfer.

    Steps:
    1. Separate SUNO song into stems (vocals + instrumentals).
    2. Analyze reference vocals (key, tempo).
    3. Extract user's voice timbre from their recording.
    4. Apply user's timbre to SUNO vocals (keeps pitch/timing intact).
    5. Mix converted vocals with separated instrumentals.
    6. Evaluate quality; iterate if needed.
    """
    user_vocals_path = Path(user_vocals_path)
    suno_song_path = Path(suno_song_path)
    output_path = Path(output_path)

    if stems_dir is None:
        stems_dir = output_path.parent / "stems"
    stems_dir = Path(stems_dir)

    # ── Step 1: Separate SUNO song (only once) ─────────────────
    click.echo("\n[Step 1/5] Separating SUNO song into stems...")
    stems = separate(
        audio_path=suno_song_path,
        output_dir=stems_dir,
        model_name=demucs_model,
        device=device,
    )
    click.echo("  Done.")

    # ── Step 2: Analyze reference vocals (only once) ───────────
    click.echo("\n[Step 2/5] Analyzing reference vocals...")
    ref_vocals_analysis, _ = load_audio(stems.vocals_path, sr=ANALYSIS_SR, mono=True)
    ref_rhythm = analyze_rhythm(ref_vocals_analysis, sr=ANALYSIS_SR)
    ref_key = detect_key(ref_vocals_analysis, sr=ANALYSIS_SR)
    ref_pitch = detect_pitch(ref_vocals_analysis, sr=ANALYSIS_SR)

    voiced_pct = np.sum(ref_pitch.voiced_flag) / len(ref_pitch.voiced_flag) * 100
    click.echo(f"  Key: {ref_key}")
    click.echo(f"  Tempo: {ref_rhythm.tempo:.1f} BPM")
    click.echo(f"  Voiced frames: {voiced_pct:.1f}%")

    # Load audio at output quality (only once)
    user_audio_hq, _ = load_audio(user_vocals_path, sr=OUTPUT_SR, mono=True)
    ref_vocals_hq, _ = load_audio(stems.vocals_path, sr=OUTPUT_SR, mono=True)
    instrumental, inst_sr = combine_stems(stems.instrumental_paths, sr=OUTPUT_SR)
    original_mix_hq, _ = load_audio(suno_song_path, sr=OUTPUT_SR, mono=True)

    click.echo(f"\n  User voice sample: {len(user_audio_hq)/OUTPUT_SR:.1f}s")

    # ── Iterative refinement loop ──────────────────────────────
    params = _TuneParams(
        blend=blend,
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
            click.echo(f"\n[Step 3/5] Transferring your voice timbre...")
        else:
            click.echo(f"\n[Iteration {iteration}/{max_iterations}] "
                       f"Refining...")
            click.echo(f"  blend={params.blend:.2f}  "
                       f"cepstral={params.n_cepstral}  "
                       f"gain={params.vocal_gain_db:+.1f}dB")

        params_history.append(params.to_dict())

        # Step 3: Voice timbre conversion
        converted_vocals = convert_voice_timbre(
            suno_vocals=ref_vocals_hq,
            user_audio=user_audio_hq,
            sr=OUTPUT_SR,
            n_cepstral=params.n_cepstral,
            blend=params.blend,
        )

        # Step 4: Mix
        final_mix, final_sr = mix_vocals_with_instrumental(
            vocals=converted_vocals,
            vocals_sr=OUTPUT_SR,
            instrumental=instrumental,
            instrumental_sr=inst_sr,
            reference_vocals=ref_vocals_hq,
            apply_fx=params.apply_effects,
            reverb_room=params.reverb_room,
            reverb_wet=params.reverb_wet,
            vocal_gain_db=params.vocal_gain_db,
        )

        # Step 5: Evaluate quality
        eval_len = min(len(converted_vocals), len(ref_vocals_hq))

        instrumental_mono = instrumental
        if instrumental_mono.ndim == 2:
            instrumental_mono = np.mean(instrumental_mono, axis=0)
        final_mix_mono = final_mix
        if final_mix_mono.ndim == 2:
            final_mix_mono = np.mean(final_mix_mono, axis=0)

        report = evaluate_full(
            corrected_audio=converted_vocals[:eval_len],
            aligned_audio=converted_vocals[:eval_len],
            original_user_audio=user_audio_hq[:min(eval_len, len(user_audio_hq))],
            ref_vocal_audio=ref_vocals_hq[:eval_len],
            instrumental_audio=instrumental_mono[:min(eval_len, len(instrumental_mono))],
            final_mix=final_mix_mono,
            original_mix=original_mix_hq,
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

        if iteration >= max_iterations:
            click.echo(f"\n  Max iterations ({max_iterations}) reached. "
                       f"Using best result (score: {best_report.overall_score:.0f}).")
            break

        # Adjust parameters
        click.echo(f"\n  Quality below target. Auto-adjusting...")
        params = _adjust_params(params, report)

    # ── Save the best result ───────────────────────────────────
    save_audio(output_path, best_mix, best_sr)
    click.echo(f"\n  Final output saved: {output_path}")
    click.echo(f"  Best parameters: {best_params}")
    click.echo("\nDone! Your song is ready.")

    return TuneResult(
        output_path=output_path,
        stems=stems,
        ref_key=ref_key,
        ref_tempo=ref_rhythm.tempo,
        quality_report=best_report,
        iterations_used=min(iteration, max_iterations),
        params_history=params_history,
    )


def _adjust_params(params: _TuneParams, report: QualityReport) -> _TuneParams:
    """Adjust parameters based on quality evaluation."""
    new = _TuneParams(
        blend=params.blend,
        n_cepstral=params.n_cepstral,
        reverb_room=params.reverb_room,
        reverb_wet=params.reverb_wet,
        vocal_gain_db=params.vocal_gain_db,
        apply_effects=params.apply_effects,
    )

    for grade in report.grades:
        if grade.grade == "POOR":
            if grade.name == "Pitch Accuracy":
                # Pitch should be preserved from SUNO; if off, reduce blend
                # (less timbre transfer = less distortion of harmonics)
                new.blend = max(0.3, params.blend - 0.15)
                new.n_cepstral = min(80, params.n_cepstral + 10)

            elif grade.name == "Vocal Clarity":
                # Artifacts from envelope transfer; use smoother envelope
                new.n_cepstral = min(80, params.n_cepstral + 10)
                new.blend = max(0.4, params.blend - 0.1)

            elif grade.name == "Mix Balance":
                try:
                    parts = grade.detail.split(":")
                    output_rms = float(parts[1].split("dB")[0].strip())
                    orig_rms = float(parts[2].split("dB")[0].strip())
                    if output_rms < orig_rms:
                        new.vocal_gain_db = params.vocal_gain_db + 3.0
                    else:
                        new.vocal_gain_db = params.vocal_gain_db - 3.0
                except (IndexError, ValueError):
                    new.vocal_gain_db = params.vocal_gain_db + 2.0

        elif grade.grade == "OK":
            if grade.name == "Vocal Clarity":
                new.n_cepstral = min(80, params.n_cepstral + 5)

    return new
