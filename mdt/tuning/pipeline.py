"""Vocal replacement pipeline: extract user timbre, apply to SUNO vocals.

Keeps SUNO's pitch/timing/rhythm exactly as-is.
Only replaces the vocal timbre (tone color) with the user's voice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np

from mdt.analysis.key import KeyResult, detect_key
from mdt.analysis.pitch import detect_pitch
from mdt.analysis.rhythm import analyze_rhythm
from mdt.audio.io import load_audio, save_audio
from mdt.audio.mastering import master_audio
from mdt.config import ANALYSIS_SR, OUTPUT_SR
from mdt.separation.models import StemResult
from mdt.separation.separator import separate
from mdt.tuning.mixer import combine_stems, mix_vocals_with_instrumental
from mdt.tuning.voice_convert import convert_voice_timbre


@dataclass
class TuneResult:
    """Result of the vocal replacement pipeline."""

    output_path: Path
    stems: StemResult
    ref_key: KeyResult
    ref_tempo: float


def run_vocal_replacement(
    user_vocals_path: str | Path,
    suno_song_path: str | Path,
    output_path: str | Path = "output.wav",
    voice_model_dir: str | Path | None = None,
    apply_effects: bool = True,
    reverb_room: float = 0.3,
    reverb_wet: float = 0.15,
    vocal_gain_db: float = 0.0,
    demucs_model: str = "htdemucs_ft",
    device: str | None = None,
    stems_dir: str | Path | None = None,
    # Legacy params (ignored, kept for CLI compatibility)
    max_iterations: int = 1,
    correction_strength: float = 0.8,
    max_shift: float = 4.0,
) -> TuneResult:
    """Run the vocal replacement pipeline.

    Steps:
    1. Separate SUNO song into stems (Demucs).
    2. Analyze reference vocals.
    3. Transfer user's voice timbre onto SUNO vocals.
    4. Master and mix with instrumental.
    """
    user_vocals_path = Path(user_vocals_path)
    suno_song_path = Path(suno_song_path)
    output_path = Path(output_path)

    if stems_dir is None:
        stems_dir = output_path.parent / "stems"
    stems_dir = Path(stems_dir)

    # ── Step 1: Separate ───────────────────────────────────────
    click.echo("\n[Step 1/4] Separating SUNO song into stems...")
    stems = separate(
        audio_path=suno_song_path,
        output_dir=stems_dir,
        model_name=demucs_model,
        device=device,
    )
    click.echo("  Done.")

    # ── Step 2: Analyze ────────────────────────────────────────
    click.echo("\n[Step 2/4] Analyzing reference vocals...")
    ref_vocals_analysis, _ = load_audio(stems.vocals_path, sr=ANALYSIS_SR, mono=True)
    ref_rhythm = analyze_rhythm(ref_vocals_analysis, sr=ANALYSIS_SR)
    ref_key = detect_key(ref_vocals_analysis, sr=ANALYSIS_SR)

    click.echo(f"  Key: {ref_key}")
    click.echo(f"  Tempo: {ref_rhythm.tempo:.1f} BPM")

    # Load audio at output quality
    user_audio_hq, _ = load_audio(user_vocals_path, sr=OUTPUT_SR, mono=True)
    ref_vocals_hq, _ = load_audio(stems.vocals_path, sr=OUTPUT_SR, mono=True)
    instrumental, inst_sr = combine_stems(stems.instrumental_paths, sr=OUTPUT_SR)

    click.echo(f"  User voice sample: {len(user_audio_hq)/OUTPUT_SR:.1f}s")

    # ── Step 3: Voice timbre transfer ──────────────────────────
    click.echo(f"\n[Step 3/4] Replacing vocal timbre with your voice...")
    converted_vocals = convert_voice_timbre(
        suno_vocals=ref_vocals_hq,
        user_audio=user_audio_hq,
        sr=OUTPUT_SR,
        model_dir=voice_model_dir,
    )

    # ── Diagnostic: save intermediate files & compare ──────────
    import librosa as _lr
    diag_dir = stems_dir
    save_audio(diag_dir / "converted_vocals.wav", converted_vocals, OUTPUT_SR)
    click.echo(f"  Saved intermediate: {diag_dir / 'converted_vocals.wav'}")

    def _centroid(y):
        return float(np.mean(_lr.feature.spectral_centroid(y=y, sr=OUTPUT_SR)))

    def _mfcc(y):
        return np.mean(_lr.feature.mfcc(y=y, sr=OUTPUT_SR, n_mfcc=13)[1:], axis=1)

    def _cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))

    min_len = min(len(ref_vocals_hq), len(converted_vocals), len(user_audio_hq))
    m_ref = _mfcc(ref_vocals_hq[:min_len])
    m_out = _mfcc(converted_vocals[:min_len])
    m_usr = _mfcc(user_audio_hq[:min_len])

    click.echo(f"  Centroid: SUNO={_centroid(ref_vocals_hq):.0f} "
               f"→ converted={_centroid(converted_vocals):.0f} "
               f"(user={_centroid(user_audio_hq):.0f})")
    click.echo(f"  MFCC sim: to_SUNO={_cos(m_ref, m_out):.3f} "
               f"to_USER={_cos(m_usr, m_out):.3f}")
    click.echo(f"  RMS: SUNO={np.sqrt(np.mean(ref_vocals_hq**2)):.4f} "
               f"converted={np.sqrt(np.mean(converted_vocals**2)):.4f}")
    click.echo(f"\n  → Listen to {diag_dir / 'converted_vocals.wav'} to check timbre")
    click.echo(f"  → Compare with {stems_dir / 'vocals.wav'} (original SUNO vocal)")
    click.echo("  Done.")

    # ── Step 4: Master + Mix ───────────────────────────────────
    click.echo("\n[Step 4/4] Mastering and mixing...")

    # Master the instrumental
    instrumental, inst_sr = master_audio(instrumental, inst_sr, target_sr=OUTPUT_SR)

    # Mix
    final_mix, final_sr = mix_vocals_with_instrumental(
        vocals=converted_vocals,
        vocals_sr=OUTPUT_SR,
        instrumental=instrumental,
        instrumental_sr=inst_sr,
        reference_vocals=ref_vocals_hq,
        apply_fx=apply_effects,
        reverb_room=reverb_room,
        reverb_wet=reverb_wet,
        vocal_gain_db=vocal_gain_db,
    )

    # Final master
    final_mix, final_sr = master_audio(final_mix, final_sr, target_sr=OUTPUT_SR)

    save_audio(output_path, final_mix, final_sr)
    click.echo(f"\n  Output saved: {output_path}")
    click.echo("  Done!")

    return TuneResult(
        output_path=output_path,
        stems=stems,
        ref_key=ref_key,
        ref_tempo=ref_rhythm.tempo,
    )
