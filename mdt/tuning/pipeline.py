"""Vocal replacement pipeline: the core orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
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
from mdt.separation.models import StemResult
from mdt.separation.separator import separate
from mdt.tuning.mixer import combine_stems, mix_vocals_with_instrumental
from mdt.tuning.pitch_correct import correct_pitch
from mdt.tuning.time_align import align_vocals


@dataclass
class TuneResult:
    """Result of the vocal replacement pipeline."""

    output_path: Path
    stems: StemResult
    ref_key: KeyResult
    ref_tempo: float
    ref_pitch: PitchCurve
    user_pitch: PitchCurve


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
) -> TuneResult:
    """Run the full vocal replacement pipeline.

    Steps:
    1. Separate SUNO song into stems (vocals + instrumentals).
    2. Analyze reference vocals (pitch, rhythm, key).
    3. Auto-tune user's vocals to match reference pitch.
    4. Time-align user's vocals to match reference timing.
    5. Mix tuned vocals with separated instrumentals.

    Parameters
    ----------
    user_vocals_path : path
        Path to user's raw singing recording.
    suno_song_path : path
        Path to the SUNO AI-generated song.
    output_path : path
        Where to save the final mixed output.
    correction_strength : float
        Pitch correction strength [0.0, 1.0].
    max_shift : float
        Maximum pitch shift in semitones.
    apply_effects : bool
        Whether to apply vocal effects (reverb, compression, EQ).
    reverb_room, reverb_wet : float
        Reverb parameters.
    vocal_gain_db : float
        Additional vocal gain in dB.
    demucs_model : str
        Demucs model name.
    device : str or None
        Torch device.
    stems_dir : path or None
        Directory for stems. None uses a temp directory.

    Returns
    -------
    TuneResult
    """
    user_vocals_path = Path(user_vocals_path)
    suno_song_path = Path(suno_song_path)
    output_path = Path(output_path)

    if stems_dir is None:
        stems_dir = output_path.parent / "stems"
    stems_dir = Path(stems_dir)

    # ── Step 1: Separate SUNO song ─────────────────────────────
    click.echo("\n[Step 1/5] Separating SUNO song into stems...")
    stems = separate(
        audio_path=suno_song_path,
        output_dir=stems_dir,
        model_name=demucs_model,
        device=device,
    )
    click.echo("  Done.")

    # ── Step 2: Analyze reference vocals ───────────────────────
    click.echo("\n[Step 2/5] Analyzing reference vocals...")
    ref_vocals, ref_sr = load_audio(stems.vocals_path, sr=ANALYSIS_SR, mono=True)

    ref_pitch = detect_pitch(ref_vocals, sr=ANALYSIS_SR)
    ref_rhythm = analyze_rhythm(ref_vocals, sr=ANALYSIS_SR)
    ref_key = detect_key(ref_vocals, sr=ANALYSIS_SR)

    voiced_pct = np.sum(ref_pitch.voiced_flag) / len(ref_pitch.voiced_flag) * 100
    click.echo(f"  Key: {ref_key}")
    click.echo(f"  Tempo: {ref_rhythm.tempo:.1f} BPM")
    click.echo(f"  Voiced frames: {voiced_pct:.1f}%")

    # ── Step 3: Pitch-correct user vocals ──────────────────────
    click.echo("\n[Step 3/5] Auto-tuning your vocals...")
    user_audio, user_sr = load_audio(user_vocals_path, sr=ANALYSIS_SR, mono=True)
    user_pitch = detect_pitch(user_audio, sr=ANALYSIS_SR)

    # Also load at output SR for high-quality processing
    user_audio_hq, _ = load_audio(user_vocals_path, sr=OUTPUT_SR, mono=True)

    tuned_vocals = correct_pitch(
        y=user_audio_hq,
        sr=OUTPUT_SR,
        user_pitch=user_pitch,
        target_pitch=ref_pitch,
        strength=correction_strength,
        max_shift=max_shift,
    )
    click.echo(f"  Correction strength: {correction_strength:.0%}")
    click.echo("  Done.")

    # ── Step 4: Time-align vocals ──────────────────────────────
    click.echo("\n[Step 4/5] Aligning timing...")
    ref_vocals_hq, ref_hq_sr = load_audio(stems.vocals_path, sr=OUTPUT_SR, mono=True)

    aligned_vocals = align_vocals(
        user_audio=tuned_vocals,
        user_sr=OUTPUT_SR,
        ref_audio=ref_vocals_hq,
        ref_sr=OUTPUT_SR,
    )
    click.echo("  Done.")

    # ── Step 5: Mix ────────────────────────────────────────────
    click.echo("\n[Step 5/5] Mixing final output...")

    # Combine non-vocal stems into instrumental
    instrumental, inst_sr = combine_stems(stems.instrumental_paths, sr=OUTPUT_SR)

    # Mix vocals with instrumental
    final_mix, final_sr = mix_vocals_with_instrumental(
        vocals=aligned_vocals,
        vocals_sr=OUTPUT_SR,
        instrumental=instrumental,
        instrumental_sr=inst_sr,
        reference_vocals=ref_vocals_hq,
        apply_fx=apply_effects,
        reverb_room=reverb_room,
        reverb_wet=reverb_wet,
        vocal_gain_db=vocal_gain_db,
    )

    save_audio(output_path, final_mix, final_sr)
    click.echo(f"  Saved: {output_path}")
    click.echo("\nDone! Your tuned song is ready.")

    return TuneResult(
        output_path=output_path,
        stems=stems,
        ref_key=ref_key,
        ref_tempo=ref_rhythm.tempo,
        ref_pitch=ref_pitch,
        user_pitch=user_pitch,
    )
