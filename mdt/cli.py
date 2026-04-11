"""CLI entry point for Million Dollar Tuner."""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
@click.version_option(package_name="million-dollar-tuner")
def main():
    """Million Dollar Tuner (MDT) — Vocal tuning & replacement for SUNO songs.

    Replace SUNO AI vocals with your own voice, automatically tuned
    and time-aligned. Separate stems, analyze tracks, and resynthesize
    instrumentals.
    """
    pass


# ────────────────────────────────────────────────────────────────
# mdt tune — Core vocal replacement
# ────────────────────────────────────────────────────────────────

@main.command()
@click.argument("user_vocals", type=click.Path(exists=True))
@click.argument("suno_song", type=click.Path(exists=True))
@click.option("-o", "--output", default="output.wav",
              help="Output file path (WAV or MP3).")
@click.option("--reverb-room", default=0.3, type=float,
              help="Reverb room size [0.0–1.0].")
@click.option("--reverb-wet", default=0.15, type=float,
              help="Reverb wet level [0.0–1.0].")
@click.option("--vocal-gain", default=0.0, type=float,
              help="Additional vocal gain in dB.")
@click.option("--no-effects", is_flag=True,
              help="Skip vocal effects chain.")
@click.option("--model", default="htdemucs_ft",
              help="Demucs model name.")
@click.option("--device", default=None,
              help="Torch device (cuda/cpu/auto).")
@click.option("--stems-dir", default=None, type=click.Path(),
              help="Directory to save separated stems.")
def tune(user_vocals, suno_song, output,
         reverb_room, reverb_wet, vocal_gain, no_effects,
         model, device, stems_dir):
    """Replace SUNO vocal timbre with your voice.

    Keeps the original song's pitch, timing, and rhythm exactly as-is.
    Only changes the tone color to sound like you.

    USER_VOCALS is a recording of your voice (just talking or singing, any content).
    SUNO_SONG is the SUNO AI-generated song (WAV/MP3).

    Example:

        mdt tune my_voice.wav suno_song.mp3 -o result.wav
    """
    from mdt.tuning.pipeline import run_vocal_replacement

    click.echo("=" * 60)
    click.echo("  Million Dollar Tuner — Voice Timbre Replacement")
    click.echo("=" * 60)

    run_vocal_replacement(
        user_vocals_path=user_vocals,
        suno_song_path=suno_song,
        output_path=output,
        apply_effects=not no_effects,
        reverb_room=reverb_room,
        reverb_wet=reverb_wet,
        vocal_gain_db=vocal_gain,
        demucs_model=model,
        device=device,
        stems_dir=stems_dir,
    )


# ────────────────────────────────────────────────────────────────
# mdt separate — Stem separation
# ────────────────────────────────────────────────────────────────

@main.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default="stems",
              help="Output directory for stems.")
@click.option("--model", default="htdemucs_ft",
              help="Demucs model (htdemucs_ft, htdemucs, htdemucs_6s).")
@click.option("--device", default=None,
              help="Torch device (cuda/cpu/auto).")
def separate(audio_file, output_dir, model, device):
    """Separate a song into stems (vocals, drums, bass, other).

    Example:

        mdt separate song.mp3 -o ./stems/
    """
    from mdt.separation.separator import separate as do_separate

    click.echo("=" * 60)
    click.echo("  Million Dollar Tuner — Stem Separation")
    click.echo("=" * 60)
    click.echo()

    result = do_separate(
        audio_path=audio_file,
        output_dir=output_dir,
        model_name=model,
        device=device,
    )

    click.echo()
    click.echo("Stems saved:")
    for name, path in result.all_paths.items():
        click.echo(f"  {name}: {path}")


# ────────────────────────────────────────────────────────────────
# mdt analyze — Track analysis
# ────────────────────────────────────────────────────────────────

@main.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--detailed", is_flag=True,
              help="Show detailed pitch and rhythm info.")
def analyze(audio_file, detailed):
    """Analyze a track: key, tempo, duration, pitch range.

    Example:

        mdt analyze song.mp3
    """
    from mdt.analysis.key import detect_key
    from mdt.analysis.pitch import detect_pitch
    from mdt.analysis.rhythm import analyze_rhythm
    from mdt.audio.io import load_audio
    from mdt.config import ANALYSIS_SR

    import numpy as np

    click.echo("=" * 60)
    click.echo("  Million Dollar Tuner — Track Analysis")
    click.echo("=" * 60)
    click.echo()

    y, sr = load_audio(audio_file, sr=ANALYSIS_SR, mono=True)
    duration = len(y) / sr
    minutes = int(duration // 60)
    seconds = duration % 60

    click.echo(f"  File: {audio_file}")
    click.echo(f"  Duration: {minutes}:{seconds:04.1f}")
    click.echo()

    click.echo("  Analyzing key...")
    key_result = detect_key(y, sr=sr)
    click.echo(f"  Key: {key_result}")

    click.echo("  Analyzing rhythm...")
    rhythm = analyze_rhythm(y, sr=sr)
    click.echo(f"  Tempo: {rhythm.tempo:.1f} BPM")
    click.echo(f"  Beats detected: {len(rhythm.beat_times)}")
    click.echo(f"  Onsets detected: {len(rhythm.onset_times)}")

    if detailed:
        click.echo()
        click.echo("  Analyzing pitch...")
        pitch = detect_pitch(y, sr=sr)
        voiced = pitch.voiced_flag
        voiced_freq = pitch.frequencies[voiced]
        if len(voiced_freq) > 0:
            min_hz = np.nanmin(voiced_freq)
            max_hz = np.nanmax(voiced_freq)
            median_hz = np.nanmedian(voiced_freq)
            click.echo(f"  Pitch range: {min_hz:.1f} Hz – {max_hz:.1f} Hz")
            click.echo(f"  Median pitch: {median_hz:.1f} Hz")
            click.echo(f"  Voiced frames: {np.sum(voiced)}/{len(voiced)} "
                       f"({np.sum(voiced)/len(voiced)*100:.1f}%)")


# ────────────────────────────────────────────────────────────────
# mdt resynth — Resynthesis (experimental)
# ────────────────────────────────────────────────────────────────

@main.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default="resynth",
              help="Output directory for MIDI and rendered audio.")
@click.option("--soundfont", default=None, type=click.Path(),
              help="Path to .sf2 SoundFont file.")
@click.option("--model", default="htdemucs_ft",
              help="Demucs model for stem separation.")
@click.option("--device", default=None,
              help="Torch device (cuda/cpu/auto).")
@click.option("--midi-only", is_flag=True,
              help="Only transcribe to MIDI, skip synthesis.")
def resynth(audio_file, output_dir, soundfont, model, device, midi_only):
    """Transcribe a song to MIDI and resynthesize with soundfonts.

    This is EXPERIMENTAL. Polyphonic transcription is approximate.
    Drums and bass work best; chords/melody are best-effort.

    Steps:
    1. Separate into stems.
    2. Transcribe each stem to MIDI.
    3. (Optional) Resynthesize MIDI back to audio.

    Example:

        mdt resynth song.mp3 -o ./resynth/ --soundfont piano.sf2
    """
    from mdt.resynthesis.synth_engine import resynthesize_all_stems
    from mdt.resynthesis.transcriber import transcribe_stem
    from mdt.separation.separator import separate as do_separate

    output_dir = Path(output_dir)
    stems_dir = output_dir / "stems"
    midi_dir = output_dir / "midi"
    synth_dir = output_dir / "audio"

    click.echo("=" * 60)
    click.echo("  Million Dollar Tuner — Resynthesis (Experimental)")
    click.echo("=" * 60)

    # Step 1: Separate
    click.echo("\n[Step 1] Separating stems...")
    stems = do_separate(
        audio_path=audio_file,
        output_dir=stems_dir,
        model_name=model,
        device=device,
    )

    # Step 2: Transcribe
    click.echo("\n[Step 2] Transcribing stems to MIDI...")
    midi_dir.mkdir(parents=True, exist_ok=True)

    for stem_name, stem_path in stems.all_paths.items():
        click.echo(f"  Transcribing: {stem_name}")
        midi_path = midi_dir / f"{stem_name}.mid"
        transcribe_stem(
            audio_path=stem_path,
            stem_type=stem_name,
            output_path=midi_path,
        )

    if midi_only:
        click.echo("\nMIDI transcription complete (--midi-only).")
        click.echo(f"MIDI files: {midi_dir}")
        return

    # Step 3: Resynthesize
    click.echo("\n[Step 3] Resynthesizing from MIDI...")
    resynthesize_all_stems(
        midi_dir=midi_dir,
        output_dir=synth_dir,
        soundfont_path=soundfont,
    )

    click.echo(f"\nResynthesis complete!")
    click.echo(f"  MIDI files: {midi_dir}")
    click.echo(f"  Audio files: {synth_dir}")


# ────────────────────────────────────────────────────────────────
# mdt autotune — Simple auto-tune (scale-based)
# ────────────────────────────────────────────────────────────────

@main.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("-o", "--output", default="autotuned.wav",
              help="Output file path.")
@click.option("--key", default=None,
              help="Target key (e.g. 'C major', 'A minor'). Auto-detected if omitted.")
@click.option("--strength", default=0.8, type=float,
              help="Correction strength [0.0–1.0]. 1.0 = hard T-Pain effect.")
def autotune(audio_file, output, key, strength):
    """Apply auto-tune to a vocal recording (scale-based).

    Unlike 'tune', this doesn't need a reference song — it just
    snaps your pitch to the nearest note in a musical scale.

    Example:

        mdt autotune my_voice.wav -o tuned.wav --key "C major"
    """
    from mdt.analysis.key import detect_key
    from mdt.audio.io import load_audio, save_audio
    from mdt.config import ANALYSIS_SR, OUTPUT_SR
    from mdt.tuning.pitch_correct import correct_pitch_to_scale

    click.echo("=" * 60)
    click.echo("  Million Dollar Tuner — Auto-Tune")
    click.echo("=" * 60)
    click.echo()

    # Detect key if not provided
    if key is None:
        click.echo("  Detecting key...")
        y_analysis, sr = load_audio(audio_file, sr=ANALYSIS_SR, mono=True)
        key_result = detect_key(y_analysis, sr=sr)
        click.echo(f"  Detected: {key_result}")
        scale_notes = _key_to_scale(key_result.root, key_result.mode)
    else:
        parts = key.strip().split()
        root = parts[0] if parts else "C"
        mode = parts[1] if len(parts) > 1 else "major"
        scale_notes = _key_to_scale(root, mode)
        click.echo(f"  Target key: {key}")

    click.echo(f"  Scale notes: {', '.join(scale_notes)}")
    click.echo(f"  Strength: {strength:.0%}")

    y, sr = load_audio(audio_file, sr=OUTPUT_SR, mono=True)
    tuned = correct_pitch_to_scale(y, sr, scale_notes=scale_notes, strength=strength)

    save_audio(output, tuned, sr)
    click.echo(f"\n  Saved: {output}")


def _key_to_scale(root: str, mode: str) -> list[str]:
    """Convert a key name to a list of scale notes."""
    all_notes = ["C", "C#", "D", "D#", "E", "F",
                 "F#", "G", "G#", "A", "A#", "B"]

    major_intervals = [0, 2, 4, 5, 7, 9, 11]
    minor_intervals = [0, 2, 3, 5, 7, 8, 10]

    try:
        root_idx = all_notes.index(root)
    except ValueError:
        # Try flats -> sharps conversion
        flat_to_sharp = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
        root = flat_to_sharp.get(root, root)
        root_idx = all_notes.index(root)

    intervals = major_intervals if mode == "major" else minor_intervals
    return [all_notes[(root_idx + i) % 12] for i in intervals]


# ────────────────────────────────────────────────────────────────
# mdt evaluate — Quality evaluation
# ────────────────────────────────────────────────────────────────

@main.command()
@click.argument("result_vocals", type=click.Path(exists=True))
@click.argument("reference_song", type=click.Path(exists=True))
@click.option("--stems-dir", default=None, type=click.Path(),
              help="Directory containing pre-separated stems.")
@click.option("--model", default="htdemucs_ft",
              help="Demucs model (if stems need to be separated).")
@click.option("--device", default=None,
              help="Torch device.")
def evaluate(result_vocals, reference_song, stems_dir, model, device):
    """Evaluate the quality of a vocal replacement result.

    Compare your processed vocals against the reference song to get
    a quality score with specific metrics.

    RESULT_VOCALS is your tuned/processed vocal file.
    REFERENCE_SONG is the original SUNO song.

    Example:

        mdt evaluate my_tuned_vocals.wav suno_song.mp3
    """
    import numpy as np

    from mdt.analysis.pitch import detect_pitch
    from mdt.audio.io import load_audio
    from mdt.config import ANALYSIS_SR, OUTPUT_SR
    from mdt.evaluation import (
        evaluate_pitch_accuracy,
        evaluate_timing_accuracy,
        evaluate_vocal_clarity,
    )
    from mdt.separation.separator import separate as do_separate

    click.echo("=" * 60)
    click.echo("  Million Dollar Tuner — Quality Evaluation")
    click.echo("=" * 60)

    # Separate reference song to get reference vocals
    if stems_dir is None:
        click.echo("\n  Separating reference song...")
        stems = do_separate(
            audio_path=reference_song,
            output_dir="eval_stems",
            model_name=model,
            device=device,
        )
        ref_vocals_path = stems.vocals_path
    else:
        from pathlib import Path
        ref_vocals_path = Path(stems_dir) / "vocals.wav"
        if not ref_vocals_path.exists():
            click.echo(f"  ERROR: {ref_vocals_path} not found.")
            return

    # Load audio
    result_audio, _ = load_audio(result_vocals, sr=OUTPUT_SR, mono=True)
    ref_audio, _ = load_audio(ref_vocals_path, sr=OUTPUT_SR, mono=True)

    # Detect reference pitch
    ref_analysis, _ = load_audio(ref_vocals_path, sr=ANALYSIS_SR, mono=True)
    ref_pitch = detect_pitch(ref_analysis, sr=ANALYSIS_SR)

    click.echo("\n  Evaluating pitch accuracy...")
    pitch_grade = evaluate_pitch_accuracy(result_audio, OUTPUT_SR, ref_pitch)
    click.echo(str(pitch_grade))

    click.echo("  Evaluating timing accuracy...")
    timing_grade = evaluate_timing_accuracy(
        result_audio, OUTPUT_SR, ref_audio, OUTPUT_SR
    )
    click.echo(str(timing_grade))

    click.echo("  Evaluating vocal clarity...")
    clarity_grade = evaluate_vocal_clarity(result_audio, OUTPUT_SR)
    click.echo(str(clarity_grade))

    # Overall
    grade_scores = {"GOOD": 100, "OK": 65, "POOR": 25}
    grades = [pitch_grade, timing_grade, clarity_grade]
    avg = sum(grade_scores[g.grade] for g in grades) / len(grades)

    click.echo("\n  " + "─" * 40)
    click.echo(f"  Overall Score: {avg:.0f}/100")

    if all(g.grade != "POOR" for g in grades):
        click.echo("  Result: PASSED")
    else:
        click.echo("  Result: NEEDS IMPROVEMENT")
        click.echo("  Tip: Try --strength 0.9 or re-record closer to the melody.")


if __name__ == "__main__":
    main()
