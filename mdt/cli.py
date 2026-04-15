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


# ────────────────────────────────────────────────────────────────
# mdt profile — Voice profile management
# ────────────────────────────────────────────────────────────────

@main.group()
def profile():
    """Manage voice profiles for neural voice conversion."""
    pass


@profile.command("create")
@click.argument("name")
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--backends", default="applio,sovits,seedvc",
              help="Comma-separated list of backends to train.")
@click.option("--epochs", default=30, type=int,
              help="Training epochs for each backend.")
@click.option("--batch-size", default=4, type=int,
              help="Training batch size.")
def profile_create(name, audio_file, backends, epochs, batch_size):
    """Create a new voice profile from a recording.

    NAME is the profile identifier (e.g. 'my_voice').
    AUDIO_FILE is a WAV/M4A recording of your voice (10+ minutes recommended).

    Example:

        mdt profile create my_voice ~/recordings/voice.wav --backends applio
    """
    from mdt.profiles.manager import create_profile

    backend_list = [b.strip() for b in backends.split(",")]
    click.echo(f"Creating profile '{name}' with backends: {backend_list}")
    click.echo(f"Training {epochs} epochs, batch size {batch_size}")
    click.echo()

    meta = create_profile(
        name=name,
        audio_path=audio_file,
        backends=backend_list,
        epochs=epochs,
        batch_size=batch_size,
    )

    click.echo()
    click.echo(f"Profile '{name}' created successfully!")
    for bname, bstatus in meta.backends.items():
        status = "trained" if bstatus.trained else "FAILED"
        click.echo(f"  {bname}: {status}")


@profile.command("list")
def profile_list():
    """List all voice profiles."""
    from mdt.profiles.manager import list_profiles

    profiles = list_profiles()
    if not profiles:
        click.echo("No profiles found. Create one with: mdt profile create <name> <audio>")
        return

    click.echo(f"{'Name':<20} {'Created':<22} {'Backends'}")
    click.echo("-" * 60)
    for p in profiles:
        backends = ", ".join(
            f"{k}({'ok' if v.trained else 'x'})" for k, v in p.backends.items()
        )
        click.echo(f"{p.name:<20} {p.created_at[:19]:<22} {backends}")


@profile.command("refresh")
@click.argument("name")
@click.option("--audio", default=None, type=click.Path(exists=True),
              help="New audio file to retrain with.")
@click.option("--backends", default=None,
              help="Comma-separated list of backends to retrain.")
@click.option("--epochs", default=30, type=int)
@click.option("--batch-size", default=4, type=int)
def profile_refresh(name, audio, backends, epochs, batch_size):
    """Retrain backends for an existing profile.

    Example:

        mdt profile refresh my_voice --audio new_recording.wav --backends applio
    """
    from mdt.profiles.manager import refresh_profile

    backend_list = [b.strip() for b in backends.split(",")] if backends else None
    click.echo(f"Refreshing profile '{name}'...")

    meta = refresh_profile(
        name=name,
        audio_path=audio,
        backends=backend_list,
        epochs=epochs,
        batch_size=batch_size,
    )

    click.echo(f"Profile '{name}' refreshed!")
    for bname, bstatus in meta.backends.items():
        status = "trained" if bstatus.trained else "FAILED"
        click.echo(f"  {bname}: {status}")


# ────────────────────────────────────────────────────────────────
# mdt convert — Neural voice conversion
# ────────────────────────────────────────────────────────────────

@main.command()
@click.argument("profile_name")
@click.argument("song", type=click.Path(exists=True))
@click.option("-o", "--output", default="converted.wav",
              help="Output file path.")
@click.option("--backend", default=None,
              help="Backend to use (applio/sovits/seedvc). Auto-selects if omitted.")
@click.option("--pitch-shift", default=0, type=int,
              help="Pitch shift in semitones.")
@click.option("--stems-dir", default=None, type=click.Path(),
              help="Pre-separated stems directory (skips separation).")
@click.option("--vocals-only", is_flag=True,
              help="Output converted vocals only (no mixing with instrumentals).")
@click.option("--sections", default=None, type=click.Path(exists=True),
              help="Song sections JSON for phrase-level chunked inference.")
def convert(profile_name, song, output, backend, pitch_shift, stems_dir, vocals_only, sections):
    """Convert a song's vocals to your voice using a trained profile.

    PROFILE_NAME is a previously created voice profile.
    SONG is the input song (WAV/MP3).

    Example:

        mdt convert my_voice suno_song.wav -o output.wav --backend applio
    """
    from pathlib import Path as P

    from mdt.profiles.manager import load_profile, get_profile_dir
    from mdt.vc.registry import get_backend, get_best_backend

    click.echo("=" * 60)
    click.echo("  Million Dollar Tuner — Neural Voice Conversion")
    click.echo("=" * 60)

    meta = load_profile(profile_name)
    profile_dir = get_profile_dir(profile_name)

    # Select backend
    if backend:
        vc = get_backend(backend)
        if backend not in meta.backends or not meta.backends[backend].trained:
            click.echo(f"WARNING: Backend '{backend}' not trained for this profile.")
    else:
        vc = get_best_backend(profile_dir)
        click.echo(f"Auto-selected backend: {vc.name}")

    # Separate stems if needed
    if stems_dir:
        vocals_path = P(stems_dir) / "vocals.wav"
        if not vocals_path.exists():
            click.echo(f"ERROR: {vocals_path} not found in stems directory")
            return
    else:
        click.echo("\nSeparating stems...")
        from mdt.separation.separator import separate as do_separate
        stems = do_separate(audio_path=song, output_dir="convert_stems")
        vocals_path = P(stems.vocals_path)
        stems_dir = str(P(stems.vocals_path).parent)

    # Run voice conversion
    converted_path = P(output).with_suffix(".vocals.wav") if not vocals_only else P(output)

    if sections:
        # Phrase-level chunked inference
        click.echo(f"\nChunked inference with {vc.name} (sections: {sections})...")
        import tempfile
        import time

        from mdt.vc.chunked_infer import chunked_inference

        def _infer_fn(inp, outp, **params):
            vc.infer(
                profile_dir=profile_dir,
                source_vocals=P(inp),
                output_path=P(outp),
                pitch_shift=pitch_shift,
                **params,
            )

        t0 = time.time()
        chunked_inference(
            source_vocals_path=str(vocals_path),
            sections_path=sections,
            infer_fn=_infer_fn,
            output_path=str(converted_path),
        )
        elapsed = time.time() - t0
        click.echo(f"Chunked conversion done in {elapsed:.1f}s")
    else:
        click.echo(f"\nConverting vocals with {vc.name}...")
        result = vc.infer(
            profile_dir=profile_dir,
            source_vocals=vocals_path,
            output_path=converted_path,
            pitch_shift=pitch_shift,
        )
        click.echo(f"Conversion done in {result.infer_time_seconds:.1f}s")

    # Mix with instrumentals
    if not vocals_only and stems_dir:
        click.echo("\nMixing with instrumentals...")
        from mdt.vc.mixer import mix_with_stems
        mix_with_stems(
            vocals_path=str(converted_path),
            stems_dir=stems_dir,
            output_path=output,
            reference_vocals_path=str(vocals_path),
        )
        click.echo(f"Saved: {output}")
    else:
        click.echo(f"Saved: {converted_path}")


# ────────────────────────────────────────────────────────────────
# mdt benchmark — Compare backends
# ────────────────────────────────────────────────────────────────

@main.command()
@click.argument("profile_name")
@click.argument("song", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default="benchmark",
              help="Output directory for benchmark results.")
@click.option("--stems-dir", default=None, type=click.Path(),
              help="Pre-separated stems directory.")
@click.option("--sections", default=None, type=click.Path(exists=True),
              help="Song sections JSON for per-section evaluation.")
def benchmark(profile_name, song, output_dir, stems_dir, sections):
    """Benchmark all trained backends for a profile on a song.

    Runs inference with each backend and compares speaker similarity,
    pitch accuracy, and artifact levels.

    Example:

        mdt benchmark my_voice suno_song.wav -o ./bench/ --sections song_sections.json
    """
    from pathlib import Path as P

    from mdt.profiles.manager import load_profile, get_profile_dir
    from mdt.vc.metrics import evaluate
    from mdt.vc.registry import get_backend

    click.echo("=" * 60)
    click.echo("  Million Dollar Tuner — Backend Benchmark")
    click.echo("=" * 60)

    meta = load_profile(profile_name)
    profile_dir = get_profile_dir(profile_name)
    output_dir = P(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get reference audio
    ref_audio = profile_dir / "reference_audio.wav"

    # Separate stems
    if stems_dir:
        vocals_path = P(stems_dir) / "vocals.wav"
    else:
        click.echo("\nSeparating stems...")
        from mdt.separation.separator import separate as do_separate
        stems = do_separate(audio_path=song, output_dir=str(output_dir / "stems"))
        vocals_path = P(stems.vocals_path)

    # Run each trained backend
    results = {}
    for bname, bstatus in meta.backends.items():
        if not bstatus.trained:
            click.echo(f"\n--- {bname}: not trained, skipping ---")
            continue

        click.echo(f"\n--- {bname}: running inference ---")
        try:
            vc = get_backend(bname)
            out_path = output_dir / f"{bname}_vocals.wav"

            result = vc.infer(
                profile_dir=profile_dir,
                source_vocals=vocals_path,
                output_path=out_path,
            )
            click.echo(f"  Done in {result.infer_time_seconds:.1f}s")

            # Evaluate
            click.echo("  Evaluating...")
            eval_result = evaluate(
                str(ref_audio), str(vocals_path), str(out_path)
            )
            results[bname] = {
                "sim_to_user": eval_result.sim_to_user,
                "f0_correlation": eval_result.f0_correlation,
                "output": str(out_path),
            }
            click.echo(f"  Speaker similarity: {eval_result.sim_to_user:.4f}")
            click.echo(f"  F0 correlation: {eval_result.f0_correlation:.4f}")

        except Exception as e:
            click.echo(f"  ERROR: {e}")
            results[bname] = {"error": str(e)}

    # Section-level evaluation if sections provided
    if sections:
        click.echo("\n--- Section-level evaluation ---")
        from mdt.vc.section_eval import evaluate_sections, print_report, save_report

        for bname, r in results.items():
            if "error" in r:
                continue
            report = evaluate_sections(
                user_audio_path=str(ref_audio),
                source_vocals_path=str(vocals_path),
                converted_vocals_path=r["output"],
                sections_path=sections,
                model_name=bname,
            )
            print_report(report)
            save_report(report, output_dir / f"section_report_{bname}.json")

    # Summary
    click.echo(f"\n{'='*60}")
    click.echo(f"  {'Backend':<12} {'→User':>8} {'F0corr':>8}")
    click.echo(f"  {'-'*28}")
    for bname, r in results.items():
        if "error" in r:
            click.echo(f"  {bname:<12} {'ERROR':>8}")
        else:
            click.echo(f"  {bname:<12} {r['sim_to_user']:>8.4f} {r['f0_correlation']:>8.4f}")

    import json
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    click.echo(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
