"""Voice conversion using so-vits-svc-fork.

This is a real ML-based voice conversion system. It:
1. Trains a small model on the user's voice (~10min of audio, ~10-20min training)
2. Uses that model to convert any vocals to sound like the user

Unlike signal processing approaches (spectral envelope, LPC, frequency warping),
this actually captures what makes a voice unique — the full spectral and temporal
characteristics learned by a neural network.

Two-phase workflow:
  Phase 1 (once): mdt train-voice <your_recordings> → trains a model
  Phase 2 (fast): mdt tune <suno_song> → uses the model to convert vocals
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import click
import numpy as np


def train_voice_model(
    audio_paths: list[str | Path],
    speaker_name: str = "user",
    output_dir: str | Path = "voice_model",
    epochs: int = 100,
) -> Path:
    """Train a so-vits-svc model on user's voice recordings.

    Parameters
    ----------
    audio_paths : list of paths
        Paths to user's voice recordings (WAV/MP3).
    speaker_name : str
        Name for the voice model.
    output_dir : path
        Where to store the trained model.
    epochs : int
        Training epochs. 100 is a good start, 200+ for better quality.

    Returns
    -------
    Path
        Directory containing the trained model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # so-vits-svc expects: dataset_raw/{speaker_name}/*.wav
    dataset_dir = output_dir / "dataset_raw" / speaker_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Copy/convert audio files into dataset dir
    click.echo(f"  Preparing dataset for speaker '{speaker_name}'...")
    for i, src in enumerate(audio_paths):
        src = Path(src)
        dst = dataset_dir / f"{speaker_name}_{i:04d}.wav"
        if src.suffix.lower() == ".wav":
            shutil.copy2(src, dst)
        else:
            # Convert to WAV using pydub
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(src))
            audio.export(str(dst), format="wav")
        click.echo(f"  Added: {src.name}")

    # Split long files into segments (so-vits-svc works better with short clips)
    click.echo("  Splitting audio into segments...")
    _run_svc(["pre-split", "--input-dir", str(dataset_dir)],
             cwd=str(output_dir))

    # Step 1: Resample
    click.echo("  Resampling...")
    _run_svc(["pre-resample"], cwd=str(output_dir))

    # Step 2: Config
    click.echo("  Generating config...")
    _run_svc(["pre-config"], cwd=str(output_dir))

    # Step 3: HuBERT features
    click.echo("  Extracting HuBERT features...")
    _run_svc(["pre-hubert"], cwd=str(output_dir))

    # Step 4: Train
    click.echo(f"  Training ({epochs} epochs)... this will take a while.")
    _run_svc(["train", "--max-epochs", str(epochs)], cwd=str(output_dir))

    click.echo(f"  Model saved to: {output_dir}")
    return output_dir


def convert_voice_timbre(
    suno_vocals: np.ndarray,
    user_audio: np.ndarray,
    sr: int = 44100,
    model_dir: str | Path | None = None,
    **kwargs,
) -> np.ndarray:
    """Convert SUNO vocals to user's voice using trained model.

    If model_dir is provided: uses the trained so-vits-svc model (best quality).
    If no model: falls back to WORLD vocoder spectral transfer (basic quality).

    Parameters
    ----------
    suno_vocals : np.ndarray
        Separated SUNO vocal track (mono).
    user_audio : np.ndarray
        User's voice recording (mono). Only used for fallback.
    sr : int
        Sample rate.
    model_dir : path or None
        Path to trained voice model directory.

    Returns
    -------
    np.ndarray
        Voice-converted audio.
    """
    if model_dir is not None:
        model_dir = Path(model_dir)
        if (model_dir / "logs").exists():
            return _convert_with_svc(suno_vocals, sr, model_dir)

    click.echo("  No trained voice model found. Using WORLD vocoder fallback.")
    click.echo("  For much better results, run: mdt train-voice <your_recordings>")
    return _convert_with_world_fallback(suno_vocals, user_audio, sr)


def _convert_with_svc(
    vocals: np.ndarray,
    sr: int,
    model_dir: Path,
) -> np.ndarray:
    """Convert using trained so-vits-svc model."""
    import soundfile as sf
    import tempfile

    # Find the latest model checkpoint
    logs_dir = model_dir / "logs" / "44k"
    if not logs_dir.exists():
        logs_dir = model_dir / "logs"

    g_paths = sorted(logs_dir.glob("G_*.pth"))
    config_paths = list((model_dir / "configs" / "44k").glob("config.json"))
    if not config_paths:
        config_paths = list(model_dir.glob("**/config.json"))

    if not g_paths or not config_paths:
        raise FileNotFoundError(
            f"No trained model found in {model_dir}. "
            f"Run 'mdt train-voice' first."
        )

    model_path = g_paths[-1]  # latest checkpoint
    config_path = config_paths[0]

    click.echo(f"  Using model: {model_path.name}")

    # Save vocals to temp file for svc inference
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.wav"
        output_path = Path(tmpdir) / "output.wav"

        sf.write(str(input_path), vocals, sr)

        # Run svc inference
        from so_vits_svc_fork.inference.core import Svc

        svc = Svc(
            net_g_path=str(model_path),
            config_path=str(config_path),
        )

        # Get speaker name from config
        import json
        with open(config_path) as f:
            config = json.load(f)
        speaker = list(config.get("spk", {}).keys())[0]

        # Infer
        audio_data, _ = sf.read(str(input_path), dtype="float32")
        converted = svc.infer_silence(
            audio=audio_data,
            speaker=speaker,
            transpose=0,
            auto_predict_f0=True,
            f0_method="dio",
        )

    # Match length
    if len(converted) > len(vocals):
        converted = converted[:len(vocals)]
    elif len(converted) < len(vocals):
        converted = np.pad(converted, (0, len(vocals) - len(converted)))

    return converted.astype(np.float32)


def _convert_with_world_fallback(
    suno_vocals: np.ndarray,
    user_audio: np.ndarray,
    sr: int,
) -> np.ndarray:
    """Fallback: WORLD vocoder spectral transfer (basic quality)."""
    import pyworld as pw
    import librosa

    suno_f64 = suno_vocals.astype(np.float64)
    user_f64 = user_audio.astype(np.float64)

    f0_suno, t_suno = pw.harvest(suno_f64, sr)
    sp_suno = pw.cheaptrick(suno_f64, f0_suno, t_suno, sr)
    ap_suno = pw.d4c(suno_f64, f0_suno, t_suno, sr)

    f0_user, t_user = pw.harvest(user_f64, sr)
    sp_user = pw.cheaptrick(user_f64, f0_user, t_user, sr)

    suno_voiced = f0_suno > 0
    user_voiced = f0_user > 0
    if np.sum(suno_voiced) < 5:
        suno_voiced[:] = True
    if np.sum(user_voiced) < 5:
        user_voiced[:] = True

    suno_avg = np.mean(sp_suno[suno_voiced], axis=0)
    user_avg = np.mean(sp_user[user_voiced], axis=0)

    transfer = user_avg / (suno_avg + 1e-20)
    transfer = np.clip(transfer, 0.03, 30.0)
    transfer = transfer * transfer  # apply twice for stronger effect

    # Frequency warp
    n_freq = sp_suno.shape[1]
    freq_bins = np.arange(n_freq, dtype=np.float64)
    sc_suno = np.sum(freq_bins * suno_avg) / (np.sum(suno_avg) + 1e-20)
    sc_user = np.sum(freq_bins * user_avg) / (np.sum(user_avg) + 1e-20)
    warp = np.clip(sc_user / (sc_suno + 1e-20), 0.5, 2.0)

    sp_out = np.zeros_like(sp_suno)
    ap_out = np.zeros_like(ap_suno)
    for i in range(len(sp_suno)):
        new_idx = freq_bins * warp
        sp_out[i] = np.interp(freq_bins, new_idx, sp_suno[i],
                               left=sp_suno[i, 0], right=sp_suno[i, -1]) * transfer
        ap_out[i] = np.interp(freq_bins, new_idx, ap_suno[i],
                               left=ap_suno[i, 0], right=ap_suno[i, -1])

    output = pw.synthesize(f0_suno, sp_out, ap_out, sr)

    if len(output) > len(suno_vocals):
        output = output[:len(suno_vocals)]
    elif len(output) < len(suno_vocals):
        output = np.pad(output, (0, len(suno_vocals) - len(output)))

    rms_in = np.sqrt(np.mean(suno_vocals ** 2)) + 1e-20
    rms_out = np.sqrt(np.mean(output ** 2)) + 1e-20
    output *= rms_in / rms_out

    return output.astype(np.float32)


def _run_svc(args: list[str], cwd: str) -> None:
    """Run an svc CLI command."""
    cmd = ["svc"] + args
    result = subprocess.run(
        cmd, cwd=cwd,
        capture_output=True, text=True, timeout=7200,
    )
    if result.returncode != 0:
        click.echo(f"  svc error: {result.stderr[:500]}")
        raise RuntimeError(f"svc command failed: {' '.join(args)}")
