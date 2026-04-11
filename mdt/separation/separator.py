"""Stem separation using Meta's Demucs, with spectral fallback."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import soundfile as sf

from mdt.config import DEFAULT_MODEL
from mdt.separation.models import StemResult


def _patch_torchaudio():
    """Patch torchaudio to avoid torchcodec ImportError.

    Newer torchaudio versions default to torchcodec backend which may
    not be installed. This patches torchaudio.load to use soundfile
    before demucs (which imports torchaudio at module level) is loaded.
    """
    try:
        import torchaudio
        import torch

        _original_load = torchaudio.load

        def _patched_load(filepath, *args, **kwargs):
            # Try original first
            try:
                return _original_load(filepath, *args, **kwargs)
            except (ImportError, RuntimeError):
                # Fall back to soundfile
                data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
                return torch.from_numpy(data.T), sr

        torchaudio.load = _patched_load

        # Also try setting backend for older torchaudio
        if hasattr(torchaudio, "set_audio_backend"):
            try:
                torchaudio.set_audio_backend("soundfile")
            except Exception:
                pass
    except ImportError:
        pass


def separate(
    audio_path: str | Path,
    output_dir: str | Path = "stems",
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    jobs: int = 0,
) -> StemResult:
    """Separate an audio file into stems.

    If stems already exist in output_dir, reuses them (cache hit).
    Otherwise runs Demucs (or spectral fallback).
    """
    audio_path = Path(audio_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check cache: if all 4 stems exist, skip separation
    cached = _check_cache(audio_path, output_dir)
    if cached is not None:
        click.echo(f"  Using cached stems from {output_dir}")
        return cached

    # Patch torchaudio BEFORE importing demucs
    _patch_torchaudio()

    try:
        return _separate_demucs(audio_path, output_dir, model_name, device)
    except Exception as e:
        click.echo(f"  Demucs failed ({type(e).__name__}: {e})")
        click.echo("  Falling back to spectral separation...")
        click.echo("  NOTE: For best results, install torchcodec: pip install torchcodec")
        return _separate_spectral(audio_path, output_dir)


def _check_cache(audio_path: Path, output_dir: Path) -> StemResult | None:
    """Return a StemResult if all 4 stems already exist in output_dir."""
    expected = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
    paths = {name.replace(".wav", ""): output_dir / name for name in expected}

    if not all(p.exists() for p in paths.values()):
        return None

    # Check stems are newer than the source audio (not stale)
    source_mtime = audio_path.stat().st_mtime
    oldest_stem = min(p.stat().st_mtime for p in paths.values())
    if oldest_stem < source_mtime:
        return None  # Source changed since stems were created

    result = StemResult(
        output_dir=output_dir,
        model="cached",
        source_path=audio_path,
    )
    result.vocals_path = paths["vocals"]
    result.drums_path = paths["drums"]
    result.bass_path = paths["bass"]
    result.other_path = paths["other"]
    return result


def _separate_demucs(
    audio_path: Path,
    output_dir: Path,
    model_name: str,
    device: str | None,
) -> StemResult:
    """Separate using Demucs neural network."""
    import torch
    import librosa
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    click.echo(f"  Model: {model_name} | Device: {device}")
    click.echo(f"  Separating: {audio_path.name}")

    model = get_model(model_name)
    model.to(device)
    model.eval()

    # Load audio with soundfile/librosa (avoids torchaudio.load entirely)
    suffix = audio_path.suffix.lower()
    if suffix == ".mp3":
        audio_np, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
        if audio_np.ndim == 1:
            audio_np = np.stack([audio_np, audio_np])
    else:
        audio_np, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        audio_np = audio_np.T  # (samples, channels) -> (channels, samples)
    waveform = torch.from_numpy(audio_np.copy())

    # Resample to model's sample rate if needed
    if sample_rate != model.samplerate:
        resampled = librosa.resample(
            waveform.numpy(), orig_sr=sample_rate, target_sr=model.samplerate
        )
        waveform = torch.from_numpy(resampled)

    # Ensure stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / (ref.std() + 1e-8)
    mix = waveform.unsqueeze(0).to(device)

    with torch.no_grad():
        sources = apply_model(model, mix, device=device)

    sources = sources.squeeze(0).cpu()
    sources = sources * ref.std() + ref.mean()

    source_names = model.sources

    result = StemResult(
        output_dir=output_dir,
        model=model_name,
        source_path=audio_path,
    )

    for i, stem_name in enumerate(source_names):
        stem_audio = sources[i].numpy().T  # (samples, channels)
        stem_path = output_dir / f"{stem_name}.wav"
        sf.write(str(stem_path), stem_audio, model.samplerate)
        click.echo(f"  Saved: {stem_path}")

        attr_name = f"{stem_name}_path"
        if hasattr(result, attr_name):
            setattr(result, attr_name, stem_path)

    return result


def _separate_spectral(
    audio_path: Path,
    output_dir: Path,
) -> StemResult:
    """Fallback: spectral separation using HPSS + frequency bands.

    Quality is much lower than Demucs. Install torchcodec to fix Demucs.
    """
    import librosa

    click.echo(f"  Spectral separation: {audio_path.name}")

    y, sr = librosa.load(str(audio_path), sr=None, mono=False)
    is_stereo = y.ndim == 2

    if is_stereo:
        y_mono = np.mean(y, axis=0)
    else:
        y_mono = y

    S = librosa.stft(y_mono)
    S_harmonic, S_percussive = librosa.decompose.hpss(S, margin=3.0)

    harmonic = librosa.istft(S_harmonic, length=len(y_mono))
    percussive = librosa.istft(S_percussive, length=len(y_mono))

    S_harm_mag = np.abs(S_harmonic)
    S_harm_phase = np.angle(S_harmonic)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    bass_mask = (freqs < 300).astype(float)[:, np.newaxis]
    S_bass = S_harm_mag * bass_mask * np.exp(1j * S_harm_phase)
    bass = librosa.istft(S_bass, length=len(y_mono))

    vocal_mask = ((freqs >= 300) & (freqs <= 4000)).astype(float)[:, np.newaxis]
    S_vocal = S_harm_mag * vocal_mask * np.exp(1j * S_harm_phase)
    vocals = librosa.istft(S_vocal, length=len(y_mono))

    other_mask = (freqs > 4000).astype(float)[:, np.newaxis]
    S_other = S_harm_mag * other_mask * np.exp(1j * S_harm_phase)
    other = librosa.istft(S_other, length=len(y_mono))

    if is_stereo and y.shape[0] >= 2:
        mid = (y[0] + y[1]) / 2
        S_mid = librosa.stft(mid)
        S_mid_mag = np.abs(S_mid)
        S_mid_phase = np.angle(S_mid)
        S_vocal_stereo = S_mid_mag * vocal_mask * np.exp(1j * S_mid_phase)
        vocals = librosa.istft(S_vocal_stereo, length=len(mid))

    result = StemResult(
        output_dir=output_dir,
        model="spectral_fallback",
        source_path=audio_path,
    )

    stems_data = {
        "vocals": vocals,
        "drums": percussive,
        "bass": bass,
        "other": other,
    }

    for name, audio in stems_data.items():
        stem_path = output_dir / f"{name}.wav"
        sf.write(str(stem_path), audio, sr)
        click.echo(f"  Saved: {stem_path}")

        attr_name = f"{name}_path"
        if hasattr(result, attr_name):
            setattr(result, attr_name, stem_path)

    return result
