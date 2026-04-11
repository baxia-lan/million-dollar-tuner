"""Stem separation using Meta's Demucs, with spectral fallback."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import soundfile as sf

from mdt.config import DEFAULT_MODEL
from mdt.separation.models import StemResult


def separate(
    audio_path: str | Path,
    output_dir: str | Path = "stems",
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    jobs: int = 0,
) -> StemResult:
    """Separate an audio file into stems.

    Tries Demucs first (best quality). If the model can't be loaded
    (e.g., no internet to download, or GPU OOM), automatically falls
    back to spectral-based separation.

    Parameters
    ----------
    audio_path : str or Path
        Path to the input audio file.
    output_dir : str or Path
        Directory to write stem files into.
    model_name : str
        Demucs model name (``htdemucs_ft``, ``htdemucs``, ``htdemucs_6s``).
    device : str or None
        ``'cuda'``, ``'cpu'``, or None (auto-detect).
    jobs : int
        Number of parallel jobs for separation.

    Returns
    -------
    StemResult
    """
    audio_path = Path(audio_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        return _separate_demucs(audio_path, output_dir, model_name, device)
    except Exception as e:
        click.echo(f"  Demucs unavailable ({type(e).__name__}: {e})")
        click.echo("  Falling back to spectral separation...")
        return _separate_spectral(audio_path, output_dir)


def _separate_demucs(
    audio_path: Path,
    output_dir: Path,
    model_name: str,
    device: str | None,
) -> StemResult:
    """Separate using Demucs neural network."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    click.echo(f"  Model: {model_name} | Device: {device}")
    click.echo(f"  Separating: {audio_path.name}")

    model = get_model(model_name)
    model.to(device)
    model.eval()

    # Load audio with soundfile/librosa (works everywhere, no torchcodec needed)
    import librosa
    suffix = audio_path.suffix.lower()
    if suffix == ".mp3":
        # soundfile can't read MP3, use librosa
        audio_np, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
        if audio_np.ndim == 1:
            audio_np = np.stack([audio_np, audio_np])
    else:
        audio_np, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        audio_np = audio_np.T  # (samples, channels) -> (channels, samples)
    waveform = torch.from_numpy(audio_np)

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
    """Fallback separation using spectral techniques.

    Uses librosa's HPSS (Harmonic-Percussive Source Separation) and
    frequency-band splitting to approximate stem separation.

    Quality is lower than Demucs but works offline with no model download.
    """
    import librosa

    click.echo(f"  Spectral separation: {audio_path.name}")

    y, sr = librosa.load(str(audio_path), sr=None, mono=False)
    is_stereo = y.ndim == 2

    if is_stereo:
        y_mono = np.mean(y, axis=0)
    else:
        y_mono = y

    # --- HPSS: Harmonic vs Percussive ---
    S = librosa.stft(y_mono)
    S_harmonic, S_percussive = librosa.decompose.hpss(S, margin=3.0)

    harmonic = librosa.istft(S_harmonic, length=len(y_mono))
    percussive = librosa.istft(S_percussive, length=len(y_mono))

    # --- Frequency band splitting on harmonic part ---
    # Vocals: typically 300 Hz - 4000 Hz, center-dominant
    # Bass: < 300 Hz
    # Other: everything else in the harmonic part

    S_harm_mag = np.abs(S_harmonic)
    S_harm_phase = np.angle(S_harmonic)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Bass mask: < 300 Hz
    bass_mask = (freqs < 300).astype(float)[:, np.newaxis]
    S_bass = S_harm_mag * bass_mask * np.exp(1j * S_harm_phase)
    bass = librosa.istft(S_bass, length=len(y_mono))

    # Vocal mask: 300 - 4000 Hz (prominent mid-range)
    vocal_mask = ((freqs >= 300) & (freqs <= 4000)).astype(float)[:, np.newaxis]
    S_vocal = S_harm_mag * vocal_mask * np.exp(1j * S_harm_phase)
    vocals = librosa.istft(S_vocal, length=len(y_mono))

    # Other: everything above 4000 Hz from harmonic
    other_mask = (freqs > 4000).astype(float)[:, np.newaxis]
    S_other = S_harm_mag * other_mask * np.exp(1j * S_harm_phase)
    other = librosa.istft(S_other, length=len(y_mono))

    # If original was stereo, extend vocal center extraction
    if is_stereo and y.shape[0] >= 2:
        # Center channel extraction: L+R (vocal is usually centered)
        mid = (y[0] + y[1]) / 2
        side = (y[0] - y[1]) / 2

        # Use mid-side to improve vocal isolation
        S_mid = librosa.stft(mid)
        S_mid_mag = np.abs(S_mid)
        S_mid_phase = np.angle(S_mid)
        S_vocal_stereo = S_mid_mag * vocal_mask * np.exp(1j * S_mid_phase)
        vocals = librosa.istft(S_vocal_stereo, length=len(mid))

    # Save stems
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


def load_stems_as_audio(
    result: StemResult,
    sr: int | None = None,
    mono: bool = False,
) -> dict[str, tuple[np.ndarray, int]]:
    """Load all stem files from a StemResult into numpy arrays."""
    from mdt.audio.io import load_audio

    stems = {}
    for name, path in result.all_paths.items():
        y, out_sr = load_audio(path, sr=sr, mono=mono)
        stems[name] = (y, out_sr)
    return stems
