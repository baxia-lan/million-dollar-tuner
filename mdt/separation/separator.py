"""Stem separation using Meta's Demucs."""

from __future__ import annotations

import sys
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
    """Separate an audio file into stems using Demucs.

    Parameters
    ----------
    audio_path : str or Path
        Path to the input audio file.
    output_dir : str or Path
        Directory to write stem files into.
    model_name : str
        Demucs model name. Recommended: ``htdemucs_ft`` (best quality)
        or ``htdemucs`` (faster, lower VRAM).
    device : str or None
        ``'cuda'``, ``'cpu'``, or None (auto-detect).
    jobs : int
        Number of parallel jobs for separation. 0 = auto.

    Returns
    -------
    StemResult
        Paths to the separated stem files.
    """
    import torch
    from demucs.api import Separator

    audio_path = Path(audio_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    click.echo(f"  Model: {model_name} | Device: {device}")
    click.echo(f"  Separating: {audio_path.name}")

    separator = Separator(model=model_name, device=device, jobs=jobs)
    origin, separated = separator.separate_audio_file(audio_path)

    # separated is a dict: stem_name -> tensor (channels, samples)
    sample_rate = separator.samplerate

    result = StemResult(
        output_dir=output_dir,
        model=model_name,
        source_path=audio_path,
    )

    for stem_name, tensor in separated.items():
        stem_path = output_dir / f"{stem_name}.wav"
        # tensor shape: (channels, samples)
        audio_np = tensor.cpu().numpy()
        # soundfile expects (samples, channels)
        sf.write(str(stem_path), audio_np.T, sample_rate)
        click.echo(f"  Saved: {stem_path}")

        attr_name = f"{stem_name}_path"
        if hasattr(result, attr_name):
            setattr(result, attr_name, stem_path)

    return result


def load_stems_as_audio(
    result: StemResult,
    sr: int | None = None,
    mono: bool = False,
) -> dict[str, tuple[np.ndarray, int]]:
    """Load all stem files from a StemResult into numpy arrays.

    Returns a dict of stem_name -> (audio_array, sample_rate).
    """
    from mdt.audio.io import load_audio

    stems = {}
    for name, path in result.all_paths.items():
        y, out_sr = load_audio(path, sr=sr, mono=mono)
        stems[name] = (y, out_sr)
    return stems
