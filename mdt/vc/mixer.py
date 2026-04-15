"""Mix converted vocals with instrumental stems."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def mix_with_stems(
    vocals_path: str | Path,
    stems_dir: str | Path,
    output_path: str | Path,
    reference_vocals_path: str | Path | None = None,
    sr: int = 44100,
) -> Path:
    """Mix converted vocals with instrumental stems (drums, bass, other).

    If reference_vocals_path is provided, match the RMS level of the
    converted vocals to the reference before mixing.
    """
    vocals_path = Path(vocals_path)
    stems_dir = Path(stems_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load converted vocals
    vocals, v_sr = sf.read(str(vocals_path))
    if vocals.ndim > 1:
        vocals = vocals.mean(axis=1)
    if v_sr != sr:
        vocals = librosa.resample(vocals, orig_sr=v_sr, target_sr=sr)

    # Load and sum instrumental stems
    instrumental = None
    for stem_name in ["drums", "bass", "other"]:
        stem_path = stems_dir / f"{stem_name}.wav"
        if not stem_path.exists():
            continue
        y, s_sr = sf.read(str(stem_path))
        if y.ndim > 1:
            y = y.mean(axis=1)
        if s_sr != sr:
            y = librosa.resample(y, orig_sr=s_sr, target_sr=sr)
        if instrumental is None:
            instrumental = y
        else:
            n = min(len(instrumental), len(y))
            instrumental = instrumental[:n] + y[:n]

    if instrumental is None:
        sf.write(str(output_path), vocals.astype(np.float32), sr)
        return output_path

    # RMS match vocals to reference
    if reference_vocals_path:
        ref, ref_sr = sf.read(str(reference_vocals_path))
        if ref.ndim > 1:
            ref = ref.mean(axis=1)
        ref_rms = np.sqrt(np.mean(ref**2))
        v_rms = np.sqrt(np.mean(vocals**2))
        if v_rms > 1e-6:
            vocals = vocals * (ref_rms / v_rms)

    # Mix
    n = min(len(vocals), len(instrumental))
    mix = vocals[:n] + instrumental[:n]

    # Prevent clipping
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix = mix / peak * 0.95

    sf.write(str(output_path), mix.astype(np.float32), sr)
    return output_path
