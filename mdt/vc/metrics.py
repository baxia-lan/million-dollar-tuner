"""Speaker similarity and melody preservation metrics."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from mdt.vc.base import EvalResult


def load_mono(path: str | Path, sr: int = 44100) -> np.ndarray:
    y, file_sr = sf.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)


def speaker_similarity(a: np.ndarray, b: np.ndarray, sr: int = 44100) -> float:
    """Cosine similarity between resemblyzer speaker embeddings."""
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder()
    emb_a = encoder.embed_utterance(preprocess_wav(a, source_sr=sr))
    emb_b = encoder.embed_utterance(preprocess_wav(b, source_sr=sr))
    return float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-10))


def f0_correlation(
    source: np.ndarray, converted: np.ndarray, sr: int = 44100
) -> tuple[float, float, int]:
    """F0 correlation between source and converted vocals.

    Returns (correlation, mean_error_hz, voiced_frames).
    """
    import pyworld as pw

    f0_src = pw.harvest(source.astype(np.float64), sr)[0]
    f0_conv = pw.harvest(converted.astype(np.float64), sr)[0]

    n = min(len(f0_src), len(f0_conv))
    both_voiced = (f0_src[:n] > 0) & (f0_conv[:n] > 0)
    voiced = int(np.sum(both_voiced))

    if voiced < 10:
        return 0.0, 0.0, voiced

    corr = float(np.corrcoef(f0_src[:n][both_voiced], f0_conv[:n][both_voiced])[0, 1])
    err = float(np.mean(np.abs(f0_src[:n][both_voiced] - f0_conv[:n][both_voiced])))
    return corr, err, voiced


def evaluate(
    user_audio: str | Path,
    source_vocals: str | Path,
    converted_vocals: str | Path,
    sr: int = 44100,
) -> EvalResult:
    """Full evaluation: speaker similarity + F0 correlation."""
    y_user = load_mono(user_audio, sr)
    y_source = load_mono(source_vocals, sr)
    y_conv = load_mono(converted_vocals, sr)

    sim_to_user = speaker_similarity(y_conv, y_user, sr)
    sim_to_source = speaker_similarity(y_conv, y_source, sr)
    sim_baseline = speaker_similarity(y_source, y_user, sr)

    corr, err, voiced = f0_correlation(y_source, y_conv, sr)

    return EvalResult(
        sim_to_user=sim_to_user,
        sim_to_source=sim_to_source,
        sim_baseline=sim_baseline,
        f0_correlation=corr,
        f0_mean_error_hz=err,
        f0_voiced_frames=voiced,
    )
