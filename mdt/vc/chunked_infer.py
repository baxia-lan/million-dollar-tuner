"""Phrase-level chunked inference with overlap/crossfade.

Instead of converting the entire song at once, split at phrase boundaries,
convert each chunk separately (with per-section param tuning), and
reassemble with crossfade. This prevents:
- Longform pitch drift (so-vits-svc issue)
- Accumulation of artifacts over time
- Allows different inference params per section type (rap vs melodic)
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_sections(sections_path: str | Path) -> list[dict]:
    with open(sections_path) as f:
        return json.load(f)["sections"]


def extract_chunk(y: np.ndarray, sr: int, start: float, end: float,
                  pad_before: float = 0.5, pad_after: float = 0.5) -> tuple[np.ndarray, float, float]:
    """Extract a chunk with padding for crossfade context.

    Returns (chunk, actual_start, actual_end) where actual_start/end
    include the padding.
    """
    actual_start = max(0.0, start - pad_before)
    actual_end = min(len(y) / sr, end + pad_after)

    s = int(actual_start * sr)
    e = int(actual_end * sr)
    return y[s:e], actual_start, actual_end


def crossfade_chunks(chunks: list[tuple[np.ndarray, float, float]],
                     sr: int, total_length: int,
                     fade_duration: float = 0.3) -> np.ndarray:
    """Reassemble chunks with crossfade overlap.

    Each chunk is (audio, start_time, end_time).
    """
    output = np.zeros(total_length, dtype=np.float32)
    weight = np.zeros(total_length, dtype=np.float32)

    fade_samples = int(fade_duration * sr)

    for audio, start_t, end_t in chunks:
        s = int(start_t * sr)
        n = min(len(audio), total_length - s)
        if n <= 0:
            continue

        # Create fade envelope
        env = np.ones(n, dtype=np.float32)
        if fade_samples > 0 and fade_samples < n:
            # Fade in
            env[:fade_samples] = np.linspace(0, 1, fade_samples, dtype=np.float32)
            # Fade out
            env[-fade_samples:] = np.linspace(1, 0, fade_samples, dtype=np.float32)

        output[s:s + n] += audio[:n] * env
        weight[s:s + n] += env

    # Normalize by overlap weight
    mask = weight > 1e-10
    output[mask] /= weight[mask]

    return output


def chunked_inference(
    source_vocals_path: str | Path,
    sections_path: str | Path,
    infer_fn,
    output_path: str | Path,
    sr: int = 44100,
    pad_seconds: float = 0.5,
    fade_seconds: float = 0.3,
    rap_params: dict | None = None,
    melodic_params: dict | None = None,
    default_params: dict | None = None,
) -> Path:
    """Convert vocals chunk-by-chunk using section boundaries.

    Parameters
    ----------
    source_vocals_path : path to source vocals WAV
    sections_path : path to song_sections.json
    infer_fn : callable(input_path, output_path, **params) -> None
        Backend inference function that converts a WAV file.
    output_path : where to save the final reassembled result
    rap_params : inference params override for rap sections
    melodic_params : inference params override for melodic sections
    default_params : default inference params for other section types
    """
    import tempfile

    source_vocals_path = Path(source_vocals_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load source
    y_source, source_sr = sf.read(str(source_vocals_path))
    if y_source.ndim > 1:
        y_source = y_source.mean(axis=1)
    if source_sr != sr:
        y_source = librosa.resample(y_source, orig_sr=source_sr, target_sr=sr)

    sections = load_sections(sections_path)
    total_samples = len(y_source)

    if default_params is None:
        default_params = {}
    if rap_params is None:
        rap_params = default_params.copy()
    if melodic_params is None:
        melodic_params = default_params.copy()

    converted_chunks = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, sec in enumerate(sections):
            start, end = sec["start"], sec["end"]
            stype = sec.get("type", "other")
            label = sec.get("label", f"section_{i}")

            # Extract chunk with padding
            chunk, actual_start, actual_end = extract_chunk(
                y_source, sr, start, end,
                pad_before=pad_seconds, pad_after=pad_seconds
            )

            if len(chunk) < sr * 0.5:  # skip very short sections
                continue

            # Save chunk to temp file
            chunk_in = Path(tmpdir) / f"chunk_{i:02d}_in.wav"
            chunk_out = Path(tmpdir) / f"chunk_{i:02d}_out.wav"
            sf.write(str(chunk_in), chunk.astype(np.float32), sr)

            # Select params based on section type
            if stype == "rap":
                params = rap_params
            elif stype == "melodic":
                params = melodic_params
            else:
                params = default_params

            print(f"  [{i+1}/{len(sections)}] {label} ({stype}) "
                  f"{actual_start:.1f}s-{actual_end:.1f}s")

            # Run inference on this chunk
            infer_fn(str(chunk_in), str(chunk_out), **params)

            # Load converted chunk
            if chunk_out.exists():
                conv, conv_sr = sf.read(str(chunk_out))
                if conv.ndim > 1:
                    conv = conv.mean(axis=1)
                if conv_sr != sr:
                    conv = librosa.resample(conv, orig_sr=conv_sr, target_sr=sr)
                converted_chunks.append((conv.astype(np.float32), actual_start, actual_end))
            else:
                # Fallback: use original chunk
                print(f"    WARNING: inference failed for {label}, using original")
                converted_chunks.append((chunk.astype(np.float32), actual_start, actual_end))

    # Reassemble with crossfade
    print(f"  Reassembling {len(converted_chunks)} chunks...")
    result = crossfade_chunks(converted_chunks, sr, total_samples, fade_seconds)

    sf.write(str(output_path), result, sr)
    print(f"  Saved: {output_path} ({len(result)/sr:.1f}s)")
    return output_path
