"""Audio I/O utilities: loading and saving WAV/MP3 files."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(
    path: str | Path,
    sr: int | None = None,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load an audio file (WAV, MP3, FLAC, OGG) and return (samples, sample_rate).

    Parameters
    ----------
    path : str or Path
        Path to the audio file.
    sr : int or None
        Target sample rate. ``None`` keeps the native rate.
    mono : bool
        If True, downmix to mono.

    Returns
    -------
    y : np.ndarray
        Audio samples as float32.  Shape ``(samples,)`` if mono,
        ``(channels, samples)`` if stereo.
    sr : int
        Sample rate of the returned audio.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".mp3":
        # librosa handles MP3 via audioread / soundfile fallback
        y, out_sr = librosa.load(str(path), sr=sr, mono=mono)
    else:
        # Use soundfile for WAV/FLAC/OGG (faster)
        y, out_sr = sf.read(str(path), dtype="float32", always_2d=True)
        # soundfile returns (samples, channels) – transpose to (channels, samples)
        y = y.T
        if mono and y.shape[0] > 1:
            y = np.mean(y, axis=0)
        elif mono:
            y = y[0]
        if sr is not None and sr != out_sr:
            y = librosa.resample(y, orig_sr=out_sr, target_sr=sr)
            out_sr = sr

    return y.astype(np.float32), out_sr


def save_audio(
    path: str | Path,
    y: np.ndarray,
    sr: int,
    format: str | None = None,
) -> Path:
    """Save audio samples to a file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    y : np.ndarray
        Audio samples (mono or stereo).
    sr : int
        Sample rate.
    format : str or None
        Force output format (``'wav'``, ``'mp3'``, ``'flac'``).
        Defaults to extension of *path*.

    Returns
    -------
    Path
        The path the file was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = (format or path.suffix.lstrip(".")).lower()

    if fmt == "mp3":
        # pydub for MP3 encoding
        from pydub import AudioSegment

        # Normalize to int16
        pcm = np.clip(y, -1.0, 1.0)
        pcm = (pcm * 32767).astype(np.int16)
        if pcm.ndim == 1:
            channels = 1
            raw = pcm.tobytes()
        else:
            channels = pcm.shape[0]
            # Interleave channels
            raw = pcm.T.flatten().tobytes()
        seg = AudioSegment(
            data=raw,
            sample_width=2,
            frame_rate=sr,
            channels=channels,
        )
        seg.export(str(path), format="mp3")
    else:
        # soundfile handles WAV/FLAC/OGG
        if y.ndim == 2:
            # (channels, samples) -> (samples, channels) for soundfile
            y = y.T
        sf.write(str(path), y, sr)

    return path


def get_duration(path: str | Path) -> float:
    """Return duration of an audio file in seconds."""
    path = Path(path)
    info = sf.info(str(path))
    return info.duration


def ensure_mono(y: np.ndarray) -> np.ndarray:
    """Convert to mono if stereo."""
    if y.ndim == 2:
        return np.mean(y, axis=0)
    return y


def ensure_stereo(y: np.ndarray) -> np.ndarray:
    """Convert mono to stereo by duplicating the channel."""
    if y.ndim == 1:
        return np.stack([y, y], axis=0)
    return y
