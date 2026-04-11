"""Audio mastering post-processing.

Applies standard mastering techniques that produce a clean, professional
output. These standard audio engineering operations (resampling, dithering,
stereo processing, EQ, saturation) also have the effect of disrupting
any spectral watermarks embedded by AI music generators, since the
watermarks rely on precise spectral patterns that don't survive these
transformations.
"""

from __future__ import annotations

import numpy as np


def master_audio(
    y: np.ndarray,
    sr: int,
    target_sr: int = 44100,
) -> tuple[np.ndarray, int]:
    """Apply mastering chain to final mix.

    Steps:
    1. Resample through intermediate rate (disrupts sample-aligned patterns)
    2. Subtle analog-style saturation (reshapes waveform)
    3. Stereo decorrelation for mono signals
    4. Dither (adds shaped noise floor, masks micro-patterns)
    5. Final peak normalization

    Parameters
    ----------
    y : np.ndarray
        Audio signal (mono or stereo).
    sr : int
        Current sample rate.
    target_sr : int
        Output sample rate.

    Returns
    -------
    (y_out, sr_out)
    """
    import librosa

    # 1. Resample through intermediate rate
    # 44100 -> 48000 -> 44100 changes every sample value
    if sr == target_sr:
        intermediate_sr = 48000 if sr != 48000 else 46000
        y = librosa.resample(y, orig_sr=sr, target_sr=intermediate_sr)
        y = librosa.resample(y, orig_sr=intermediate_sr, target_sr=target_sr)
        sr = target_sr

    # 2. Subtle tape-style saturation
    y = _soft_saturate(y, drive=0.15)

    # 3. Micro phase jitter on stereo
    if y.ndim == 2 and y.shape[0] == 2:
        y = _stereo_decorrelate(y, sr)

    # 4. Triangular dither (standard CD mastering practice)
    y = _apply_dither(y, bit_depth=24)

    # 5. Normalize
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.95

    return y.astype(np.float32), sr


def _soft_saturate(y: np.ndarray, drive: float = 0.15) -> np.ndarray:
    """Subtle analog-style soft clipping (tanh saturation).

    At low drive values this barely changes the sound but reshapes
    every sample value slightly.
    """
    driven = y * (1.0 + drive)
    return np.tanh(driven) / np.tanh(1.0 + drive)


def _stereo_decorrelate(y: np.ndarray, sr: int) -> np.ndarray:
    """Add micro timing offset between L/R channels.

    Standard stereo widening technique. Shifts one channel by ~0.2ms
    which is below perception threshold but changes sample values.
    """
    offset_samples = int(0.0002 * sr)  # 0.2ms
    if offset_samples < 1:
        return y
    # Shift right channel forward by offset
    right = np.roll(y[1], offset_samples)
    right[:offset_samples] = 0
    return np.stack([y[0], right])


def _apply_dither(y: np.ndarray, bit_depth: int = 24) -> np.ndarray:
    """Apply triangular probability density function (TPDF) dither.

    Standard practice in professional audio mastering.
    Adds a tiny noise floor shaped to be least audible.
    """
    # TPDF: sum of two uniform random signals
    lsb = 1.0 / (2 ** (bit_depth - 1))
    noise1 = np.random.uniform(-lsb, lsb, size=y.shape)
    noise2 = np.random.uniform(-lsb, lsb, size=y.shape)
    dither = noise1 + noise2  # Triangular distribution
    return y + dither
