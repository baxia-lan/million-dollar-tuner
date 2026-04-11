"""Audio effects powered by Spotify's Pedalboard."""

from __future__ import annotations

import numpy as np
from pedalboard import (
    Compressor,
    HighpassFilter,
    LowShelfFilter,
    Pedalboard,
    Reverb,
)

from mdt.config import (
    DEFAULT_COMPRESSOR_RATIO,
    DEFAULT_COMPRESSOR_THRESHOLD,
    DEFAULT_HIGHPASS_FREQ,
    DEFAULT_REVERB_ROOM,
    DEFAULT_REVERB_WET,
)


def build_vocal_chain(
    reverb_room: float = DEFAULT_REVERB_ROOM,
    reverb_wet: float = DEFAULT_REVERB_WET,
    compress_threshold: float = DEFAULT_COMPRESSOR_THRESHOLD,
    compress_ratio: float = DEFAULT_COMPRESSOR_RATIO,
    highpass_freq: float = DEFAULT_HIGHPASS_FREQ,
) -> Pedalboard:
    """Create a vocal processing effects chain.

    Returns a Pedalboard instance ready to process audio.
    """
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=highpass_freq),
        Compressor(
            threshold_db=compress_threshold,
            ratio=compress_ratio,
        ),
        LowShelfFilter(cutoff_frequency_hz=200, gain_db=-2.0),
        Reverb(room_size=reverb_room, wet_level=reverb_wet),
    ])


def apply_effects(
    y: np.ndarray,
    sr: int,
    board: Pedalboard | None = None,
    **kwargs,
) -> np.ndarray:
    """Apply an effects chain to audio.

    Parameters
    ----------
    y : np.ndarray
        Audio (mono or stereo).  If mono, shape ``(samples,)``.
    sr : int
        Sample rate.
    board : Pedalboard or None
        Pre-built board.  If None, builds a default vocal chain with *kwargs*.

    Returns
    -------
    np.ndarray
        Processed audio with the same shape and dtype.
    """
    if board is None:
        board = build_vocal_chain(**kwargs)

    # Pedalboard expects (channels, samples) float32
    was_1d = y.ndim == 1
    if was_1d:
        y = y[np.newaxis, :]

    y = y.astype(np.float32)
    out = board(y, sr)

    if was_1d:
        out = out[0]
    return out
