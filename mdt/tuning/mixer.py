"""Final mix: combine tuned vocals with separated instrumentals."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mdt.audio.convert import match_rms
from mdt.audio.effects import apply_effects, build_vocal_chain
from mdt.audio.io import load_audio, save_audio


def mix_vocals_with_instrumental(
    vocals: np.ndarray,
    vocals_sr: int,
    instrumental: np.ndarray,
    instrumental_sr: int,
    reference_vocals: np.ndarray | None = None,
    apply_fx: bool = True,
    reverb_room: float = 0.3,
    reverb_wet: float = 0.15,
    vocal_gain_db: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Mix processed vocals with instrumental.

    Parameters
    ----------
    vocals : np.ndarray
        Tuned and time-aligned user vocals (mono).
    vocals_sr : int
        Sample rate of vocals.
    instrumental : np.ndarray
        Instrumental mix (can be mono or stereo).
    instrumental_sr : int
        Sample rate of instrumental.
    reference_vocals : np.ndarray or None
        Original SUNO vocals for RMS matching.
    apply_fx : bool
        Whether to apply vocal effects chain.
    reverb_room, reverb_wet : float
        Reverb parameters.
    vocal_gain_db : float
        Additional gain for vocals in dB.

    Returns
    -------
    (mix, sr) : tuple
        Final mix and sample rate.
    """
    import librosa

    # Ensure same sample rate
    sr = instrumental_sr
    if vocals_sr != sr:
        vocals = librosa.resample(vocals, orig_sr=vocals_sr, target_sr=sr)

    # Level matching: match user vocal RMS to reference vocals
    if reference_vocals is not None:
        vocals = match_rms(vocals, reference_vocals)

    # Apply additional gain
    if vocal_gain_db != 0.0:
        vocals = vocals * (10.0 ** (vocal_gain_db / 20.0))

    # Apply vocal effects chain
    if apply_fx:
        board = build_vocal_chain(
            reverb_room=reverb_room,
            reverb_wet=reverb_wet,
        )
        vocals = apply_effects(vocals, sr, board=board)

    # Ensure matching length
    if instrumental.ndim == 1:
        # Mono instrumental
        target_len = len(instrumental)
        vocals = _match_length(vocals, target_len)
        mix = vocals + instrumental
    else:
        # Stereo instrumental (channels, samples)
        target_len = instrumental.shape[-1]
        vocals = _match_length(vocals, target_len)
        # Center-pan the vocals
        mix = instrumental.copy()
        mix[0] += vocals
        mix[1] += vocals

    # Prevent clipping
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix = mix / peak * 0.95

    return mix, sr


def combine_stems(
    stem_paths: list[Path],
    sr: int | None = None,
) -> tuple[np.ndarray, int]:
    """Load and sum multiple stem files into one mix.

    Parameters
    ----------
    stem_paths : list of Path
        Paths to stem WAV files.
    sr : int or None
        Target sample rate. None uses the native rate of the first file.

    Returns
    -------
    (mix, sr) : tuple
    """
    import librosa

    mix = None
    out_sr = sr

    for path in stem_paths:
        y, file_sr = load_audio(path, sr=sr, mono=False)
        if out_sr is None:
            out_sr = file_sr

        if mix is None:
            mix = y
        else:
            # Ensure same length
            if y.ndim != mix.ndim:
                if y.ndim == 1:
                    y = np.stack([y, y])
                else:
                    mix = np.stack([mix, mix]) if mix.ndim == 1 else mix

            min_len = min(
                mix.shape[-1] if mix.ndim > 1 else len(mix),
                y.shape[-1] if y.ndim > 1 else len(y),
            )
            if mix.ndim > 1:
                mix = mix[:, :min_len] + y[:, :min_len]
            else:
                mix = mix[:min_len] + y[:min_len]

    return mix, out_sr


def _match_length(y: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or trim audio to match target length."""
    if len(y) >= target_len:
        return y[:target_len]
    else:
        return np.pad(y, (0, target_len - len(y)))
