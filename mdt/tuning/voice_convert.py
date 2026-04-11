"""Voice timbre transfer: replace SUNO vocal timbre with the user's timbre.

Source-filter model:
- Source (excitation: pitch harmonics + timing) → keep from SUNO
- Filter (vocal tract: formants + timbre) → take from USER

Only transfers the spectral SHAPE (relative peaks/valleys), not
the absolute volume. Uses per-frame energy normalization to guarantee
output volume matches input volume exactly.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct


def _cepstral_envelope(log_mag: np.ndarray, order: int) -> np.ndarray:
    """Low-quefrency cepstral envelope of a single log-magnitude frame."""
    cep = dct(log_mag, type=2, norm="ortho")
    cep[order:] = 0
    return idct(cep, type=2, norm="ortho")


def convert_voice_timbre(
    suno_vocals: np.ndarray,
    user_audio: np.ndarray,
    sr: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    order: int = 20,
) -> np.ndarray:
    """Replace SUNO vocal timbre with user's timbre.

    For each SUNO frame:
      1. Extract envelope and residual (pitch harmonics)
      2. Replace envelope shape with user's shape
      3. Force output frame energy = input frame energy (no volume change)
      4. Recombine with original phase
    """
    import librosa

    # ── Extract user's average spectral shape ──────────────────
    S_user = librosa.stft(user_audio, n_fft=n_fft, hop_length=hop_length)
    mag_user = np.abs(S_user)

    # Only loud frames
    frame_energy = np.sum(mag_user ** 2, axis=0)
    loud = frame_energy > np.median(frame_energy)
    if np.sum(loud) < 5:
        loud = np.ones(mag_user.shape[1], dtype=bool)

    log_user = np.log(np.maximum(mag_user[:, loud], 1e-10))
    user_shapes = np.zeros_like(log_user)
    for i in range(log_user.shape[1]):
        env = _cepstral_envelope(log_user[:, i], order)
        user_shapes[:, i] = env - np.mean(env)  # zero-mean shape
    user_shape = np.mean(user_shapes, axis=1)
    # Re-center
    user_shape -= np.mean(user_shape)

    # ── Process SUNO vocals ────────────────────────────────────
    S_suno = librosa.stft(suno_vocals, n_fft=n_fft, hop_length=hop_length)
    mag_suno = np.abs(S_suno)
    phase_suno = np.angle(S_suno)

    log_suno = np.log(np.maximum(mag_suno, 1e-10))
    n_freq, n_frames = log_suno.shape
    new_mag = np.zeros_like(mag_suno)

    for i in range(n_frames):
        suno_env = _cepstral_envelope(log_suno[:, i], order)
        residual = log_suno[:, i] - suno_env

        # Replace shape: suno_level + user_shape + residual
        suno_level = np.mean(suno_env)
        log_frame = suno_level + user_shape + residual
        new_mag[:, i] = np.exp(log_frame)

    # ── Per-frame energy normalization ─────────────────────────
    # This is the critical step: force each frame's energy to match
    # the original exactly. This guarantees:
    # - Output is never silent (energy is preserved)
    # - Output is never clipping (energy doesn't explode)
    # - Only the spectral DISTRIBUTION changes, not the level
    orig_energy = np.sum(mag_suno ** 2, axis=0) + 1e-20
    new_energy = np.sum(new_mag ** 2, axis=0) + 1e-20
    scale = np.sqrt(orig_energy / new_energy)
    new_mag *= scale[np.newaxis, :]

    # Reconstruct
    S_out = new_mag * np.exp(1j * phase_suno)
    out = librosa.istft(S_out, hop_length=hop_length, length=len(suno_vocals))

    return out.astype(np.float32)
