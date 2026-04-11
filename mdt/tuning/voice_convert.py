"""Voice timbre transfer: replace SUNO vocal timbre with the user's timbre.

Source-filter model:
- Source (excitation: pitch harmonics + timing) → keep from SUNO
- Filter (vocal tract: formants + timbre) → take from USER

Algorithm:
1. Compute STFT of SUNO vocals
2. Separate each frame into spectral envelope (timbre) and residual (pitch)
3. Compute average spectral envelope from user's voice
4. Replace SUNO's envelope with user's envelope, keep SUNO's residual
5. Reconstruct

The key: residual = magnitude / envelope captures the harmonic peaks
(pitch). Replacing the envelope changes WHICH frequencies are emphasized
(timbre), without moving the harmonic peaks (pitch stays the same).
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
    blend: float = 0.8,
) -> np.ndarray:
    """Replace SUNO vocal timbre with user's timbre.

    Parameters
    ----------
    suno_vocals : np.ndarray
        Separated SUNO vocal track (mono).
    user_audio : np.ndarray
        User's voice recording (mono).
    sr : int
        Sample rate.
    order : int
        Cepstral order. Controls how much detail the envelope captures.
        Lower = smoother envelope = more dramatic timbre change.
        20 is good for voice conversion (captures ~5 formants).
    blend : float
        0.0 = keep original SUNO voice, 1.0 = full user timbre.

    Returns
    -------
    np.ndarray
        Vocals with user's timbre, same length/pitch/timing as input.
    """
    import librosa

    # ── Extract user's average timbre (log-domain envelope) ────
    S_user = librosa.stft(user_audio, n_fft=n_fft, hop_length=hop_length)
    mag_user = np.abs(S_user)

    # Only use loud frames
    frame_energy = np.sum(mag_user ** 2, axis=0)
    threshold = np.median(frame_energy)
    loud = frame_energy > threshold
    if np.sum(loud) < 5:
        loud = np.ones(mag_user.shape[1], dtype=bool)

    # Average log-envelope across loud frames
    log_user_mag = np.log(np.maximum(mag_user[:, loud], 1e-10))
    user_envelopes = np.zeros_like(log_user_mag)
    for i in range(log_user_mag.shape[1]):
        user_envelopes[:, i] = _cepstral_envelope(log_user_mag[:, i], order)
    user_avg_env = np.mean(user_envelopes, axis=1)  # (n_freq,)

    # ── Process SUNO vocals frame by frame ─────────────────────
    S_suno = librosa.stft(suno_vocals, n_fft=n_fft, hop_length=hop_length)
    mag_suno = np.abs(S_suno)
    phase_suno = np.angle(S_suno)

    log_suno = np.log(np.maximum(mag_suno, 1e-10))
    n_freq, n_frames = log_suno.shape
    log_output = np.zeros_like(log_suno)

    for i in range(n_frames):
        # Separate this frame into envelope + residual
        suno_env = _cepstral_envelope(log_suno[:, i], order)
        residual = log_suno[:, i] - suno_env  # pitch harmonics

        # Blend between SUNO envelope and user envelope
        new_env = (1.0 - blend) * suno_env + blend * user_avg_env

        # Recombine: user timbre + SUNO pitch
        log_output[:, i] = new_env + residual

    # Convert back to magnitude
    new_mag = np.exp(log_output)

    # Reconstruct with SUNO's original phase
    S_out = new_mag * np.exp(1j * phase_suno)
    out = librosa.istft(S_out, hop_length=hop_length, length=len(suno_vocals))

    return out.astype(np.float32)
