"""Voice timbre transfer: replace SUNO vocal timbre with the user's timbre.

Source-filter model:
- Source (pitch harmonics, timing) → keep from SUNO vocals
- Filter (formants, timbre) → take from USER's voice

Algorithm:
1. STFT both signals
2. Extract spectral envelope (smooth shape) from each via cepstral liftering
3. Transfer ratio: gain = user_envelope / suno_envelope (per-frame)
4. Apply gain to SUNO magnitude, keep SUNO phase
5. Reconstruct
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct


def extract_spectral_envelope(
    magnitude_spectrum: np.ndarray,
    n_cepstral: int = 40,
) -> np.ndarray:
    """Extract spectral envelope via cepstral smoothing.

    Parameters
    ----------
    magnitude_spectrum : np.ndarray
        Shape ``(n_freq,)`` or ``(n_freq, n_frames)``.
    n_cepstral : int
        Cepstral coefficients to keep. Lower = smoother.

    Returns
    -------
    np.ndarray
        Spectral envelope, same shape.
    """
    log_mag = np.log(np.maximum(magnitude_spectrum, 1e-10))

    if log_mag.ndim == 1:
        cep = dct(log_mag, type=2, norm="ortho")
        cep[n_cepstral:] = 0
        return np.exp(idct(cep, type=2, norm="ortho"))

    envelopes = np.zeros_like(log_mag)
    for i in range(log_mag.shape[1]):
        cep = dct(log_mag[:, i], type=2, norm="ortho")
        cep[n_cepstral:] = 0
        envelopes[:, i] = np.exp(idct(cep, type=2, norm="ortho"))
    return envelopes


def extract_user_timbre(
    user_audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_cepstral: int = 40,
) -> np.ndarray:
    """Extract the user's average spectral envelope (their timbre signature).

    Only uses loud frames (voice, not silence).

    Returns
    -------
    np.ndarray
        Shape ``(n_fft // 2 + 1,)`` — average spectral envelope.
    """
    import librosa

    S = librosa.stft(user_audio, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)

    # Keep only loud frames (top 50% energy)
    frame_energy = np.sum(mag ** 2, axis=0)
    threshold = np.median(frame_energy)
    loud = frame_energy > threshold

    if np.sum(loud) < 5:
        loud = np.ones(mag.shape[1], dtype=bool)

    env = extract_spectral_envelope(mag[:, loud], n_cepstral)
    avg = np.mean(env, axis=1)
    return avg


def convert_voice_timbre(
    suno_vocals: np.ndarray,
    user_audio: np.ndarray,
    sr: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_cepstral: int = 40,
    blend: float = 0.8,
) -> np.ndarray:
    """Replace SUNO vocal timbre with user's timbre.

    Keeps SUNO's pitch, timing, and rhythm exactly as-is.

    Parameters
    ----------
    suno_vocals : np.ndarray
        Separated SUNO vocal track (mono).
    user_audio : np.ndarray
        User's voice recording (mono).
    sr : int
        Sample rate.
    n_cepstral : int
        Envelope smoothness. Lower = more timbre change.
    blend : float
        0.0 = keep original, 1.0 = full user timbre.

    Returns
    -------
    np.ndarray
        Vocals with user's timbre.
    """
    import librosa

    # 1. Get user's average timbre envelope
    user_env = extract_user_timbre(
        user_audio, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_cepstral=n_cepstral,
    )

    # 2. STFT of SUNO vocals
    S = librosa.stft(suno_vocals, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)
    phase = np.angle(S)

    # 3. Per-frame: compute gain = user_env / suno_env
    suno_env = extract_spectral_envelope(mag, n_cepstral)

    # Gain ratio: how to reshape each frame's envelope to match user
    # gain > 1 in bands where user is louder, < 1 where user is quieter
    user_env_2d = user_env[:, np.newaxis]  # broadcast over frames
    gain = user_env_2d / np.maximum(suno_env, 1e-10)

    # Limit extreme gains to avoid artifacts
    gain = np.clip(gain, 0.1, 10.0)

    # Blend: interpolate gain toward 1.0 (no change)
    gain = 1.0 + blend * (gain - 1.0)

    # 4. Apply gain to magnitude, keep phase
    new_mag = mag * gain

    # 5. Preserve original energy per frame
    # Scale so each frame's total energy matches the original
    orig_energy = np.sum(mag ** 2, axis=0, keepdims=True) + 1e-10
    new_energy = np.sum(new_mag ** 2, axis=0, keepdims=True) + 1e-10
    new_mag = new_mag * np.sqrt(orig_energy / new_energy)

    # 6. Reconstruct
    S_out = new_mag * np.exp(1j * phase)
    out = librosa.istft(S_out, hop_length=hop_length, length=len(suno_vocals))

    return out.astype(np.float32)
