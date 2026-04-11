"""Voice timbre transfer: replace SUNO vocal timbre with the user's timbre.

Uses the source-filter model of speech:
- Source (excitation) = pitch harmonics, timing, rhythm → keep from SUNO
- Filter (vocal tract) = formants, timbre, tone color → take from USER

Algorithm (cepstral spectral envelope transfer):
1. STFT both the SUNO vocals and the user's voice sample
2. Extract spectral envelope from each via cepstral smoothing
3. For each frame of SUNO vocals:
   - fine_structure = suno_magnitude / suno_envelope  (pitch + rhythm)
   - new_magnitude = user_envelope * fine_structure    (user timbre + suno content)
4. Reconstruct with SUNO's original phase
5. Result: sounds like the user singing with SUNO's exact pitch and timing
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct

from mdt.config import OUTPUT_SR


def extract_spectral_envelope(
    magnitude_spectrum: np.ndarray,
    n_cepstral: int = 60,
) -> np.ndarray:
    """Extract spectral envelope from a magnitude spectrum via cepstral smoothing.

    Parameters
    ----------
    magnitude_spectrum : np.ndarray
        Magnitude spectrum, shape ``(n_freq,)`` or ``(n_freq, n_frames)``.
    n_cepstral : int
        Number of cepstral coefficients to keep. Controls smoothness.
        Lower = smoother envelope (more timbre, less pitch detail).

    Returns
    -------
    np.ndarray
        Spectral envelope, same shape as input.
    """
    # Log magnitude (avoid log(0))
    log_mag = np.log(magnitude_spectrum + 1e-10)

    if log_mag.ndim == 1:
        # Single frame
        cepstrum = dct(log_mag, type=2, norm="ortho")
        cepstrum[n_cepstral:] = 0  # Lifter: keep only low quefrency
        envelope = np.exp(idct(cepstrum, type=2, norm="ortho"))
        return envelope

    # Multiple frames: process each column
    envelopes = np.zeros_like(log_mag)
    for i in range(log_mag.shape[1]):
        cepstrum = dct(log_mag[:, i], type=2, norm="ortho")
        cepstrum[n_cepstral:] = 0
        envelopes[:, i] = np.exp(idct(cepstrum, type=2, norm="ortho"))
    return envelopes


def extract_user_timbre(
    user_audio: np.ndarray,
    sr: int = OUTPUT_SR,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_cepstral: int = 60,
    energy_threshold: float = 0.01,
) -> np.ndarray:
    """Extract the user's average voice timbre as a spectral envelope.

    Only uses voiced/loud frames to avoid capturing silence/noise.

    Parameters
    ----------
    user_audio : np.ndarray
        User's voice recording (mono).
    sr : int
        Sample rate.
    n_fft : int
        FFT size.
    hop_length : int
        Hop size.
    n_cepstral : int
        Cepstral smoothing order.
    energy_threshold : float
        Minimum frame energy to include (filters out silence).

    Returns
    -------
    np.ndarray
        Average spectral envelope, shape ``(n_fft // 2 + 1,)``.
    """
    import librosa

    # Compute STFT
    S = librosa.stft(user_audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(S)  # (n_freq, n_frames)

    # Filter out silent frames
    frame_energy = np.mean(magnitude ** 2, axis=0)
    threshold = energy_threshold * np.max(frame_energy)
    voiced_mask = frame_energy > threshold

    if np.sum(voiced_mask) < 5:
        # Not enough voiced frames, use all
        voiced_mask = np.ones(magnitude.shape[1], dtype=bool)

    voiced_magnitude = magnitude[:, voiced_mask]

    # Extract envelope for each voiced frame
    envelopes = extract_spectral_envelope(voiced_magnitude, n_cepstral)

    # Average across frames to get the user's characteristic timbre
    avg_envelope = np.mean(envelopes, axis=1)

    # Normalize so it doesn't change overall energy
    avg_envelope = avg_envelope / (np.mean(avg_envelope) + 1e-10)

    return avg_envelope


def convert_voice_timbre(
    suno_vocals: np.ndarray,
    user_audio: np.ndarray,
    sr: int = OUTPUT_SR,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_cepstral: int = 60,
    blend: float = 0.8,
) -> np.ndarray:
    """Replace the timbre of SUNO vocals with the user's timbre.

    Keeps SUNO's pitch, timing, and rhythm exactly as-is.
    Only changes the vocal tone color to sound like the user.

    Parameters
    ----------
    suno_vocals : np.ndarray
        Separated SUNO vocal track (mono).
    user_audio : np.ndarray
        User's voice recording (mono). Used only to extract timbre.
    sr : int
        Sample rate (both signals must match).
    n_fft : int
        FFT size.
    hop_length : int
        Hop size.
    n_cepstral : int
        Cepstral smoothing order. Lower = more timbre transfer.
        Typical range: 30-80. 60 is a good default.
    blend : float
        Blend between original timbre (0.0) and user timbre (1.0).
        0.0 = no change, 1.0 = full replacement.

    Returns
    -------
    np.ndarray
        Vocals with user's timbre, same length and timing as suno_vocals.
    """
    import librosa

    # 1. Extract user's average timbre
    user_envelope = extract_user_timbre(
        user_audio, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_cepstral=n_cepstral,
    )

    # 2. STFT of SUNO vocals
    S_suno = librosa.stft(suno_vocals, n_fft=n_fft, hop_length=hop_length)
    magnitude_suno = np.abs(S_suno)
    phase_suno = np.angle(S_suno)

    # 3. Extract SUNO's per-frame spectral envelope
    suno_envelope = extract_spectral_envelope(magnitude_suno, n_cepstral)

    # 4. Compute fine structure (pitch harmonics + timing)
    # fine_structure = magnitude / envelope
    fine_structure = magnitude_suno / (suno_envelope + 1e-10)

    # 5. Build new magnitude: user timbre envelope * SUNO fine structure
    # user_envelope is (n_freq,), broadcast over frames
    user_env_2d = user_envelope[:, np.newaxis]

    # Blend between original and user envelope
    blended_envelope = (1 - blend) * suno_envelope + blend * user_env_2d

    new_magnitude = blended_envelope * fine_structure

    # 6. Reconstruct with original phase
    S_converted = new_magnitude * np.exp(1j * phase_suno)

    # 7. ISTFT
    converted = librosa.istft(
        S_converted, hop_length=hop_length, length=len(suno_vocals)
    )

    return converted.astype(np.float32)
