"""Voice conversion: piecewise formant warp + per-bin statistical mapping.

Two-stage approach:
1. Piecewise frequency warping — independently repositions each formant
   peak from SUNO's position to user's position
2. Per-bin statistical mapping (log domain) — adjusts energy distribution
   to match user's mean spectral shape and dynamic range

Stage 1 fixes formant POSITIONS (which per-bin mapping alone cannot do).
Stage 2 fixes energy LEVELS and spectral shape.
"""

from __future__ import annotations

import numpy as np


def convert_voice_timbre(
    suno_vocals: np.ndarray,
    user_audio: np.ndarray,
    sr: int = 44100,
    **kwargs,
) -> np.ndarray:
    """Replace SUNO vocal timbre with user's timbre."""
    import pyworld as pw
    from scipy.ndimage import uniform_filter1d

    suno_f64 = suno_vocals.astype(np.float64)
    user_f64 = user_audio.astype(np.float64)

    # ── WORLD analysis ────────────────────────────────────────
    f0_suno, t_suno = pw.harvest(suno_f64, sr)
    sp_suno = pw.cheaptrick(suno_f64, f0_suno, t_suno, sr)
    ap_suno = pw.d4c(suno_f64, f0_suno, t_suno, sr)

    f0_user, t_user = pw.harvest(user_f64, sr)
    sp_user = pw.cheaptrick(user_f64, f0_user, t_user, sr)
    ap_user = pw.d4c(user_f64, f0_user, t_user, sr)

    # ── Voiced frame masks ────────────────────────────────────
    suno_voiced = f0_suno > 0
    user_voiced = f0_user > 0
    if np.sum(suno_voiced) < 10:
        suno_voiced[:] = True
    if np.sum(user_voiced) < 10:
        user_voiced[:] = True

    n_freq = sp_suno.shape[1]
    freqs = np.linspace(0, sr / 2, n_freq)
    eps = 1e-16

    # ── Stage 1: Piecewise formant warping ────────────────────
    avg_sp_suno = np.mean(sp_suno[suno_voiced], axis=0)
    avg_sp_user = np.mean(sp_user[user_voiced], axis=0)

    suno_formants = _detect_formants(avg_sp_suno, freqs)
    user_formants = _detect_formants(avg_sp_user, freqs)

    sp_warped = _apply_piecewise_warp(sp_suno, freqs, suno_formants, user_formants)

    # ── Stage 2: Per-bin statistical mapping ──────────────────
    log_sp_warped = np.log(sp_warped + eps)
    log_sp_user = np.log(sp_user + eps)

    w_mean = np.mean(log_sp_warped[suno_voiced], axis=0)
    w_std = np.maximum(np.std(log_sp_warped[suno_voiced], axis=0), 1e-6)

    u_mean = np.mean(log_sp_user[user_voiced], axis=0)
    u_std = np.maximum(np.std(log_sp_user[user_voiced], axis=0), 1e-6)

    z = (log_sp_warped - w_mean[np.newaxis, :]) / w_std[np.newaxis, :]
    log_sp_out = u_mean[np.newaxis, :] + z * u_std[np.newaxis, :]

    # Temporal smoothing
    log_sp_out = uniform_filter1d(log_sp_out, size=3, axis=0)
    sp_out = np.exp(log_sp_out)

    # ── Aperiodicity transfer ─────────────────────────────────
    user_ap_mean = np.mean(ap_user[user_voiced], axis=0)
    suno_ap_mean = np.mean(ap_suno[suno_voiced], axis=0)

    ap_variation = ap_suno - suno_ap_mean[np.newaxis, :]
    ap_out = user_ap_mean[np.newaxis, :] + 0.3 * ap_variation
    ap_out = np.clip(ap_out, 0.0, 1.0)

    # ── Synthesize ────────────────────────────────────────────
    output = pw.synthesize(f0_suno, sp_out, ap_out, sr)

    target_len = len(suno_vocals)
    if len(output) > target_len:
        output = output[:target_len]
    elif len(output) < target_len:
        output = np.pad(output, (0, target_len - len(output)))

    return output.astype(np.float32)


def _detect_formants(avg_sp: np.ndarray, freqs: np.ndarray, n_formants: int = 3) -> list[float]:
    """Detect formant peak frequencies from average spectral envelope."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks

    log_sp = np.log(avg_sp + 1e-16)
    log_sp_smooth = uniform_filter1d(log_sp, size=5)

    bin_spacing = freqs[1] - freqs[0]
    min_distance = max(1, int(200 / bin_spacing))

    peaks, props = find_peaks(log_sp_smooth, distance=min_distance, prominence=0.2)

    voice_peaks = [(freqs[p], props["prominences"][i])
                   for i, p in enumerate(peaks) if 200 < freqs[p] < 5000]

    voice_peaks.sort(key=lambda x: x[1], reverse=True)
    formants = sorted([f for f, _ in voice_peaks[:n_formants]])
    return formants


def _apply_piecewise_warp(
    sp: np.ndarray,
    freqs: np.ndarray,
    source_formants: list[float],
    target_formants: list[float],
) -> np.ndarray:
    """Apply piecewise-linear frequency warp to spectral envelope."""
    max_freq = freqs[-1]

    src = [0.0] + list(source_formants) + [max_freq]
    tgt = [0.0] + list(target_formants) + [max_freq]

    while len(tgt) < len(src):
        idx = len(tgt) - 1
        tgt.insert(idx, src[idx])
    while len(src) < len(tgt):
        idx = len(src) - 1
        src.insert(idx, tgt[idx])

    warped_freqs = np.interp(freqs, src, tgt)

    sp_warped = np.zeros_like(sp)
    for i in range(len(sp)):
        sp_warped[i] = np.interp(freqs, warped_freqs, sp[i],
                                  left=sp[i, 0], right=sp[i, -1])

    return sp_warped
