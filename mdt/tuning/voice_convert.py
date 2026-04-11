"""Voice conversion using WORLD vocoder with frequency warping.

Instead of applying an EQ (which barely changes perceived voice identity),
we WARP the spectral envelope along the frequency axis — physically moving
the formant peaks to different positions. This is equivalent to changing
the vocal tract length, which is the primary physical difference between
voices.

EQ:      boosts/cuts energy at fixed positions → subtle, same voice
Warping: MOVES resonant peaks to new positions → clearly different voice
"""

from __future__ import annotations

import numpy as np


def convert_voice_timbre(
    suno_vocals: np.ndarray,
    user_audio: np.ndarray,
    sr: int = 44100,
    **kwargs,
) -> np.ndarray:
    """Replace SUNO vocal timbre with user's timbre.

    Uses WORLD vocoder + spectral envelope frequency warping.
    Formant peaks are physically moved to match user's voice.
    """
    import pyworld as pw

    suno_f64 = suno_vocals.astype(np.float64)
    user_f64 = user_audio.astype(np.float64)

    # ── Analyze both voices ────────────────────────────────────
    f0_suno, t_suno = pw.harvest(suno_f64, sr)
    sp_suno = pw.cheaptrick(suno_f64, f0_suno, t_suno, sr)
    ap_suno = pw.d4c(suno_f64, f0_suno, t_suno, sr)

    f0_user, t_user = pw.harvest(user_f64, sr)
    sp_user = pw.cheaptrick(user_f64, f0_user, t_user, sr)

    # ── Compute frequency warping ratio ────────────────────────
    # Use spectral centroid of voiced frames to determine how much
    # to stretch/compress the frequency axis
    suno_voiced = f0_suno > 0
    user_voiced = f0_user > 0

    if np.sum(suno_voiced) < 5:
        suno_voiced = np.ones(len(f0_suno), dtype=bool)
    if np.sum(user_voiced) < 5:
        user_voiced = np.ones(len(f0_user), dtype=bool)

    suno_avg = np.mean(sp_suno[suno_voiced], axis=0)
    user_avg = np.mean(sp_user[user_voiced], axis=0)

    # Weighted centroid of spectral envelopes
    freqs = np.arange(len(suno_avg), dtype=np.float64)
    suno_centroid = np.sum(freqs * suno_avg) / (np.sum(suno_avg) + 1e-20)
    user_centroid = np.sum(freqs * user_avg) / (np.sum(user_avg) + 1e-20)

    # Warp ratio: < 1 shifts formants down, > 1 shifts up
    warp_ratio = user_centroid / (suno_centroid + 1e-20)
    warp_ratio = np.clip(warp_ratio, 0.5, 2.0)

    n_freq = sp_suno.shape[1]

    # ── Warp spectral envelope of each frame ───────────────────
    sp_warped = np.zeros_like(sp_suno)
    old_indices = np.arange(n_freq, dtype=np.float64)

    # Also compute EQ transfer for additional timbre matching
    transfer = user_avg / (suno_avg + 1e-20)
    transfer = np.clip(transfer, 0.1, 10.0)

    for i in range(len(sp_suno)):
        # Step 1: Frequency warp (moves formant positions)
        # Map old frequency bins to new positions
        new_indices = old_indices * warp_ratio
        # Interpolate: resample the spectral envelope along warped axis
        warped = np.interp(old_indices, new_indices, sp_suno[i],
                           left=sp_suno[i, 0], right=sp_suno[i, -1])

        # Step 2: Apply residual EQ transfer (fine-tune timbre)
        warped = warped * transfer

        sp_warped[i] = warped

    # ── Also warp the aperiodicity ─────────────────────────────
    ap_warped = np.zeros_like(ap_suno)
    for i in range(len(ap_suno)):
        new_indices = old_indices * warp_ratio
        ap_warped[i] = np.interp(old_indices, new_indices, ap_suno[i],
                                  left=ap_suno[i, 0], right=ap_suno[i, -1])

    # ── Synthesize ─────────────────────────────────────────────
    output = pw.synthesize(f0_suno, sp_warped, ap_warped, sr)

    # Match length
    if len(output) > len(suno_vocals):
        output = output[:len(suno_vocals)]
    elif len(output) < len(suno_vocals):
        output = np.pad(output, (0, len(suno_vocals) - len(output)))

    return output.astype(np.float32)
