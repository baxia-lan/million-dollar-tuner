"""Voice conversion using WORLD vocoder with spectral transfer function.

Instead of replacing SUNO's spectral envelope with user's average
(which sounds robotic and barely audible), we compute the DIFFERENCE
between user's and SUNO's average spectral envelopes and apply it
as a "voice EQ" to each frame.

This preserves the natural frame-to-frame variation of SUNO's vocals
while shifting the overall timbre toward the user's voice.
"""

from __future__ import annotations

import numpy as np


def convert_voice_timbre(
    suno_vocals: np.ndarray,
    user_audio: np.ndarray,
    sr: int = 44100,
    **kwargs,
) -> np.ndarray:
    """Replace SUNO vocal timbre with user's timbre using WORLD vocoder.

    Algorithm:
    1. WORLD analysis: decompose both into F0 + spectral envelope + aperiodicity
    2. Compute spectral transfer function: ratio of user/suno average envelopes
    3. Apply transfer function to each SUNO frame (like a voice-specific EQ)
    4. Synthesize: SUNO's F0 + warped envelope + SUNO's aperiodicity
    """
    import pyworld as pw

    suno_f64 = suno_vocals.astype(np.float64)
    user_f64 = user_audio.astype(np.float64)

    # ── Analyze SUNO vocals ────────────────────────────────────
    f0_suno, t_suno = pw.harvest(suno_f64, sr)
    sp_suno = pw.cheaptrick(suno_f64, f0_suno, t_suno, sr)
    ap_suno = pw.d4c(suno_f64, f0_suno, t_suno, sr)

    # ── Analyze user voice ─────────────────────────────────────
    f0_user, t_user = pw.harvest(user_f64, sr)
    sp_user = pw.cheaptrick(user_f64, f0_user, t_user, sr)

    # ── Compute spectral transfer function ─────────────────────
    # Average spectral envelope of voiced frames only
    suno_voiced = f0_suno > 0
    user_voiced = f0_user > 0

    if np.sum(suno_voiced) < 5:
        suno_voiced = np.ones(len(f0_suno), dtype=bool)
    if np.sum(user_voiced) < 5:
        user_voiced = np.ones(len(f0_user), dtype=bool)

    suno_avg = np.mean(sp_suno[suno_voiced], axis=0)
    user_avg = np.mean(sp_user[user_voiced], axis=0)

    # Transfer function = user / suno (in each frequency bin)
    # This is the EQ curve that transforms SUNO's voice into user's voice
    transfer = user_avg / (suno_avg + 1e-20)

    # Apply transfer TWICE for stronger effect
    # (like applying the same EQ curve twice — pushes timbre further toward user)
    transfer = transfer * transfer

    # Clip after squaring (±30dB max per bin)
    transfer = np.clip(transfer, 0.03, 30.0)

    # ── Apply transfer function to each SUNO frame ─────────────
    sp_converted = sp_suno * transfer[np.newaxis, :]

    # ── Synthesize with WORLD ──────────────────────────────────
    output = pw.synthesize(f0_suno, sp_converted, ap_suno, sr)

    # Match length
    if len(output) > len(suno_vocals):
        output = output[:len(suno_vocals)]
    elif len(output) < len(suno_vocals):
        output = np.pad(output, (0, len(suno_vocals) - len(output)))

    return output.astype(np.float32)
