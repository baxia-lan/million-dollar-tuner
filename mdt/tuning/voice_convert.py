"""Voice conversion using WORLD vocoder.

WORLD is a mature speech analysis-synthesis system. It decomposes
speech into three independent components:
  1. F0 (pitch contour)
  2. Spectral envelope (timbre / vocal tract shape)
  3. Aperiodicity (breathiness / noise content)

To convert voice: keep SUNO's F0 + aperiodicity, replace spectral
envelope with user's. This is the standard approach in speech synthesis
research and actually works on real audio.
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

    Parameters
    ----------
    suno_vocals : np.ndarray
        Separated SUNO vocal track (mono, float64 or float32).
    user_audio : np.ndarray
        User's voice recording (mono). Any content — speaking or singing.
    sr : int
        Sample rate.

    Returns
    -------
    np.ndarray
        SUNO vocals resynthesized with user's voice timbre.
        Same pitch, timing, rhythm as input.
    """
    import pyworld as pw

    # WORLD requires float64
    suno_f64 = suno_vocals.astype(np.float64)
    user_f64 = user_audio.astype(np.float64)

    # ── Analyze SUNO vocals ────────────────────────────────────
    # F0: pitch contour (what we KEEP)
    f0_suno, t_suno = pw.harvest(suno_f64, sr)
    # Spectral envelope: timbre (what we REPLACE)
    sp_suno = pw.cheaptrick(suno_f64, f0_suno, t_suno, sr)
    # Aperiodicity: breathiness/noise (what we KEEP)
    ap_suno = pw.d4c(suno_f64, f0_suno, t_suno, sr)

    # ── Extract user's average timbre ──────────────────────────
    f0_user, t_user = pw.harvest(user_f64, sr)
    sp_user = pw.cheaptrick(user_f64, f0_user, t_user, sr)

    # Only use voiced frames (where user is actually speaking/singing)
    voiced = f0_user > 0
    if np.sum(voiced) < 5:
        # Not enough voiced frames — use all non-silent frames
        frame_energy = np.sum(sp_user, axis=1)
        voiced = frame_energy > np.median(frame_energy)

    user_avg_sp = np.mean(sp_user[voiced], axis=0)  # average timbre

    # ── Replace spectral envelope ──────────────────────────────
    sp_converted = np.zeros_like(sp_suno)
    for i in range(len(sp_suno)):
        suno_energy = np.sum(sp_suno[i])
        if suno_energy < 1e-20:
            # Silent frame — keep silent
            sp_converted[i] = sp_suno[i]
        else:
            # Replace timbre shape, match energy level
            user_energy = np.sum(user_avg_sp) + 1e-20
            sp_converted[i] = user_avg_sp * (suno_energy / user_energy)

    # ── Synthesize with WORLD ──────────────────────────────────
    output = pw.synthesize(f0_suno, sp_converted, ap_suno, sr)

    # Match length
    if len(output) > len(suno_vocals):
        output = output[:len(suno_vocals)]
    elif len(output) < len(suno_vocals):
        output = np.pad(output, (0, len(suno_vocals) - len(output)))

    return output.astype(np.float32)
