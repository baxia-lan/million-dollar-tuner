"""Voice conversion via LPC (Linear Predictive Coding).

LPC models the vocal tract as an all-pole filter:
  speech = excitation * vocal_tract_filter

To convert voice A to voice B:
  1. Analyze A: extract excitation (pitch pulses) and LPC filter (vocal tract)
  2. Analyze B: extract LPC filter (vocal tract)
  3. Synthesize: pass A's excitation through B's filter
  Result: A's pitch/timing + B's vocal tract = B's voice singing A's melody

This is fundamentally different from spectral envelope methods because
LPC explicitly models the resonant structure of the vocal tract
(formants), producing a clearly audible voice change.
"""

from __future__ import annotations

import numpy as np


def _lpc_coefficients(frame: np.ndarray, order: int) -> np.ndarray:
    """Compute LPC coefficients using autocorrelation method (Levinson-Durbin)."""
    from scipy.signal import lfilter

    # Autocorrelation
    n = len(frame)
    r = np.correlate(frame, frame, mode='full')[n - 1:n + order]

    if r[0] == 0:
        return np.zeros(order + 1)

    # Levinson-Durbin recursion
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]

    for i in range(1, order + 1):
        lam = -np.sum(a[1:i] * r[i - 1:0:-1]) - r[i]
        lam /= (e + 1e-20)
        lam = np.clip(lam, -0.999, 0.999)  # stability

        # Update coefficients
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + lam * a[i - j]
        a_new[i] = lam
        a = a_new

        e *= (1 - lam * lam)
        if e <= 0:
            break

    return a


def _lpc_residual(frame: np.ndarray, lpc: np.ndarray) -> np.ndarray:
    """Get the excitation signal by inverse-filtering with LPC coefficients."""
    from scipy.signal import lfilter
    # Inverse filter: A(z) * speech = excitation
    residual = lfilter(lpc, [1.0], frame)
    return residual


def _lpc_synthesize(residual: np.ndarray, lpc: np.ndarray) -> np.ndarray:
    """Synthesize speech by filtering excitation through LPC vocal tract."""
    from scipy.signal import lfilter
    # Synthesis filter: speech = 1/A(z) * excitation
    return lfilter([1.0], lpc, residual)


def convert_voice_timbre(
    suno_vocals: np.ndarray,
    user_audio: np.ndarray,
    sr: int = 44100,
    frame_ms: int = 25,
    hop_ms: int = 10,
    lpc_order: int = 40,
) -> np.ndarray:
    """Replace SUNO vocal timbre with user's timbre using LPC cross-synthesis.

    Parameters
    ----------
    suno_vocals : np.ndarray
        Separated SUNO vocal track (mono, float32).
    user_audio : np.ndarray
        User's voice recording (mono, float32).
    sr : int
        Sample rate.
    frame_ms : int
        Frame length in milliseconds.
    hop_ms : int
        Hop size in milliseconds.
    lpc_order : int
        LPC order. Higher = more detailed vocal tract model.
        24 is good for 44.1kHz (captures formants F1-F5).

    Returns
    -------
    np.ndarray
        SUNO vocals resynthesized with user's vocal tract.
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    # ── Extract user's average LPC filter ──────────────────────
    user_lpc = _extract_average_lpc(user_audio, sr, frame_len, hop_len, lpc_order)

    # ── Frame-by-frame cross-synthesis ─────────────────────────
    n_samples = len(suno_vocals)
    output = np.zeros(n_samples + frame_len, dtype=np.float64)
    window = np.hanning(frame_len)

    # Overlap-add synthesis
    pos = 0
    while pos + frame_len <= n_samples:
        frame = suno_vocals[pos:pos + frame_len] * window

        # Skip silent frames
        energy = np.sum(frame ** 2)
        if energy < 1e-10:
            pos += hop_len
            continue

        # 1. Get SUNO's excitation (inverse filter with SUNO's LPC)
        suno_lpc = _lpc_coefficients(frame, lpc_order)
        excitation = _lpc_residual(frame, suno_lpc)

        # 2. Resynthesize with USER's vocal tract filter
        converted = _lpc_synthesize(excitation, user_lpc)

        # 3. Match energy of original frame
        conv_energy = np.sum(converted ** 2) + 1e-20
        converted *= np.sqrt(energy / conv_energy)

        # Overlap-add
        output[pos:pos + frame_len] += converted * window
        pos += hop_len

    output = output[:n_samples]

    # Match overall RMS to input (guarantees same volume)
    rms_in = np.sqrt(np.mean(suno_vocals ** 2)) + 1e-20
    rms_out = np.sqrt(np.mean(output ** 2)) + 1e-20
    output *= rms_in / rms_out

    return output.astype(np.float32)


def _extract_average_lpc(
    audio: np.ndarray,
    sr: int,
    frame_len: int,
    hop_len: int,
    lpc_order: int,
) -> np.ndarray:
    """Extract average LPC from voiced frames via autocorrelation averaging.

    Instead of averaging LPC coefficients directly (which can produce
    unstable filters), we average the autocorrelation vectors and then
    solve for LPC once. This guarantees a stable filter.
    """
    window = np.hanning(frame_len)
    n_samples = len(audio)

    # Collect autocorrelation from loud frames
    all_energies = []
    all_autocorrs = []
    pos = 0
    while pos + frame_len <= n_samples:
        frame = audio[pos:pos + frame_len] * window
        energy = np.sum(frame ** 2)
        if energy > 1e-8:
            r = np.correlate(frame, frame, mode='full')[frame_len - 1:frame_len + lpc_order]
            all_energies.append(energy)
            all_autocorrs.append(r)
        pos += hop_len

    if not all_autocorrs:
        lpc = np.zeros(lpc_order + 1)
        lpc[0] = 1.0
        return lpc

    # Keep top 50% loudest
    pairs = sorted(zip(all_energies, all_autocorrs), reverse=True)
    keep = max(len(pairs) // 2, 3)
    loud_autocorrs = [ac for _, ac in pairs[:keep]]

    # Average autocorrelation → solve for LPC
    avg_r = np.mean(loud_autocorrs, axis=0)

    # Construct a "virtual" frame whose autocorrelation = avg_r
    # Then compute LPC on it. Since LPC only uses autocorrelation,
    # we can feed avg_r directly into Levinson-Durbin.
    lpc = _levinson_durbin(avg_r, lpc_order)
    return lpc


def _levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
    """Solve for LPC coefficients from autocorrelation using Levinson-Durbin."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]

    if e <= 0:
        return a

    for i in range(1, order + 1):
        lam = -np.sum(a[1:i] * r[i - 1:0:-1]) - r[i]
        lam /= (e + 1e-20)
        lam = np.clip(lam, -0.999, 0.999)

        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + lam * a[i - j]
        a_new[i] = lam
        a = a_new

        e *= (1 - lam * lam)
        if e <= 0:
            break

    return a
