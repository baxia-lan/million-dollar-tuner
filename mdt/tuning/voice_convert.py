"""Voice conversion via LPC cross-synthesis.

LPC (Linear Predictive Coding) models the vocal tract as an all-pole filter.
To convert voice A → voice B:
  1. Extract A's excitation signal (pitch pulses) via inverse filtering
  2. Extract B's vocal tract filter (LPC coefficients)
  3. Pass A's excitation through B's filter
  Result: A's pitch/timing + B's vocal tract = B singing A's melody

IMPORTANT: LPC is numerically stable only at low sample rates (≤16kHz).
High sample rate audio is automatically downsampled for LPC processing,
then upsampled back to original rate.
"""

from __future__ import annotations

import numpy as np


# LPC processing sample rate (standard for speech coding, numerically stable)
_LPC_SR = 16000
_LPC_ORDER = 20


def convert_voice_timbre(
    suno_vocals: np.ndarray,
    user_audio: np.ndarray,
    sr: int = 44100,
    **kwargs,
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

    Returns
    -------
    np.ndarray
        SUNO vocals with user's voice timbre. Same length/pitch/timing.
    """
    import librosa

    # Downsample to 16kHz for stable LPC processing
    suno_16k = librosa.resample(suno_vocals, orig_sr=sr, target_sr=_LPC_SR)
    user_16k = librosa.resample(user_audio, orig_sr=sr, target_sr=_LPC_SR)

    # Do LPC cross-synthesis at 16kHz
    out_16k = _lpc_cross_synthesis(suno_16k, user_16k, _LPC_SR, _LPC_ORDER)

    # Upsample back to original rate
    output = librosa.resample(out_16k, orig_sr=_LPC_SR, target_sr=sr)

    # Match length exactly
    if len(output) > len(suno_vocals):
        output = output[:len(suno_vocals)]
    elif len(output) < len(suno_vocals):
        output = np.pad(output, (0, len(suno_vocals) - len(output)))

    # Match RMS to input
    rms_in = np.sqrt(np.mean(suno_vocals ** 2)) + 1e-20
    rms_out = np.sqrt(np.mean(output ** 2)) + 1e-20
    output *= rms_in / rms_out

    return output.astype(np.float32)


def _lpc_cross_synthesis(
    source: np.ndarray,
    target: np.ndarray,
    sr: int,
    order: int,
) -> np.ndarray:
    """LPC cross-synthesis at a stable sample rate.

    Takes source's excitation + target's vocal tract filter.
    """
    from scipy.signal import lfilter

    frame_len = int(sr * 0.025)  # 25ms frames
    hop_len = int(sr * 0.010)    # 10ms hop

    # Extract target's average vocal tract (LPC filter)
    target_lpc = _average_lpc(target, frame_len, hop_len, order)

    # Process source frame by frame
    n = len(source)
    output = np.zeros(n + frame_len, dtype=np.float64)
    window = np.hanning(frame_len)

    pos = 0
    while pos + frame_len <= n:
        frame = source[pos:pos + frame_len] * window
        energy = np.sum(frame ** 2)

        if energy < 1e-10:
            pos += hop_len
            continue

        # Inverse filter: extract excitation from source
        source_lpc = _lpc_auto(frame, order)
        excitation = lfilter(source_lpc, [1.0], frame)

        # Synthesis filter: pass excitation through target's vocal tract
        converted = lfilter([1.0], target_lpc, excitation)

        # Match frame energy
        conv_e = np.sum(converted ** 2) + 1e-20
        converted *= np.sqrt(energy / conv_e)

        output[pos:pos + frame_len] += converted * window
        pos += hop_len

    output = output[:n]

    # Global RMS match
    rms_in = np.sqrt(np.mean(source ** 2)) + 1e-20
    rms_out = np.sqrt(np.mean(output ** 2)) + 1e-20
    output *= rms_in / rms_out

    return output


def _lpc_auto(frame: np.ndarray, order: int) -> np.ndarray:
    """Compute LPC via autocorrelation + Levinson-Durbin."""
    n = len(frame)
    r = np.correlate(frame, frame, mode='full')[n - 1:n + order]

    if r[0] < 1e-10:
        a = np.zeros(order + 1)
        a[0] = 1.0
        return a

    # Levinson-Durbin
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]

    for i in range(1, order + 1):
        lam = -(np.sum(a[1:i] * r[i - 1:0:-1]) + r[i]) / (e + 1e-20)
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


def _average_lpc(
    audio: np.ndarray,
    frame_len: int,
    hop_len: int,
    order: int,
) -> np.ndarray:
    """Average LPC via autocorrelation averaging (guaranteed stable)."""
    window = np.hanning(frame_len)
    n = len(audio)

    autocorrs = []
    energies = []
    pos = 0
    while pos + frame_len <= n:
        frame = audio[pos:pos + frame_len] * window
        e = np.sum(frame ** 2)
        if e > 1e-8:
            r = np.correlate(frame, frame, mode='full')[frame_len - 1:frame_len + order]
            autocorrs.append(r)
            energies.append(e)
        pos += hop_len

    if not autocorrs:
        a = np.zeros(order + 1)
        a[0] = 1.0
        return a

    # Keep top 50% loudest frames
    pairs = sorted(zip(energies, autocorrs), reverse=True)
    keep = max(len(pairs) // 2, 3)
    avg_r = np.mean([ac for _, ac in pairs[:keep]], axis=0)

    return _lpc_auto_from_r(avg_r, order)


def _lpc_auto_from_r(r: np.ndarray, order: int) -> np.ndarray:
    """Levinson-Durbin from autocorrelation vector."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]

    if e < 1e-10:
        return a

    for i in range(1, order + 1):
        lam = -(np.sum(a[1:i] * r[i - 1:0:-1]) + r[i]) / (e + 1e-20)
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
