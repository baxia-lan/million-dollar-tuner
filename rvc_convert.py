#!/usr/bin/env python3
"""Voice conversion using enhanced WORLD vocoder + resemblyzer evaluation.

rvc-python requires a pre-trained .pth model and cannot train from raw audio.
It also fails to install on Python 3.14 (numpy/fairseq/av build failures).

This script instead uses the project's existing WORLD-based voice conversion
(formant warping + statistical spectral mapping) and evaluates the result
with resemblyzer speaker embeddings and F0 correlation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Project root
PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from mdt.audio.io import load_audio, save_audio
from mdt.audio.convert import match_rms
from mdt.config import OUTPUT_SR
from mdt.tuning.voice_convert import convert_voice_timbre


# ── Paths ─────────────────────────────────────────────────────
USER_RECORDING = Path("/Users/sarsa/Downloads/recording.wav")
SUNO_VOCALS = PROJECT / "stems_dtr" / "vocals.wav"
STEMS = {
    "drums": PROJECT / "stems_dtr" / "drums.wav",
    "bass": PROJECT / "stems_dtr" / "bass.wav",
    "other": PROJECT / "stems_dtr" / "other.wav",
}
OUT_VOCALS = Path("/Users/sarsa/Downloads/rvc_vocals.wav")
OUT_MIX = Path("/Users/sarsa/Downloads/rvc_mix.wav")


def compute_resemblyzer_similarity(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sr: int,
    label_a: str = "A",
    label_b: str = "B",
) -> float:
    """Compute cosine similarity between speaker embeddings."""
    from resemblyzer import VoiceEncoder, preprocess_wav
    import librosa

    encoder = VoiceEncoder("cpu")

    # Resemblyzer expects 16kHz mono
    if sr != 16000:
        a_16k = librosa.resample(audio_a.astype(np.float32), orig_sr=sr, target_sr=16000)
        b_16k = librosa.resample(audio_b.astype(np.float32), orig_sr=sr, target_sr=16000)
    else:
        a_16k = audio_a.astype(np.float32)
        b_16k = audio_b.astype(np.float32)

    # preprocess_wav normalizes and trims silence
    a_proc = preprocess_wav(a_16k, source_sr=16000)
    b_proc = preprocess_wav(b_16k, source_sr=16000)

    embed_a = encoder.embed_utterance(a_proc)
    embed_b = encoder.embed_utterance(b_proc)

    cos_sim = float(np.dot(embed_a, embed_b) / (
        np.linalg.norm(embed_a) * np.linalg.norm(embed_b) + 1e-10
    ))
    return cos_sim


def compute_f0_correlation(
    source: np.ndarray,
    converted: np.ndarray,
    sr: int,
) -> float:
    """Compute F0 correlation between source and converted vocals.

    High correlation means the pitch contour was preserved during conversion.
    """
    import pyworld as pw

    src_f64 = source.astype(np.float64)
    cvt_f64 = converted.astype(np.float64)

    f0_src, _ = pw.harvest(src_f64, sr)
    f0_cvt, _ = pw.harvest(cvt_f64, sr)

    # Align lengths
    min_len = min(len(f0_src), len(f0_cvt))
    f0_src = f0_src[:min_len]
    f0_cvt = f0_cvt[:min_len]

    # Only compare where both are voiced
    both_voiced = (f0_src > 0) & (f0_cvt > 0)
    if np.sum(both_voiced) < 10:
        return 0.0

    corr = np.corrcoef(f0_src[both_voiced], f0_cvt[both_voiced])[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def mix_stems_with_vocals(
    vocals: np.ndarray,
    vocals_sr: int,
    stem_paths: dict[str, Path],
    reference_vocals: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Mix converted vocals with instrumental stems."""
    import librosa
    from mdt.audio.effects import apply_effects, build_vocal_chain

    sr = vocals_sr

    # Level-match to reference
    vocals = match_rms(vocals, reference_vocals)

    # Apply vocal effects chain (subtle reverb + compression)
    board = build_vocal_chain(reverb_room=0.3, reverb_wet=0.15)
    vocals = apply_effects(vocals, sr, board=board)

    # Load and sum instrumental stems
    instrumental = None
    for name, path in stem_paths.items():
        stem, stem_sr = load_audio(path, sr=sr, mono=True)
        if instrumental is None:
            instrumental = stem
        else:
            min_len = min(len(instrumental), len(stem))
            instrumental = instrumental[:min_len] + stem[:min_len]

    # Match lengths
    target_len = len(instrumental)
    if len(vocals) > target_len:
        vocals = vocals[:target_len]
    elif len(vocals) < target_len:
        vocals = np.pad(vocals, (0, target_len - len(vocals)))

    # Mix
    mix = vocals + instrumental

    # Prevent clipping
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix = mix / peak * 0.95

    return mix, sr


def main():
    print("=" * 60)
    print("Voice Conversion: WORLD vocoder + Resemblyzer evaluation")
    print("=" * 60)

    # ── RVC-python feasibility report ─────────────────────────
    print("\n--- rvc-python feasibility ---")
    print("  Package: rvc-python 0.1.5 (installed, no-deps)")
    print("  Status: INFERENCE ONLY - requires pre-trained .pth model file")
    print("  Training: NOT supported (only preprocessing script exists,")
    print("            full training needs RVC WebUI repository)")
    print("  Zero-shot: NOT supported (no reference-based conversion)")
    print("  Python 3.14: BROKEN (numpy, av, fairseq, faiss-cpu fail to build)")
    print("  Conclusion: Cannot use rvc-python without a pre-trained voice model.")
    print("  Falling back to WORLD vocoder-based voice conversion.\n")

    # ── Load audio ────────────────────────────────────────────
    print("[1/5] Loading audio files...")
    user_audio, user_sr = load_audio(USER_RECORDING, sr=OUTPUT_SR, mono=True)
    suno_vocals, suno_sr = load_audio(SUNO_VOCALS, sr=OUTPUT_SR, mono=True)
    print(f"  User recording: {len(user_audio)/OUTPUT_SR:.1f}s @ {OUTPUT_SR}Hz")
    print(f"  SUNO vocals:    {len(suno_vocals)/OUTPUT_SR:.1f}s @ {OUTPUT_SR}Hz")

    # ── Voice conversion ──────────────────────────────────────
    print("\n[2/5] Converting vocals (WORLD vocoder: formant warp + spectral mapping)...")
    converted = convert_voice_timbre(
        suno_vocals=suno_vocals,
        user_audio=user_audio,
        sr=OUTPUT_SR,
    )
    print(f"  Converted vocals: {len(converted)/OUTPUT_SR:.1f}s")

    # ── Save converted vocals ─────────────────────────────────
    print(f"\n[3/5] Saving converted vocals to {OUT_VOCALS}...")
    save_audio(OUT_VOCALS, converted, OUTPUT_SR)
    print("  Done.")

    # ── Mix with instrumentals ────────────────────────────────
    print(f"\n[4/5] Mixing with instrumental stems -> {OUT_MIX}...")
    mix, mix_sr = mix_stems_with_vocals(
        vocals=converted,
        vocals_sr=OUTPUT_SR,
        stem_paths=STEMS,
        reference_vocals=suno_vocals,
    )
    save_audio(OUT_MIX, mix, mix_sr)
    print("  Done.")

    # ── Evaluation ────────────────────────────────────────────
    print("\n[5/5] Evaluating with resemblyzer speaker embeddings + F0 correlation...")

    # Speaker similarity: converted vs user (want HIGH)
    sim_conv_user = compute_resemblyzer_similarity(
        converted, user_audio, OUTPUT_SR,
        label_a="converted", label_b="user"
    )

    # Speaker similarity: converted vs SUNO (want LOW if voice changed)
    sim_conv_suno = compute_resemblyzer_similarity(
        converted, suno_vocals, OUTPUT_SR,
        label_a="converted", label_b="SUNO"
    )

    # Baseline: SUNO vs user (the gap we're trying to bridge)
    sim_suno_user = compute_resemblyzer_similarity(
        suno_vocals, user_audio, OUTPUT_SR,
        label_a="SUNO", label_b="user"
    )

    # F0 correlation: should be high (preserving melody)
    print("  Computing F0 correlation (this takes a moment)...")
    f0_corr = compute_f0_correlation(suno_vocals, converted, OUTPUT_SR)

    # ── Report ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  Output files:")
    print(f"    Converted vocals: {OUT_VOCALS}")
    print(f"    Full mix:         {OUT_MIX}")

    print(f"\n  Resemblyzer speaker embeddings (cosine similarity):")
    print(f"    converted <-> user:  {sim_conv_user:.4f}  (want HIGH - voice should match user)")
    print(f"    converted <-> SUNO:  {sim_conv_suno:.4f}  (want LOWER than baseline)")
    print(f"    SUNO <-> user:       {sim_suno_user:.4f}  (baseline - gap to bridge)")

    # Did conversion move embedding toward user?
    if sim_conv_user > sim_suno_user:
        delta = sim_conv_user - sim_suno_user
        print(f"    -> Conversion IMPROVED similarity to user by +{delta:.4f}")
    else:
        delta = sim_suno_user - sim_conv_user
        print(f"    -> Conversion did NOT improve similarity (worse by {delta:.4f})")

    print(f"\n  F0 (pitch) correlation:")
    print(f"    source <-> converted: {f0_corr:.4f}  (want HIGH - melody preserved)")
    if f0_corr > 0.9:
        print(f"    -> EXCELLENT pitch preservation")
    elif f0_corr > 0.7:
        print(f"    -> GOOD pitch preservation")
    elif f0_corr > 0.5:
        print(f"    -> MODERATE pitch preservation")
    else:
        print(f"    -> POOR pitch preservation")

    # MFCC similarity for additional context
    import librosa
    def _mfcc(y):
        return np.mean(librosa.feature.mfcc(y=y, sr=OUTPUT_SR, n_mfcc=13)[1:], axis=1)
    def _cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))

    min_len = min(len(suno_vocals), len(converted), len(user_audio))
    m_suno = _mfcc(suno_vocals[:min_len])
    m_conv = _mfcc(converted[:min_len])
    m_user = _mfcc(user_audio[:min_len])

    print(f"\n  MFCC cosine similarity:")
    print(f"    converted <-> user: {_cos(m_conv, m_user):.4f}")
    print(f"    converted <-> SUNO: {_cos(m_conv, m_suno):.4f}")
    print(f"    SUNO <-> user:      {_cos(m_suno, m_user):.4f}")

    # Spectral centroid
    def _centroid(y):
        return float(np.mean(librosa.feature.spectral_centroid(y=y, sr=OUTPUT_SR)))

    print(f"\n  Spectral centroid (Hz):")
    print(f"    SUNO:      {_centroid(suno_vocals):.0f}")
    print(f"    Converted: {_centroid(converted):.0f}")
    print(f"    User:      {_centroid(user_audio):.0f}")

    print("\n" + "=" * 60)
    print("NOTE: rvc-python requires a pre-trained voice model (.pth)")
    print("which must be trained using the full RVC WebUI on a GPU.")
    print("With only ~2min of user voice data and CPU-only macOS,")
    print("WORLD vocoder-based conversion is the practical approach.")
    print("=" * 60)


if __name__ == "__main__":
    main()
