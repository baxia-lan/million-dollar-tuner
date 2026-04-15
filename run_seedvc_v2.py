#!/usr/bin/env python3
"""Seed-VC V2 voice conversion: SUNO vocals -> user voice."""

import sys
import os
import numpy as np

# ── Fix hydra module resolution: seed_vc configs reference 'modules.*' ──
# Hydra configs use bare 'modules.v2.vc_wrapper' etc., so we add seed_vc dir
# to sys.path AND alias already-imported seed_vc.modules.* as modules.*
import seed_vc as _seed_vc_pkg
_seed_vc_dir = os.path.dirname(_seed_vc_pkg.__file__)
if _seed_vc_dir not in sys.path:
    sys.path.insert(0, _seed_vc_dir)

# ── MPS float64 workaround (MUST be before any torch import) ──
import torch
torch.backends.mps.is_available = lambda: False

# ── BigVGAN patch ──
# Import via seed_vc namespace first, then make it the SAME object under
# the bare 'modules' namespace so hydra picks up the patched version.
from seed_vc.modules.bigvgan import bigvgan

_orig_from_pretrained = bigvgan.BigVGAN._from_pretrained.__func__

@classmethod
def _patched_from_pretrained(
    cls, *, model_id, revision=None, cache_dir=None, force_download=False,
    proxies=None, resume_download=False, local_files_only=False, token=None,
    map_location='cpu', strict=False, use_cuda_kernel=False, **kw
):
    return _orig_from_pretrained(
        cls,
        model_id=model_id, revision=revision, cache_dir=cache_dir,
        force_download=force_download, proxies=proxies,
        resume_download=resume_download, local_files_only=local_files_only,
        token=token, map_location=map_location, strict=strict,
        use_cuda_kernel=use_cuda_kernel, **kw
    )

bigvgan.BigVGAN._from_pretrained = _patched_from_pretrained

# ── Alias seed_vc.modules.* into bare modules.* so hydra finds patched code ──
import importlib
_modules_pkg = importlib.import_module("seed_vc.modules")
sys.modules.setdefault("modules", _modules_pkg)
# Pre-import and alias key sub-packages hydra will need
for _sub in ["bigvgan", "bigvgan.bigvgan", "campplus", "astral_quantization",
             "audio", "v2", "v2.vc_wrapper", "v2.hf_utils", "v2.cfm",
             "v2.dit_wrapper", "v2.length_regulator", "v2.ar",
             "v2.dit_model", "v2.model",
             "astral_quantization.default_model",
             "astral_quantization.convnext",
             "astral_quantization.bsq",
             "campplus.DTDNN"]:
    _full = f"seed_vc.modules.{_sub}"
    try:
        _mod = importlib.import_module(_full)
        sys.modules.setdefault(f"modules.{_sub}", _mod)
    except Exception as _e:
        print(f"  [warn] could not pre-import {_full}: {_e}")

# ── Imports ──
import soundfile as sf
from seed_vc.Models.audio import AudioData
from seed_vc.api import inference_v2, get_audio_numpy, load_v2_models

ROOT = "/Users/sarsa/claude/million-dollar-tuner"
STEMS = os.path.join(ROOT, "stems_dtr")
OUT_DIR = "/Users/sarsa/Downloads"


def load_audio_data(path: str) -> AudioData:
    """Load a WAV file into an AudioData object (mono, int16)."""
    data, sr = sf.read(path, dtype="float32")
    # Convert to mono if stereo
    if data.ndim == 2:
        data = data.mean(axis=1)
    # Convert to int16
    samples_int16 = (data * 32767).astype(np.int16)
    duration = len(samples_int16) / sr
    return AudioData(
        samples=samples_int16.tolist(),
        mel_chunks=None,
        duration=duration,
        samples_count=len(samples_int16),
        sample_rate=sr,
        metadata=None,
    )


def main():
    # ── Load audio ──
    print("Loading source vocals...")
    source = load_audio_data(os.path.join(STEMS, "vocals.wav"))
    print(f"  Source: {source.duration:.1f}s @ {source.sample_rate}Hz, {source.samples_count} samples")

    print("Loading target reference...")
    target = load_audio_data("/Users/sarsa/Downloads/recording.wav")
    print(f"  Target: {target.duration:.1f}s @ {target.sample_rate}Hz, {target.samples_count} samples")

    # ── Run Seed-VC V2 ──
    print("\nRunning Seed-VC V2 inference...")
    print("  similarity_cfg_rate=0.9, intelligibility_cfg_rate=0.5, diffusion_steps=50")

    # Load V2 models directly (bypass the API's hardcoded float16)
    from types import SimpleNamespace
    from seed_vc import inference_v2 as _infv2

    args = SimpleNamespace(
        diffusion_steps=50, length_adjust=1.0,
        intelligibility_cfg_rate=0.5, similarity_cfg_rate=0.9,
        top_p=0.9, temperature=1.0, repetition_penalty=1.0,
        convert_style=False, anonymization_only=False,
        compile=False, ar_checkpoint_path=None, cfm_checkpoint_path=None,
    )
    if _infv2.vc_wrapper_v2 is None:
        _infv2.vc_wrapper_v2 = load_v2_models(args)

    device = torch.device("cpu")
    # Use float32 on CPU (float16 autocast is not supported on CPU)
    sr_out, audio_np = _infv2.vc_wrapper_v2.convert_voice_with_streaming_arrays(
        source_wave=get_audio_numpy(source),
        target_wave=get_audio_numpy(target),
        source_sr=int(source.sample_rate),
        target_sr=int(target.sample_rate),
        diffusion_steps=50,
        length_adjust=1.0,
        intelligebility_cfg_rate=0.5,
        similarity_cfg_rate=0.9,
        top_p=0.9,
        temperature=1.0,
        repetition_penalty=1.0,
        convert_style=False,
        anonymization_only=False,
        device=device,
        dtype=torch.float32,  # float32 for CPU
        stream_output=False,
    )
    print(f"  Output: {len(audio_np)/sr_out:.1f}s @ {sr_out}Hz")

    # ── Save converted vocals ──
    out_vocals = os.path.join(OUT_DIR, "seedvc_v2.wav")
    sf.write(out_vocals, audio_np, sr_out)
    print(f"\nSaved converted vocals: {out_vocals}")

    # ── Mix with instrumentals ──
    print("\nMixing with instrumentals...")
    # Load all instrumental stems
    drums, sr_d = sf.read(os.path.join(STEMS, "drums.wav"), dtype="float32")
    bass, sr_b = sf.read(os.path.join(STEMS, "bass.wav"), dtype="float32")
    other, sr_o = sf.read(os.path.join(STEMS, "other.wav"), dtype="float32")

    # Convert instrumentals to mono
    if drums.ndim == 2:
        drums = drums.mean(axis=1)
    if bass.ndim == 2:
        bass = bass.mean(axis=1)
    if other.ndim == 2:
        other = other.mean(axis=1)

    # All stems should be same SR (44100). Resample output if needed.
    sr_inst = sr_d  # assume all instrumentals are same SR
    print(f"  Instrumental SR: {sr_inst}, Output SR: {sr_out}")

    if sr_out != sr_inst:
        import librosa
        audio_resampled = librosa.resample(audio_np.astype(np.float32), orig_sr=sr_out, target_sr=sr_inst)
    else:
        audio_resampled = audio_np.astype(np.float32)

    # Match lengths
    inst_mix = drums + bass + other
    min_len = min(len(inst_mix), len(audio_resampled))
    mix = inst_mix[:min_len] + audio_resampled[:min_len]

    # Normalize to prevent clipping
    peak = np.abs(mix).max()
    if peak > 0.95:
        mix = mix * (0.95 / peak)

    out_mix = os.path.join(OUT_DIR, "seedvc_v2_mix.wav")
    sf.write(out_mix, mix, sr_inst)
    print(f"Saved full mix: {out_mix}")

    # ── Resemblyzer speaker similarity ──
    print("\n=== Speaker Similarity (resemblyzer) ===")
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder()

    # Load and embed the three audio sources
    wav_output = preprocess_wav(out_vocals)
    wav_target = preprocess_wav("/Users/sarsa/Downloads/recording.wav")
    wav_source = preprocess_wav(os.path.join(STEMS, "vocals.wav"))

    emb_output = encoder.embed_utterance(wav_output)
    emb_target = encoder.embed_utterance(wav_target)
    emb_source = encoder.embed_utterance(wav_source)

    sim_out_target = np.dot(emb_output, emb_target) / (np.linalg.norm(emb_output) * np.linalg.norm(emb_target))
    sim_out_source = np.dot(emb_output, emb_source) / (np.linalg.norm(emb_output) * np.linalg.norm(emb_source))
    sim_target_source = np.dot(emb_target, emb_source) / (np.linalg.norm(emb_target) * np.linalg.norm(emb_source))

    print(f"  Output vs Target (user):  {sim_out_target:.4f}  (higher = more similar to user)")
    print(f"  Output vs Source (SUNO):  {sim_out_source:.4f}  (lower = better conversion)")
    print(f"  Target vs Source (baseline): {sim_target_source:.4f}")

    # ── F0 Correlation (melody preservation) ──
    print("\n=== F0 Correlation (melody preservation) ===")
    import librosa

    # Load source vocals for F0
    src_audio, src_sr = sf.read(os.path.join(STEMS, "vocals.wav"), dtype="float32")
    if src_audio.ndim == 2:
        src_audio = src_audio.mean(axis=1)

    # Use pyin for F0 extraction
    print("  Extracting F0 from source vocals...")
    f0_src, voiced_src, _ = librosa.pyin(
        src_audio, fmin=65, fmax=800, sr=src_sr,
        frame_length=2048, hop_length=512,
    )

    # Load output for F0
    out_audio_f0, out_sr_f0 = sf.read(out_vocals, dtype="float32")
    if out_audio_f0.ndim == 2:
        out_audio_f0 = out_audio_f0.mean(axis=1)

    # Resample output to source SR if needed for consistent frame alignment
    if out_sr_f0 != src_sr:
        out_audio_f0 = librosa.resample(out_audio_f0, orig_sr=out_sr_f0, target_sr=src_sr)

    print("  Extracting F0 from converted output...")
    f0_out, voiced_out, _ = librosa.pyin(
        out_audio_f0, fmin=65, fmax=800, sr=src_sr,
        frame_length=2048, hop_length=512,
    )

    # Align lengths
    min_f0_len = min(len(f0_src), len(f0_out))
    f0_src = f0_src[:min_f0_len]
    f0_out = f0_out[:min_f0_len]
    voiced_src = voiced_src[:min_f0_len]
    voiced_out = voiced_out[:min_f0_len]

    # Only compare where both are voiced
    both_voiced = voiced_src & voiced_out
    n_voiced = both_voiced.sum()
    print(f"  Frames with both voiced: {n_voiced} / {min_f0_len}")

    if n_voiced > 10:
        f0_s = f0_src[both_voiced]
        f0_o = f0_out[both_voiced]

        # Pearson correlation
        corr = np.corrcoef(f0_s, f0_o)[0, 1]
        print(f"  F0 Pearson correlation: {corr:.4f}")

        # Also log-scale correlation (more perceptually relevant)
        log_corr = np.corrcoef(np.log2(f0_s), np.log2(f0_o))[0, 1]
        print(f"  F0 log2 correlation:   {log_corr:.4f}")

        # Mean absolute error in semitones
        semitone_diff = 12 * np.abs(np.log2(f0_o / f0_s))
        mae_st = np.mean(semitone_diff)
        median_st = np.median(semitone_diff)
        print(f"  Mean F0 error:   {mae_st:.2f} semitones")
        print(f"  Median F0 error: {median_st:.2f} semitones")
    else:
        print("  Not enough voiced frames for correlation!")

    print("\nDone!")


if __name__ == "__main__":
    main()
