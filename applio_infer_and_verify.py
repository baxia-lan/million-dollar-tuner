"""
Applio RVC Inference + Quality Verification
Patches the torchfcpe import issue before loading Applio modules.
"""

import os
import sys
import time
import json
import types

# ---- Fix sys.path ----
APPLIO_DIR = "/Users/sarsa/claude/Applio"
MDT_DIR = "/Users/sarsa/claude/million-dollar-tuner"

sys.path = [p for p in sys.path if "rvc-clean" not in p]
if APPLIO_DIR in sys.path:
    sys.path.remove(APPLIO_DIR)
sys.path.insert(0, APPLIO_DIR)
train_dir = os.path.join(APPLIO_DIR, "rvc", "train")
if train_dir in sys.path:
    sys.path.remove(train_dir)
sys.path.insert(1, train_dir)

os.chdir(APPLIO_DIR)

# ---- Patch torchfcpe to avoid the import chain issue ----
# Create a fake torchfcpe module so the import doesn't fail
fake_torchfcpe = types.ModuleType("torchfcpe")
fake_torchfcpe.spawn_infer_model_from_pt = None  # dummy
sys.modules["torchfcpe"] = fake_torchfcpe

import numpy as np

# Now we can safely import Applio modules
USER_VOICE = "/Users/sarsa/Downloads/yj_voice.wav"
SUNO_VOCALS = os.path.join(MDT_DIR, "stems_dtr", "vocals.wav")
DRUMS = os.path.join(MDT_DIR, "stems_dtr", "drums.wav")
BASS = os.path.join(MDT_DIR, "stems_dtr", "bass.wav")
OTHER = os.path.join(MDT_DIR, "stems_dtr", "other.wav")
OUTPUT_VOCALS = "/Users/sarsa/Downloads/applio_vocals.wav"
OUTPUT_MIX = "/Users/sarsa/Downloads/applio_mix.wav"

MODEL_NAME = "yj_voice"
EMBEDDER_MODEL = "contentvec"


def find_model():
    import glob
    exp_dir = os.path.join("logs", MODEL_NAME)
    model_files = sorted(glob.glob(os.path.join(exp_dir, f"{MODEL_NAME}_*.pth")))
    if model_files:
        return model_files[-1]
    return None


def run_inference(model_path, index_path):
    print("\n" + "=" * 60)
    print("STEP 1: Running inference")
    print("=" * 60)

    if not model_path or not os.path.exists(model_path):
        print(f"  ERROR: Model not found: {model_path}")
        return False

    print(f"  Model: {model_path}")
    print(f"  Index: {index_path}")

    from rvc.infer.infer import VoiceConverter
    vc = VoiceConverter()

    start = time.time()
    vc.convert_audio(
        audio_input_path=SUNO_VOCALS,
        audio_output_path=OUTPUT_VOCALS,
        model_path=model_path,
        index_path=index_path or "",
        pitch=0,
        f0_method="rmvpe",
        index_rate=0.75 if index_path and os.path.exists(index_path) else 0.0,
        volume_envelope=1.0,
        protect=0.5,
        hop_length=128,
        split_audio=False,
        f0_autotune=False,
        f0_autotune_strength=1.0,
        embedder_model=EMBEDDER_MODEL,
        clean_audio=False,
        export_format="WAV",
        sid=0,
    )
    infer_time = time.time() - start
    print(f"  Inference done in {infer_time:.1f}s")
    print(f"  Output: {OUTPUT_VOCALS}")
    return True


def mix_audio():
    print("\n" + "=" * 60)
    print("STEP 2: Mixing with instrumentals")
    print("=" * 60)

    import soundfile as sf
    import librosa

    vocals, sr_v = sf.read(OUTPUT_VOCALS)
    print(f"  Converted vocals: sr={sr_v}, len={len(vocals)}")

    drums, _ = librosa.load(DRUMS, sr=sr_v, mono=True)
    bass, _ = librosa.load(BASS, sr=sr_v, mono=True)
    other, _ = librosa.load(OTHER, sr=sr_v, mono=True)

    if len(vocals.shape) > 1:
        vocals = vocals.mean(axis=1)

    target_len = max(len(drums), len(bass), len(other), len(vocals))
    def pad_to(arr, length):
        if len(arr) < length:
            return np.pad(arr, (0, length - len(arr)))
        return arr[:length]

    vocals = pad_to(vocals, target_len)
    drums = pad_to(drums, target_len)
    bass = pad_to(bass, target_len)
    other = pad_to(other, target_len)

    mix = vocals * 1.0 + drums * 0.9 + bass * 0.9 + other * 0.85
    peak = np.abs(mix).max()
    if peak > 0:
        mix = mix / peak * 0.95

    sf.write(OUTPUT_MIX, mix, sr_v)
    print(f"  Mix saved: {OUTPUT_MIX}")


def verify_quality():
    print("\n" + "=" * 60)
    print("STEP 3: Quality verification")
    print("=" * 60)

    import soundfile as sf
    import librosa

    print("  Computing speaker embeddings with resemblyzer...")
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder()

    user_wav = preprocess_wav(USER_VOICE)
    suno_wav = preprocess_wav(SUNO_VOCALS)
    output_wav = preprocess_wav(OUTPUT_VOCALS)

    emb_user = encoder.embed_utterance(user_wav)
    emb_suno = encoder.embed_utterance(suno_wav)
    emb_output = encoder.embed_utterance(output_wav)

    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_output_user = cosine_sim(emb_output, emb_user)
    sim_output_suno = cosine_sim(emb_output, emb_suno)
    sim_user_suno = cosine_sim(emb_user, emb_suno)

    print(f"  Speaker similarity (output vs user voice): {sim_output_user:.4f}")
    print(f"  Speaker similarity (output vs SUNO vocals): {sim_output_suno:.4f}")
    print(f"  Speaker similarity (user voice vs SUNO):    {sim_user_suno:.4f}")

    print("\n  Computing F0 correlation...")
    suno_audio, _ = librosa.load(SUNO_VOCALS, sr=16000, mono=True)
    output_audio, _ = librosa.load(OUTPUT_VOCALS, sr=16000, mono=True)

    f0_suno, _, _ = librosa.pyin(suno_audio, fmin=50, fmax=1100, sr=16000)
    f0_output, _, _ = librosa.pyin(output_audio, fmin=50, fmax=1100, sr=16000)

    min_len = min(len(f0_suno), len(f0_output))
    f0_suno = f0_suno[:min_len]
    f0_output = f0_output[:min_len]

    mask = ~(np.isnan(f0_suno) | np.isnan(f0_output))
    if mask.sum() > 10:
        f0_corr = np.corrcoef(f0_suno[mask], f0_output[mask])[0, 1]
        print(f"  F0 correlation (output vs SUNO): {f0_corr:.4f}")
        print(f"  Voiced frames used: {mask.sum()} / {min_len}")
    else:
        f0_corr = 0.0
        print("  WARNING: Too few voiced frames for F0 correlation")

    return {
        "sim_output_user": sim_output_user,
        "sim_output_suno": sim_output_suno,
        "sim_user_suno": sim_user_suno,
        "f0_correlation": float(f0_corr) if not np.isnan(f0_corr) else 0.0,
    }


if __name__ == "__main__":
    total_start = time.time()

    model_path = find_model()
    print(f"Found model: {model_path}")

    index_path = os.path.join("logs", MODEL_NAME, "yj_voice.index")
    if not os.path.exists(index_path):
        index_path = ""

    success = run_inference(model_path, index_path)

    if success:
        mix_audio()
        metrics = verify_quality()
    else:
        metrics = {}

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s")
    if metrics:
        print(f"  Speaker sim (output vs user): {metrics['sim_output_user']:.4f}")
        print(f"  Speaker sim (output vs SUNO): {metrics['sim_output_suno']:.4f}")
        print(f"  Speaker sim (user vs SUNO):   {metrics['sim_user_suno']:.4f}")
        print(f"  F0 correlation:               {metrics['f0_correlation']:.4f}")
    print(f"\n  Converted vocals: {OUTPUT_VOCALS}")
    print(f"  Final mix: {OUTPUT_MIX}")
    print("=" * 60)
