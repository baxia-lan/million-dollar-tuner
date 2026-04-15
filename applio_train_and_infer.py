"""
Applio RVC Training + Inference (Phase 2)
Preprocessing and feature extraction already done.
Just run training, inference, mixing, and quality verification.
"""

import os
import sys
import time
import json
import glob
import subprocess

import numpy as np

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

# Purge wrong rvc modules
for key in list(sys.modules.keys()):
    if key == "rvc" or key.startswith("rvc."):
        mod = sys.modules[key]
        if mod and hasattr(mod, "__file__") and mod.__file__ and "rvc-clean" in mod.__file__:
            del sys.modules[key]

os.chdir(APPLIO_DIR)

# ---- Config ----
USER_VOICE = "/Users/sarsa/Downloads/yj_voice.wav"
SUNO_VOCALS = os.path.join(MDT_DIR, "stems_dtr", "vocals.wav")
DRUMS = os.path.join(MDT_DIR, "stems_dtr", "drums.wav")
BASS = os.path.join(MDT_DIR, "stems_dtr", "bass.wav")
OTHER = os.path.join(MDT_DIR, "stems_dtr", "other.wav")
OUTPUT_VOCALS = "/Users/sarsa/Downloads/applio_vocals.wav"
OUTPUT_MIX = "/Users/sarsa/Downloads/applio_mix.wav"

MODEL_NAME = "yj_voice"
SAMPLE_RATE = 40000
EMBEDDER_MODEL = "contentvec"
BATCH_SIZE = 4
TOTAL_EPOCHS = 5  # Minimal epochs for CPU/MPS training
SAVE_EVERY_EPOCH = 5  # Only save at the end
VOCODER = "HiFi-GAN"

exp_dir = os.path.join("logs", MODEL_NAME)


def run_training():
    print("\n" + "=" * 60)
    print("STEP 1: Training RVC model")
    print("=" * 60)

    pg = os.path.join("rvc", "models", "pretraineds", "hifi-gan", "f0G40k.pth")
    pd = os.path.join("rvc", "models", "pretraineds", "hifi-gan", "f0D40k.pth")
    print(f"  Pretrained G: {pg} (exists={os.path.exists(pg)})")
    print(f"  Pretrained D: {pd} (exists={os.path.exists(pd)})")
    print(f"  Epochs: {TOTAL_EPOCHS}, Batch size: {BATCH_SIZE}")

    # Check data integrity
    filelist = os.path.join(exp_dir, "filelist.txt")
    with open(filelist, "r") as f:
        lines = f.readlines()
    print(f"  Filelist entries: {len(lines)}")

    train_script = os.path.join("rvc", "train", "train.py")
    cmd = [
        sys.executable,
        train_script,
        MODEL_NAME,
        str(SAVE_EVERY_EPOCH),
        str(TOTAL_EPOCHS),
        pg,
        pd,
        "-",  # CPU
        str(BATCH_SIZE),
        str(SAMPLE_RATE),
        "True",   # save_only_latest
        "True",   # save_every_weights
        "False",  # cache_data_in_gpu
        "False",  # overtraining_detector
        "50",     # overtraining_threshold
        "False",  # cleanup (don't delete existing checkpoints)
        VOCODER,
        "False",  # checkpointing
    ]

    print(f"  Starting training subprocess...")
    start = time.time()

    env = os.environ.copy()
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = "29500"

    result = subprocess.run(cmd, cwd=APPLIO_DIR, env=env, timeout=7200)

    train_time = time.time() - start
    print(f"  Training subprocess finished in {train_time:.1f}s (rc={result.returncode})")
    # Applio uses os._exit(2333333) for normal completion

    # Find the trained model
    model_files = sorted(glob.glob(os.path.join(exp_dir, f"{MODEL_NAME}_*.pth")))
    if model_files:
        model_path = model_files[-1]
        print(f"  Trained model: {model_path}")
    else:
        print("  ERROR: No trained model found!")
        model_path = None

    return model_path, train_time


def run_inference(model_path, index_path):
    print("\n" + "=" * 60)
    print("STEP 2: Running inference")
    print("=" * 60)

    if not model_path or not os.path.exists(model_path):
        print("  ERROR: No trained model available")
        return False

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
        index_rate=0.75 if index_path else 0.0,
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
    print("STEP 3: Mixing with instrumentals")
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
    print("STEP 4: Quality verification")
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
    print("=" * 60)
    print("Applio RVC Pipeline - Training + Inference")
    print("=" * 60)

    total_start = time.time()

    # Step 1: Train
    model_path, train_time = run_training()

    # Step 2: Inference
    index_path = os.path.join(exp_dir, "yj_voice.index")
    if not os.path.exists(index_path):
        index_path = ""
    success = run_inference(model_path, index_path)

    if success:
        # Step 3: Mix
        mix_audio()

        # Step 4: Verify
        metrics = verify_quality()
    else:
        metrics = {}

    # Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Training: {train_time:.1f}s ({TOTAL_EPOCHS} epochs)")
    if metrics:
        print(f"  Speaker sim (output vs user): {metrics['sim_output_user']:.4f}")
        print(f"  Speaker sim (output vs SUNO): {metrics['sim_output_suno']:.4f}")
        print(f"  Speaker sim (user vs SUNO):   {metrics['sim_user_suno']:.4f}")
        print(f"  F0 correlation:               {metrics['f0_correlation']:.4f}")
    print(f"\n  Converted vocals: {OUTPUT_VOCALS}")
    print(f"  Final mix: {OUTPUT_MIX}")
    print("=" * 60)
