"""
Applio RVC Training + Inference Pipeline
Runs entirely on CPU (macOS Apple Silicon).
Must be invoked from the Applio directory as cwd.
"""

import os
import sys
import time
import json
import shutil
import glob
import multiprocessing

# ---- Fix sys.path: Remove conflicting rvc-clean path, ensure Applio is first ----
APPLIO_DIR = "/Users/sarsa/claude/Applio"
MDT_DIR = "/Users/sarsa/claude/million-dollar-tuner"

# Remove any rvc-clean or similar conflicting paths
sys.path = [p for p in sys.path if "rvc-clean" not in p]
# Ensure Applio is at the very front
if APPLIO_DIR in sys.path:
    sys.path.remove(APPLIO_DIR)
sys.path.insert(0, APPLIO_DIR)
train_dir = os.path.join(APPLIO_DIR, "rvc", "train")
if train_dir in sys.path:
    sys.path.remove(train_dir)
sys.path.insert(1, train_dir)

# Also purge any cached rvc module that was loaded from the wrong path
for key in list(sys.modules.keys()):
    if key == "rvc" or key.startswith("rvc."):
        mod = sys.modules[key]
        if mod and hasattr(mod, "__file__") and mod.__file__ and "rvc-clean" in mod.__file__:
            del sys.modules[key]

import numpy as np

# ---- Configuration ----
USER_VOICE = "/Users/sarsa/Downloads/yj_voice.wav"
SUNO_VOCALS = os.path.join(MDT_DIR, "stems_dtr", "vocals.wav")
DRUMS = os.path.join(MDT_DIR, "stems_dtr", "drums.wav")
BASS = os.path.join(MDT_DIR, "stems_dtr", "bass.wav")
OTHER = os.path.join(MDT_DIR, "stems_dtr", "other.wav")
OUTPUT_VOCALS = "/Users/sarsa/Downloads/applio_vocals.wav"
OUTPUT_MIX = "/Users/sarsa/Downloads/applio_mix.wav"

MODEL_NAME = "yj_voice"
SAMPLE_RATE = 40000
F0_METHOD = "rmvpe"
EMBEDDER_MODEL = "contentvec"
BATCH_SIZE = 4
TOTAL_EPOCHS = 20  # Keep low for CPU training
SAVE_EVERY_EPOCH = 10
VOCODER = "HiFi-GAN"

# Ensure we are in the Applio directory
os.chdir(APPLIO_DIR)

# ---- Step 0: Download Prerequisites ----
def download_prerequisites():
    """Download pretrained models, predictors, and embedders if missing."""
    print("\n" + "=" * 60)
    print("STEP 0: Downloading prerequisites")
    print("=" * 60)

    from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline
    prequisites_download_pipeline(
        pretraineds_hifigan=True,
        models=True,
        exe=False,
    )
    # Verify downloads
    pg, pd = get_pretrained_paths()
    print(f"  Pretrained G: {pg} (exists={os.path.exists(pg)})")
    print(f"  Pretrained D: {pd} (exists={os.path.exists(pd)})")

    rmvpe_path = os.path.join("rvc", "models", "predictors", "rmvpe.pt")
    print(f"  RMVPE model: {rmvpe_path} (exists={os.path.exists(rmvpe_path)})")

    cv_path = os.path.join("rvc", "models", "embedders", "contentvec")
    cv_bin = os.path.join(cv_path, "pytorch_model.bin")
    print(f"  ContentVec: {cv_bin} (exists={os.path.exists(cv_bin)})")


def get_pretrained_paths():
    """Get pretrained model paths for the given sample rate and vocoder."""
    from rvc.lib.tools.pretrained_selector import pretrained_selector
    return pretrained_selector(VOCODER, SAMPLE_RATE)


# ---- Step 1: Preprocess ----
def run_preprocess():
    print("\n" + "=" * 60)
    print("STEP 1: Preprocessing audio")
    print("=" * 60)

    exp_dir = os.path.join("logs", MODEL_NAME)
    os.makedirs(exp_dir, exist_ok=True)

    # Create a dataset directory with the user voice
    dataset_dir = os.path.join(exp_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    target_wav = os.path.join(dataset_dir, "yj_voice.wav")
    if not os.path.exists(target_wav):
        shutil.copy2(USER_VOICE, target_wav)
    print(f"  Dataset: {dataset_dir}")

    from rvc.train.preprocess.preprocess import PreProcess, save_dataset_duration

    start = time.time()
    pp = PreProcess(sr=SAMPLE_RATE, exp_dir=exp_dir)

    # Process the audio file directly (avoid multiprocessing issues)
    audio_length = pp.process_audio(
        path=target_wav,
        idx0=0,
        sid=0,
        cut_preprocess="Automatic",
        process_effects=True,
        noise_reduction=False,
        reduction_strength=0.7,
        chunk_len=3.0,
        overlap_len=0.3,
        normalization_mode="pre",
    )

    save_dataset_duration(
        os.path.join(exp_dir, "model_info.json"),
        dataset_duration=audio_length,
    )

    elapsed = time.time() - start
    n_slices = len(glob.glob(os.path.join(exp_dir, "sliced_audios", "*.wav")))
    n_16k = len(glob.glob(os.path.join(exp_dir, "sliced_audios_16k", "*.wav")))
    print(f"  Preprocessing done in {elapsed:.1f}s")
    print(f"  Sliced audios: {n_slices}, 16k versions: {n_16k}")
    print(f"  Audio duration: {audio_length:.1f}s")
    return exp_dir


# ---- Step 2: Feature Extraction ----
def run_feature_extraction(exp_dir):
    print("\n" + "=" * 60)
    print("STEP 2: Feature extraction (F0 + embeddings)")
    print("=" * 60)

    wav_path = os.path.join(exp_dir, "sliced_audios_16k")
    os.makedirs(os.path.join(exp_dir, "f0"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "f0_voiced"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "extracted"), exist_ok=True)

    # Build file list
    files = []
    for f in sorted(glob.glob(os.path.join(wav_path, "*.wav"))):
        fn = os.path.basename(f)
        files.append([
            f,
            os.path.join(exp_dir, "f0", fn + ".npy"),
            os.path.join(exp_dir, "f0_voiced", fn + ".npy"),
            os.path.join(exp_dir, "extracted", fn.replace("wav", "npy")),
        ])

    print(f"  Files to process: {len(files)}")

    # --- F0 Extraction ---
    print("  Extracting F0 with rmvpe...")
    start = time.time()

    # Import RMVPE directly to avoid torchfcpe import chain issue
    from rvc.lib.predictors.RMVPE import RMVPE0Predictor
    from rvc.lib.utils import load_audio_16k as _load_16k
    import torch

    class SimpleFeatureInput:
        """Minimal F0 extractor using RMVPE, avoids importing FCPE."""
        def __init__(self, device="cpu"):
            self.hop_size = 160
            self.sample_rate = 16000
            self.f0_bin = 256
            self.f0_max = 1100.0
            self.f0_min = 50.0
            self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
            self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
            self.device = device
            self.model = RMVPE0Predictor(
                os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
                device=self.device,
            )

        def compute_f0(self, x):
            return self.model.infer_from_audio(x, thred=0.03)

        def coarse_f0(self, f0):
            f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
            f0_mel = np.clip(
                (f0_mel - self.f0_mel_min) * (self.f0_bin - 2) / (self.f0_mel_max - self.f0_mel_min) + 1,
                1, self.f0_bin - 1,
            )
            return np.rint(f0_mel).astype(int)

        def process_file(self, file_info):
            inp_path, opt_path_coarse, opt_path_full, _ = file_info
            if os.path.exists(opt_path_coarse) and os.path.exists(opt_path_full):
                return
            try:
                np_arr = _load_16k(inp_path)
                feature_pit = self.compute_f0(np_arr)
                np.save(opt_path_full, feature_pit, allow_pickle=False)
                coarse_pit = self.coarse_f0(feature_pit)
                np.save(opt_path_coarse, coarse_pit, allow_pickle=False)
            except Exception as error:
                print(f"  Error extracting F0 from {inp_path}: {error}")

    from tqdm import tqdm
    fe = SimpleFeatureInput(device="cpu")
    for fi in tqdm(files, desc="  F0"):
        fe.process_file(fi)

    f0_time = time.time() - start
    print(f"  F0 extraction done in {f0_time:.1f}s")

    # --- Embedding Extraction ---
    print(f"  Extracting embeddings with {EMBEDDER_MODEL}...")
    start = time.time()

    import torch
    from rvc.lib.utils import load_audio_16k, load_embedding
    model = load_embedding(EMBEDDER_MODEL).to("cpu").float()
    model.eval()

    for fi in tqdm(files, desc="  Embeddings"):
        wav_file_path, _, _, out_file_path = fi
        if os.path.exists(out_file_path):
            continue
        feats = torch.from_numpy(load_audio_16k(wav_file_path)).float()
        feats = feats.view(1, -1)
        with torch.no_grad():
            result = model(feats)["last_hidden_state"]
        feats_out = result.squeeze(0).float().cpu().numpy()
        if not np.isnan(feats_out).any():
            np.save(out_file_path, feats_out, allow_pickle=False)
        else:
            print(f"  WARNING: {wav_file_path} produced NaN values; skipping.")

    del model
    emb_time = time.time() - start
    print(f"  Embedding extraction done in {emb_time:.1f}s")

    # Save embedder model info
    model_info_path = os.path.join(exp_dir, "model_info.json")
    if os.path.exists(model_info_path):
        with open(model_info_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data["embedder_model"] = EMBEDDER_MODEL
    with open(model_info_path, "w") as f:
        json.dump(data, f, indent=4)

    # Generate config and filelist
    from rvc.train.extract.preparing_files import generate_config, generate_filelist
    generate_config(SAMPLE_RATE, exp_dir)
    generate_filelist(exp_dir, SAMPLE_RATE, include_mutes=2)

    n_f0 = len(glob.glob(os.path.join(exp_dir, "f0", "*.npy")))
    n_emb = len(glob.glob(os.path.join(exp_dir, "extracted", "*.npy")))
    print(f"  F0 files: {n_f0}, Embedding files: {n_emb}")

    # Check filelist
    filelist_path = os.path.join(exp_dir, "filelist.txt")
    with open(filelist_path, "r") as f:
        lines = f.readlines()
    print(f"  Filelist entries: {len(lines)}")

    return f0_time, emb_time


# ---- Step 3: Training ----
def run_training(exp_dir):
    print("\n" + "=" * 60)
    print("STEP 3: Training RVC model")
    print("=" * 60)

    pg, pd = get_pretrained_paths()
    if not pg or not pd or not os.path.exists(pg) or not os.path.exists(pd):
        print("  WARNING: Pretrained models not found. Training from scratch.")
        pg, pd = "", ""
    else:
        print(f"  Using pretrained G: {pg}")
        print(f"  Using pretrained D: {pd}")

    print(f"  Epochs: {TOTAL_EPOCHS}, Batch size: {BATCH_SIZE}, Sample rate: {SAMPLE_RATE}")
    print(f"  Vocoder: {VOCODER}")

    # Training uses subprocess because it needs DDP init_process_group
    # We'll run it as a subprocess
    import subprocess
    train_script = os.path.join("rvc", "train", "train.py")

    cmd = [
        sys.executable,
        train_script,
        MODEL_NAME,              # model_name
        str(SAVE_EVERY_EPOCH),   # save_every_epoch
        str(TOTAL_EPOCHS),       # total_epoch
        pg,                      # pretrainG
        pd,                      # pretrainD
        "-",                     # gpus (- means CPU)
        str(BATCH_SIZE),         # batch_size
        str(SAMPLE_RATE),        # sample_rate
        "True",                  # save_only_latest
        "True",                  # save_every_weights
        "False",                 # cache_data_in_gpu
        "False",                 # overtraining_detector
        "50",                    # overtraining_threshold
        "True",                  # cleanup
        VOCODER,                 # vocoder
        "False",                 # checkpointing
    ]

    print(f"  Command: {' '.join(cmd)}")
    start = time.time()

    env = os.environ.copy()
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = "29500"

    result = subprocess.run(
        cmd,
        cwd=APPLIO_DIR,
        env=env,
        timeout=7200,  # 2 hour timeout
    )

    train_time = time.time() - start
    # Note: Applio's train.py calls os._exit(2333333) on completion, which looks like failure
    # but is actually their way of signaling "done"
    print(f"  Training completed in {train_time:.1f}s (return code: {result.returncode})")
    if result.returncode != 0:
        print(f"  (Note: Applio uses os._exit(2333333) for normal completion)")

    # Find the trained model
    model_files = sorted(glob.glob(os.path.join(exp_dir, f"{MODEL_NAME}_*.pth")))
    if model_files:
        model_path = model_files[-1]
        print(f"  Trained model: {model_path}")
    else:
        print("  ERROR: No trained model found!")
        model_path = None

    return model_path, train_time


# ---- Step 4: Build Index ----
def run_index(exp_dir):
    print("\n" + "=" * 60)
    print("STEP 4: Building FAISS index")
    print("=" * 60)

    import subprocess
    index_script = os.path.join("rvc", "train", "process", "extract_index.py")
    cmd = [sys.executable, index_script, exp_dir, "Auto"]
    result = subprocess.run(cmd, cwd=APPLIO_DIR)
    print(f"  Index build return code: {result.returncode}")

    index_files = glob.glob(os.path.join(exp_dir, "*.index"))
    if index_files:
        print(f"  Index file: {index_files[0]}")
        return index_files[0]
    else:
        print("  No index file generated (will run inference without index)")
        return ""


# ---- Step 5: Inference ----
def run_inference(model_path, index_path):
    print("\n" + "=" * 60)
    print("STEP 5: Running inference")
    print("=" * 60)

    if not model_path or not os.path.exists(model_path):
        print("  ERROR: No trained model available for inference")
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


# ---- Step 6: Mix with instrumentals ----
def mix_audio():
    print("\n" + "=" * 60)
    print("STEP 6: Mixing with instrumentals")
    print("=" * 60)

    import soundfile as sf
    import librosa

    # Load converted vocals
    vocals, sr_v = sf.read(OUTPUT_VOCALS)
    print(f"  Converted vocals: sr={sr_v}, len={len(vocals)}")

    # Load stems at the same sample rate
    drums, _ = librosa.load(DRUMS, sr=sr_v, mono=True)
    bass, _ = librosa.load(BASS, sr=sr_v, mono=True)
    other, _ = librosa.load(OTHER, sr=sr_v, mono=True)

    # Ensure vocals are mono
    if len(vocals.shape) > 1:
        vocals = vocals.mean(axis=1)

    # Match lengths
    target_len = max(len(drums), len(bass), len(other), len(vocals))
    def pad_to(arr, length):
        if len(arr) < length:
            return np.pad(arr, (0, length - len(arr)))
        return arr[:length]

    vocals = pad_to(vocals, target_len)
    drums = pad_to(drums, target_len)
    bass = pad_to(bass, target_len)
    other = pad_to(other, target_len)

    # Mix: vocals + instrumentals
    mix = vocals * 1.0 + drums * 0.9 + bass * 0.9 + other * 0.85

    # Normalize
    peak = np.abs(mix).max()
    if peak > 0:
        mix = mix / peak * 0.95

    sf.write(OUTPUT_MIX, mix, sr_v)
    print(f"  Mix saved: {OUTPUT_MIX}")


# ---- Step 7: Quality Verification ----
def verify_quality():
    print("\n" + "=" * 60)
    print("STEP 7: Quality verification")
    print("=" * 60)

    import soundfile as sf
    import librosa

    # --- Resemblyzer speaker similarity ---
    print("  Computing speaker embeddings with resemblyzer...")
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder()

    # Load and preprocess audio files
    user_wav = preprocess_wav(USER_VOICE)
    suno_wav = preprocess_wav(SUNO_VOCALS)
    output_wav = preprocess_wav(OUTPUT_VOCALS)

    # Compute embeddings
    emb_user = encoder.embed_utterance(user_wav)
    emb_suno = encoder.embed_utterance(suno_wav)
    emb_output = encoder.embed_utterance(output_wav)

    # Cosine similarity
    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_output_user = cosine_sim(emb_output, emb_user)
    sim_output_suno = cosine_sim(emb_output, emb_suno)
    sim_user_suno = cosine_sim(emb_user, emb_suno)

    print(f"  Speaker similarity (output vs user voice): {sim_output_user:.4f}")
    print(f"  Speaker similarity (output vs SUNO vocals): {sim_output_suno:.4f}")
    print(f"  Speaker similarity (user voice vs SUNO):    {sim_user_suno:.4f}")

    # --- F0 Correlation ---
    print("\n  Computing F0 correlation...")
    suno_audio, sr_s = librosa.load(SUNO_VOCALS, sr=16000, mono=True)
    output_audio, sr_o = librosa.load(OUTPUT_VOCALS, sr=16000, mono=True)

    # Use pyin for F0 extraction
    f0_suno, _, _ = librosa.pyin(suno_audio, fmin=50, fmax=1100, sr=16000)
    f0_output, _, _ = librosa.pyin(output_audio, fmin=50, fmax=1100, sr=16000)

    # Align lengths
    min_len = min(len(f0_suno), len(f0_output))
    f0_suno = f0_suno[:min_len]
    f0_output = f0_output[:min_len]

    # Filter out NaN values (unvoiced frames)
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


# ---- Main ----
if __name__ == "__main__":
    print("=" * 60)
    print("Applio RVC Pipeline - Full Training + Inference")
    print("=" * 60)

    total_start = time.time()
    exp_dir = os.path.join("logs", MODEL_NAME)

    # Step 0: Check prerequisites (skip download - already done)
    pg, pd = get_pretrained_paths()
    rmvpe_path = os.path.join("rvc", "models", "predictors", "rmvpe.pt")
    cv_bin = os.path.join("rvc", "models", "embedders", "contentvec", "pytorch_model.bin")
    print(f"  Pretrained G: {pg} (exists={os.path.exists(pg)})")
    print(f"  Pretrained D: {pd} (exists={os.path.exists(pd)})")
    print(f"  RMVPE model: {rmvpe_path} (exists={os.path.exists(rmvpe_path)})")
    print(f"  ContentVec: {cv_bin} (exists={os.path.exists(cv_bin)})")

    # Step 1: Preprocess
    exp_dir = run_preprocess()

    # Step 2: Feature extraction
    f0_time, emb_time = run_feature_extraction(exp_dir)

    # Step 3: Training
    model_path, train_time = run_training(exp_dir)

    # Step 4: Build index
    index_path = run_index(exp_dir)

    # Step 5: Inference
    success = run_inference(model_path, index_path)

    if success:
        # Step 6: Mix
        mix_audio()

        # Step 7: Verify
        metrics = verify_quality()
    else:
        metrics = {}

    # ---- Summary ----
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  F0 extraction: {f0_time:.1f}s")
    print(f"  Embedding extraction: {emb_time:.1f}s")
    print(f"  Training: {train_time:.1f}s ({TOTAL_EPOCHS} epochs)")
    if metrics:
        print(f"  Speaker similarity (output vs user): {metrics['sim_output_user']:.4f}")
        print(f"  Speaker similarity (output vs SUNO): {metrics['sim_output_suno']:.4f}")
        print(f"  Speaker similarity (user vs SUNO):   {metrics['sim_user_suno']:.4f}")
        print(f"  F0 correlation (output vs SUNO):     {metrics['f0_correlation']:.4f}")
    print(f"\n  Converted vocals: {OUTPUT_VOCALS}")
    print(f"  Final mix: {OUTPUT_MIX}")
    print("=" * 60)
