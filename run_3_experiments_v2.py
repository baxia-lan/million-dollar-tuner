"""
Run 3 sequential Applio RVC training experiments with different pretrained bases.
Each experiment: preprocess -> feature extraction -> training -> inference.
After all 3, run a unified benchmark comparing results.

V2: Adjusted epochs/save interval for CPU training (~5 min/epoch).
"""
import os
import sys
import subprocess
import time
import json
import textwrap
import glob

PYTHON = sys.executable
APPLIO_DIR = "/Users/sarsa/claude/Applio"
MDT_DIR = "/Users/sarsa/claude/million-dollar-tuner"
USER_VOICE = "/Users/sarsa/Downloads/yj_voice.wav"
INFERENCE_SOURCE = os.path.join(MDT_DIR, "stems_dtr", "vocals.wav")

EXPERIMENTS = [
    {
        "model_name": "yj_singer32k",
        "sample_rate": 32000,
        "pretrained_G": "rvc/models/pretraineds/SingerPreTrain_32k/f0G_SingerPreTrain.pth",
        "pretrained_D": "rvc/models/pretraineds/SingerPreTrain_32k/f0D_SingerPreTrain.pth",
        "output_wav": "/Users/sarsa/Downloads/ab_singer32k_vocals.wav",
        "master_port": "29503",
    },
    {
        "model_name": "yj_titan48k",
        "sample_rate": 48000,
        "pretrained_G": "rvc/models/pretraineds/TITAN_48k/G-f048k-TITAN-Medium.pth",
        "pretrained_D": "rvc/models/pretraineds/TITAN_48k/D-f048k-TITAN-Medium.pth",
        "output_wav": "/Users/sarsa/Downloads/ab_titan48k_vocals.wav",
        "master_port": "29504",
    },
    {
        "model_name": "yj_snowie48k",
        "sample_rate": 48000,
        "pretrained_G": "rvc/models/pretraineds/SnowieV3.1_48k/G_SnowieV3.1_48k.pth",
        "pretrained_D": "rvc/models/pretraineds/SnowieV3.1_48k/D_SnowieV3.1_48k.pth",
        "output_wav": "/Users/sarsa/Downloads/ab_snowie48k_vocals.wav",
        "master_port": "29505",
    },
]

# CPU training reality: ~3-3.5s/step, ~98 steps/epoch = ~5 min/epoch
# For a 2-hour budget per experiment: max ~24 epochs
# With pretrained bases, even 20 epochs should show meaningful differentiation
TOTAL_EPOCHS = 20
SAVE_EVERY_EPOCH = 5  # Save checkpoints at epochs 5, 10, 15, 20
BATCH_SIZE = 4


def write_preprocess_script(exp):
    """Write a preprocessing subprocess script for one experiment."""
    script = textwrap.dedent(f'''\
        import os, sys, types, time, shutil, glob, json

        APPLIO_DIR = "{APPLIO_DIR}"
        sys.path = [p for p in sys.path if "rvc-clean" not in p]
        if APPLIO_DIR in sys.path:
            sys.path.remove(APPLIO_DIR)
        sys.path.insert(0, APPLIO_DIR)
        os.chdir(APPLIO_DIR)

        # Patch torchfcpe
        fake = types.ModuleType("torchfcpe")
        fake.spawn_infer_model_from_pt = None
        sys.modules["torchfcpe"] = fake

        model_name = "{exp['model_name']}"
        sample_rate = {exp['sample_rate']}
        user_voice = "{USER_VOICE}"

        exp_dir = os.path.join("logs", model_name)
        os.makedirs(exp_dir, exist_ok=True)

        # Create dataset directory with copy
        dataset_dir = os.path.join(exp_dir, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        target_wav = os.path.join(dataset_dir, "yj_voice.wav")
        if not os.path.exists(target_wav):
            shutil.copy2(user_voice, target_wav)

        from rvc.train.preprocess.preprocess import PreProcess, save_dataset_duration

        print(f"Preprocessing {{model_name}} at {{sample_rate}}Hz...")
        start = time.time()
        pp = PreProcess(sr=sample_rate, exp_dir=exp_dir)

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
        print(f"Preprocessing done in {{elapsed:.1f}}s")
        print(f"Sliced audios: {{n_slices}}, 16k versions: {{n_16k}}")
        print(f"Audio duration: {{audio_length:.1f}}s")
    ''')
    path = os.path.join(MDT_DIR, f"_exp_preprocess_{exp['model_name']}.py")
    with open(path, "w") as f:
        f.write(script)
    return path


def write_extract_script(exp):
    """Write a feature extraction subprocess script for one experiment."""
    script = textwrap.dedent(f'''\
        import os, sys, types, time, glob, json
        import numpy as np

        APPLIO_DIR = "{APPLIO_DIR}"
        sys.path = [p for p in sys.path if "rvc-clean" not in p]
        if APPLIO_DIR in sys.path:
            sys.path.remove(APPLIO_DIR)
        sys.path.insert(0, APPLIO_DIR)
        os.chdir(APPLIO_DIR)

        # Patch torchfcpe
        fake = types.ModuleType("torchfcpe")
        fake.spawn_infer_model_from_pt = None
        sys.modules["torchfcpe"] = fake

        model_name = "{exp['model_name']}"
        sample_rate = {exp['sample_rate']}
        exp_dir = os.path.join("logs", model_name)

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
        print(f"Files to process: {{len(files)}}")

        # --- F0 Extraction ---
        print("Extracting F0 with rmvpe...")
        start = time.time()

        from rvc.lib.predictors.RMVPE import RMVPE0Predictor
        from rvc.lib.utils import load_audio_16k as _load_16k
        import torch
        from tqdm import tqdm

        class SimpleFeatureInput:
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
                    print(f"Error extracting F0 from {{inp_path}}: {{error}}")

        fe = SimpleFeatureInput(device="cpu")
        for fi in tqdm(files, desc="F0"):
            fe.process_file(fi)

        f0_time = time.time() - start
        print(f"F0 extraction done in {{f0_time:.1f}}s")

        # --- Embedding Extraction ---
        print("Extracting embeddings with contentvec...")
        start = time.time()

        from rvc.lib.utils import load_audio_16k, load_embedding
        model = load_embedding("contentvec").to("cpu").float()
        model.eval()

        for fi in tqdm(files, desc="Embeddings"):
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
                print(f"WARNING: {{wav_file_path}} produced NaN values; skipping.")

        del model
        emb_time = time.time() - start
        print(f"Embedding extraction done in {{emb_time:.1f}}s")

        # Save embedder model info
        model_info_path = os.path.join(exp_dir, "model_info.json")
        if os.path.exists(model_info_path):
            with open(model_info_path, "r") as f:
                data = json.load(f)
        else:
            data = {{}}
        data["embedder_model"] = "contentvec"
        with open(model_info_path, "w") as f:
            json.dump(data, f, indent=4)

        # Generate config and filelist
        from rvc.train.extract.preparing_files import generate_config, generate_filelist
        generate_config(sample_rate, exp_dir)
        generate_filelist(exp_dir, sample_rate, include_mutes=2)

        n_f0 = len(glob.glob(os.path.join(exp_dir, "f0", "*.npy")))
        n_emb = len(glob.glob(os.path.join(exp_dir, "extracted", "*.npy")))
        print(f"F0 files: {{n_f0}}, Embedding files: {{n_emb}}")

        filelist_path = os.path.join(exp_dir, "filelist.txt")
        with open(filelist_path, "r") as f:
            lines = f.readlines()
        print(f"Filelist entries: {{len(lines)}}")
    ''')
    path = os.path.join(MDT_DIR, f"_exp_extract_{exp['model_name']}.py")
    with open(path, "w") as f:
        f.write(script)
    return path


def write_infer_script(exp):
    """Write an inference subprocess script for one experiment."""
    script = textwrap.dedent(f'''\
        import os, sys, types, time, glob
        import numpy as np

        APPLIO_DIR = "{APPLIO_DIR}"
        MDT_DIR = "{MDT_DIR}"
        sys.path = [p for p in sys.path if "rvc-clean" not in p]
        if APPLIO_DIR in sys.path:
            sys.path.remove(APPLIO_DIR)
        sys.path.insert(0, APPLIO_DIR)
        os.chdir(APPLIO_DIR)

        # Patch torchfcpe
        fake = types.ModuleType("torchfcpe")
        fake.spawn_infer_model_from_pt = None
        sys.modules["torchfcpe"] = fake

        import torch
        import soundfile as sf

        model_name = "{exp['model_name']}"
        exp_dir = os.path.join("logs", model_name)
        INPUT_AUDIO = "{INFERENCE_SOURCE}"
        OUTPUT_AUDIO = "{exp['output_wav']}"

        # Find the best/latest model checkpoint
        model_files = sorted(glob.glob(os.path.join(exp_dir, f"{{model_name}}_*.pth")))
        if not model_files:
            print(f"ERROR: No model files found in {{exp_dir}}")
            sys.exit(1)

        MODEL_PATH = model_files[-1]  # latest
        print(f"Model: {{MODEL_PATH}}")
        print(f"Input: {{INPUT_AUDIO}}")
        print(f"Output: {{OUTPUT_AUDIO}}")

        # Load model
        from rvc.lib.algorithm.synthesizers import Synthesizer
        from rvc.lib.utils import load_audio_infer, load_embedding
        from rvc.configs.config import Config

        config = Config()
        print(f"Device: {{config.device}}")

        cpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
        use_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")
        text_enc_hidden_dim = 768 if version == "v2" else 256
        vocoder = cpt.get("vocoder", "HiFi-GAN")

        print(f"Target SR: {{tgt_sr}}, Version: {{version}}, Vocoder: {{vocoder}}, Use F0: {{use_f0}}")

        net_g = Synthesizer(
            *cpt["config"],
            use_f0=use_f0,
            text_enc_hidden_dim=text_enc_hidden_dim,
            vocoder=vocoder,
        )
        del net_g.enc_q
        net_g.load_state_dict(cpt["weight"], strict=False)
        net_g = net_g.to(config.device).float()
        net_g.eval()

        # Load ContentVec
        print("Loading ContentVec embedder...")
        hubert_model = load_embedding("contentvec").to(config.device).float()
        hubert_model.eval()

        # Load audio
        print("Loading audio...")
        audio = load_audio_infer(INPUT_AUDIO, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max

        # Setup pipeline
        from rvc.infer.pipeline import Pipeline
        vc = Pipeline(tgt_sr, config)

        # Run conversion
        print("Running voice conversion...")
        start = time.time()

        sid = 0
        audio_opt = vc.pipeline(
            model=hubert_model,
            net_g=net_g,
            sid=sid,
            audio=audio,
            pitch=0,
            f0_method="rmvpe",
            file_index="",
            index_rate=0.0,
            pitch_guidance=bool(use_f0),
            volume_envelope=1.0,
            version=version,
            protect=0.5,
            f0_autotune=False,
            f0_autotune_strength=1.0,
            proposed_pitch=False,
            proposed_pitch_threshold=155.0,
        )

        elapsed = time.time() - start
        print(f"Conversion done in {{elapsed:.1f}}s")

        sf.write(OUTPUT_AUDIO, audio_opt, tgt_sr, format="WAV")
        print(f"Saved: {{OUTPUT_AUDIO}}")
        print(f"Output shape: {{audio_opt.shape}}, SR: {{tgt_sr}}")
    ''')
    path = os.path.join(MDT_DIR, f"_exp_infer_{exp['model_name']}.py")
    with open(path, "w") as f:
        f.write(script)
    return path


def run_subprocess(script_path, description, timeout=7200, env_extra=None):
    """Run a Python script as subprocess and return (success, elapsed_time)."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Script: {script_path}")
    print(f"{'='*60}")

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    start = time.time()
    try:
        result = subprocess.run(
            [PYTHON, script_path],
            cwd=APPLIO_DIR,
            env=env,
            timeout=timeout,
        )
        elapsed = time.time() - start

        # Applio training exits with 2333333 on success
        success = result.returncode == 0 or result.returncode == 2333333
        # On macOS, os._exit(2333333) gets truncated: 2333333 % 256 = 181
        if result.returncode == 181 or result.returncode == -15:
            success = True

        print(f"  Return code: {result.returncode}, Success: {success}, Time: {elapsed:.1f}s")
        return success, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  TIMEOUT after {elapsed:.1f}s")
        return False, elapsed


def run_experiment(exp_idx, exp):
    """Run a full experiment: preprocess + extract + train + infer."""
    model_name = exp["model_name"]
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT {exp_idx+1}/3: {model_name}")
    print(f"# Sample rate: {exp['sample_rate']}, Pretrained: {os.path.basename(exp['pretrained_G'])}")
    print(f"{'#'*60}")

    exp_start = time.time()
    timings = {}

    exp_dir = os.path.join(APPLIO_DIR, "logs", model_name)

    # Check if preprocessing already done (from prior run)
    sliced_dir = os.path.join(exp_dir, "sliced_audios")
    if os.path.isdir(sliced_dir) and len(glob.glob(os.path.join(sliced_dir, "*.wav"))) > 100:
        print(f"  Preprocessing already done ({len(glob.glob(os.path.join(sliced_dir, '*.wav')))} slices), skipping...")
        timings["preprocess"] = 0
    else:
        # 1. Preprocessing
        preprocess_script = write_preprocess_script(exp)
        ok, t = run_subprocess(preprocess_script, f"Preprocessing {model_name}")
        timings["preprocess"] = t
        if not ok:
            print(f"ERROR: Preprocessing failed for {model_name}")
            return timings, False

    # Check if feature extraction already done
    extracted_dir = os.path.join(exp_dir, "extracted")
    filelist_path = os.path.join(exp_dir, "filelist.txt")
    if os.path.isdir(extracted_dir) and len(glob.glob(os.path.join(extracted_dir, "*.npy"))) > 100 and os.path.exists(filelist_path):
        print(f"  Feature extraction already done, skipping...")
        timings["extract"] = 0
    else:
        # 2. Feature extraction
        extract_script = write_extract_script(exp)
        ok, t = run_subprocess(extract_script, f"Feature extraction {model_name}")
        timings["extract"] = t
        if not ok:
            print(f"ERROR: Feature extraction failed for {model_name}")
            return timings, False

    # Check if a model checkpoint already exists
    existing_models = sorted(glob.glob(os.path.join(exp_dir, f"{model_name}_*.pth")))
    if existing_models:
        print(f"  Found existing checkpoint: {existing_models[-1]}")
        print(f"  Skipping training, using existing model.")
        timings["train"] = 0
    else:
        # 3. Training
        train_cmd = [
            PYTHON,
            os.path.join("rvc", "train", "train.py"),
            model_name,
            str(SAVE_EVERY_EPOCH),
            str(TOTAL_EPOCHS),
            exp["pretrained_G"],
            exp["pretrained_D"],
            "-",  # gpus (- means CPU)
            str(BATCH_SIZE),
            str(exp["sample_rate"]),
            "True",   # save_only_latest
            "True",   # save_every_weights
            "False",  # cache_data_in_gpu
            "False",  # overtraining_detector
            "50",     # overtraining_threshold
            "False",  # cleanup
            "HiFi-GAN",  # vocoder
            "False",  # checkpointing
        ]

        train_env = {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": exp["master_port"],
        }

        print(f"\n{'='*60}")
        print(f"  Training {model_name} ({TOTAL_EPOCHS} epochs, save every {SAVE_EVERY_EPOCH})")
        print(f"  Command: {' '.join(train_cmd)}")
        print(f"{'='*60}")

        env = os.environ.copy()
        env.update(train_env)

        train_start = time.time()
        try:
            result = subprocess.run(
                train_cmd,
                cwd=APPLIO_DIR,
                env=env,
                timeout=7200,
            )
            train_elapsed = time.time() - train_start
            print(f"  Training return code: {result.returncode}, Time: {train_elapsed:.1f}s")
        except subprocess.TimeoutExpired:
            train_elapsed = time.time() - train_start
            print(f"  Training TIMEOUT after {train_elapsed:.1f}s")
        timings["train"] = train_elapsed

    # 4. Inference
    infer_script = write_infer_script(exp)
    ok, t = run_subprocess(infer_script, f"Inference {model_name}")
    timings["infer"] = t

    if not ok:
        print(f"WARNING: Inference failed for {model_name}")

    total = time.time() - exp_start
    timings["total"] = total
    print(f"\n  Experiment {model_name} total time: {total:.1f}s ({total/60:.1f}min)")
    return timings, os.path.exists(exp["output_wav"])


def run_benchmark():
    """Run unified benchmark comparing all 3 experiments."""
    print(f"\n{'#'*60}")
    print(f"# UNIFIED BENCHMARK")
    print(f"{'#'*60}")

    script = textwrap.dedent(f'''\
        import os, sys, types
        import numpy as np

        APPLIO_DIR = "{APPLIO_DIR}"
        sys.path = [p for p in sys.path if "rvc-clean" not in p]
        if APPLIO_DIR in sys.path:
            sys.path.remove(APPLIO_DIR)
        sys.path.insert(0, APPLIO_DIR)

        # Patch torchfcpe
        fake = types.ModuleType("torchfcpe")
        fake.spawn_infer_model_from_pt = None
        sys.modules["torchfcpe"] = fake

        import librosa
        import json

        USER_VOICE = "{USER_VOICE}"
        SOURCE_VOCALS = "{INFERENCE_SOURCE}"

        outputs = {{
            "SingerPreTrain 32k": "/Users/sarsa/Downloads/ab_singer32k_vocals.wav",
            "TITAN 48k": "/Users/sarsa/Downloads/ab_titan48k_vocals.wav",
            "SnowieV3.1 48k": "/Users/sarsa/Downloads/ab_snowie48k_vocals.wav",
        }}

        # Filter to only existing files
        available = {{k: v for k, v in outputs.items() if os.path.exists(v)}}
        if not available:
            print("ERROR: No output files found for benchmarking!")
            sys.exit(1)
        print(f"Benchmarking {{len(available)}} outputs...")

        # --- Resemblyzer ---
        print("Computing speaker embeddings with resemblyzer...")
        from resemblyzer import VoiceEncoder, preprocess_wav

        encoder = VoiceEncoder()

        user_wav = preprocess_wav(USER_VOICE)
        source_wav = preprocess_wav(SOURCE_VOCALS)
        emb_user = encoder.embed_utterance(user_wav)
        emb_source = encoder.embed_utterance(source_wav)

        def cosine_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        baseline_sim = cosine_sim(emb_user, emb_source)
        print(f"Baseline (user vs source vocals): {{baseline_sim:.4f}}")
        print()

        results = {{}}
        for name, path in available.items():
            print(f"--- {{name}} ---")
            out_wav = preprocess_wav(path)
            emb_out = encoder.embed_utterance(out_wav)

            sim_user = cosine_sim(emb_out, emb_user)
            sim_source = cosine_sim(emb_out, emb_source)
            print(f"  Similarity to user voice: {{sim_user:.4f}}")
            print(f"  Similarity to source:     {{sim_source:.4f}}")

            # F0 correlation
            source_audio, _ = librosa.load(SOURCE_VOCALS, sr=16000, mono=True)
            output_audio, _ = librosa.load(path, sr=16000, mono=True)

            f0_source, _, _ = librosa.pyin(source_audio, fmin=50, fmax=1100, sr=16000)
            f0_output, _, _ = librosa.pyin(output_audio, fmin=50, fmax=1100, sr=16000)

            min_len = min(len(f0_source), len(f0_output))
            f0_source = f0_source[:min_len]
            f0_output = f0_output[:min_len]

            mask = ~(np.isnan(f0_source) | np.isnan(f0_output))
            if mask.sum() > 10:
                f0_corr = np.corrcoef(f0_source[mask], f0_output[mask])[0, 1]
            else:
                f0_corr = 0.0
            print(f"  F0 correlation with source: {{f0_corr:.4f}}")
            print()

            results[name] = {{
                "sim_user": sim_user,
                "sim_source": sim_source,
                "f0_corr": float(f0_corr) if not np.isnan(f0_corr) else 0.0,
            }}

        # Summary
        print("=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"Baseline (user vs source): {{baseline_sim:.4f}}")
        print()
        print(f"{{'':<22}} {{'User Sim':>10}} {{'Source Sim':>10}} {{'F0 Corr':>10}} {{'Identity +':>10}}")
        print("-" * 62)

        best_name = None
        best_score = -1
        for name, r in results.items():
            identity_delta = r["sim_user"] - baseline_sim
            print(f"{{name:<22}} {{r['sim_user']:>10.4f}} {{r['sim_source']:>10.4f}} {{r['f0_corr']:>10.4f}} {{identity_delta:>+10.4f}}")
            # Score: prioritize identity similarity, penalize melody loss
            score = r["sim_user"] * 2 + r["f0_corr"] * 0.5
            if score > best_score:
                best_score = score
                best_name = name

        print()
        print(f"BEST PRETRAIN: {{best_name}}")
        print("(Scored by: 2*user_similarity + 0.5*f0_correlation)")

        # Save results
        with open("{MDT_DIR}/benchmark_results.json", "w") as f:
            json.dump({{
                "baseline_sim": baseline_sim,
                "results": results,
                "best": best_name,
            }}, f, indent=2)
        print(f"\\nResults saved to {MDT_DIR}/benchmark_results.json")
    ''')

    bench_path = os.path.join(MDT_DIR, "_benchmark.py")
    with open(bench_path, "w") as f:
        f.write(script)

    ok, t = run_subprocess(bench_path, "Unified Benchmark", timeout=600)
    return ok, t


if __name__ == "__main__":
    print("=" * 60)
    print("3-EXPERIMENT RVC PRETRAIN COMPARISON (v2)")
    print(f"Training data: {USER_VOICE}")
    print(f"Epochs: {TOTAL_EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"Save every: {SAVE_EVERY_EPOCH} epochs")
    print("=" * 60)

    total_start = time.time()
    all_timings = {}
    any_success = False

    for i, exp in enumerate(EXPERIMENTS):
        timings, success = run_experiment(i, exp)
        all_timings[exp["model_name"]] = timings
        if success:
            any_success = True
        if not success:
            print(f"WARNING: Experiment {exp['model_name']} produced no output.")

    # Run benchmark if any experiments succeeded
    if any_success:
        bench_ok, bench_time = run_benchmark()
        all_timings["benchmark"] = bench_time
    else:
        print("ERROR: No experiments produced outputs. Cannot benchmark.")

    # Final summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print("FINAL TIMING SUMMARY")
    print(f"{'='*60}")
    for name, t in all_timings.items():
        if isinstance(t, dict):
            print(f"  {name}:")
            for phase, secs in t.items():
                print(f"    {phase}: {secs:.1f}s ({secs/60:.1f}min)")
        else:
            print(f"  {name}: {t:.1f}s ({t/60:.1f}min)")
    print(f"\n  TOTAL: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min, {total_elapsed/3600:.1f}hr)")
    print("=" * 60)
