"""Run a single Applio A/B training experiment."""
import sys, os, types, time, glob
import subprocess
import json
import argparse

APPLIO = "/Users/sarsa/claude/Applio"

def preprocess_and_extract(name, sr, audio_path):
    """Preprocess audio + extract F0 + extract ContentVec embeddings."""
    sys.path.insert(0, APPLIO)
    os.chdir(APPLIO)
    sys.modules.setdefault("torchfcpe", types.ModuleType("torchfcpe"))
    for k in list(sys.modules.keys()):
        if k == "rvc" or k.startswith("rvc."):
            del sys.modules[k]

    import numpy as np
    import torch
    import soundfile as sf
    import librosa

    exp_dir = os.path.join("logs", name)
    os.makedirs(exp_dir, exist_ok=True)
    audio_dir = os.path.join(exp_dir, "sliced_audios")
    f0_dir = os.path.join(exp_dir, "f0")
    f0v_dir = os.path.join(exp_dir, "f0_voiced")
    emb_dir = os.path.join(exp_dir, "extracted")

    # Step 1: Preprocess
    if not os.path.exists(audio_dir) or len(os.listdir(audio_dir)) == 0:
        print(f"  Preprocessing {name} at sr={sr}...")
        from rvc.train.preprocess.preprocess import PreProcess
        pp = PreProcess(sr=sr, exp_dir=exp_dir, per=3.0)
        pp.process_audio(audio_path, exp_dir, 0, 0.0, 0.0)
        print(f"  Sliced: {len(os.listdir(audio_dir))} files")
    else:
        print(f"  Audio already sliced: {len(os.listdir(audio_dir))} files")

    # Step 2: F0
    if not os.path.exists(f0_dir) or len(os.listdir(f0_dir)) == 0:
        print("  Extracting F0 (RMVPE)...")
        os.makedirs(f0_dir, exist_ok=True)
        os.makedirs(f0v_dir, exist_ok=True)
        from rvc.lib.predictors.RMVPE import RMVPE0Predictor
        rmvpe = RMVPE0Predictor(
            os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
            is_half=False, device="cpu"
        )
        for fname in sorted(os.listdir(audio_dir)):
            if not fname.endswith(".wav"):
                continue
            y, s = sf.read(os.path.join(audio_dir, fname))
            if s != 16000:
                y = librosa.resample(y, orig_sr=s, target_sr=16000)
            f0 = rmvpe.infer_from_audio(torch.from_numpy(y).float(), 160)
            np.save(os.path.join(f0_dir, fname + ".npy"), f0)
            coarse = np.zeros_like(f0)
            coarse[f0 > 1] = 1
            np.save(os.path.join(f0v_dir, fname + ".npy"), coarse)
        print(f"  F0: {len(os.listdir(f0_dir))} files")
    else:
        print(f"  F0 already done: {len(os.listdir(f0_dir))} files")

    # Step 3: ContentVec embeddings
    if not os.path.exists(emb_dir) or len(os.listdir(emb_dir)) == 0:
        print("  Extracting ContentVec embeddings...")
        os.makedirs(emb_dir, exist_ok=True)
        from rvc.lib.utils import load_embedding
        embedder, _ = load_embedding("contentvec", None)
        embedder = embedder.to("cpu").float()

        audio_16k = os.path.join(exp_dir, "sliced_audios_16k")
        src_dir = audio_16k if os.path.exists(audio_16k) else audio_dir

        for fname in sorted(os.listdir(src_dir)):
            if not fname.endswith(".wav"):
                continue
            y, s = sf.read(os.path.join(src_dir, fname))
            if s != 16000:
                y = librosa.resample(y, orig_sr=s, target_sr=16000)
            feats = torch.from_numpy(y).float().unsqueeze(0)
            with torch.no_grad():
                out = embedder(feats, output_hidden_states=True)
                feat = out.hidden_states[12].squeeze(0).cpu().numpy()
            base = fname.replace(".wav", "")
            np.save(os.path.join(emb_dir, base + ".npy"), feat)
        print(f"  Embeddings: {len(os.listdir(emb_dir))} files")
    else:
        print(f"  Embeddings already done: {len(os.listdir(emb_dir))} files")

    # Step 4: Generate filelist
    lines = []
    for fname in sorted(os.listdir(audio_dir)):
        if not fname.endswith(".wav"):
            continue
        base = fname.replace(".wav", "")
        emb_p = os.path.join(exp_dir, "extracted", base + ".npy")
        f0_p = os.path.join(f0_dir, fname + ".npy")
        f0v_p = os.path.join(f0v_dir, fname + ".npy")
        if os.path.exists(emb_p) and os.path.exists(f0_p):
            wav_p = os.path.join(exp_dir, "sliced_audios", fname)
            lines.append(f"{wav_p}|{emb_p}|{f0_p}|{f0v_p}|0")

    filelist = os.path.join(exp_dir, "filelist.txt")
    with open(filelist, "w") as f:
        f.write("\n".join(lines))
    print(f"  Filelist: {len(lines)} entries")

    # Step 5: Generate config.json
    from rvc.configs.config import Config
    cfg = Config()
    config_data = {
        "data": {"max_wav_value": 32768.0, "sample_rate": sr, "filter_length": 2048,
                 "hop_length": sr // 100, "win_length": sr // 25,
                 "n_mel_channels": 128, "mel_fmin": 0.0, "mel_fmax": None},
        "model": {"inter_channels": 192, "hidden_channels": 192, "filter_channels": 768,
                  "n_heads": 2, "n_layers": 6, "kernel_size": 3, "p_dropout": 0,
                  "resblock": "1", "resblock_kernel_sizes": [3, 7, 11],
                  "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                  "upsample_rates": [12, 10, 2, 2] if sr >= 40000 else [10, 8, 2, 2],
                  "upsample_initial_channel": 512,
                  "upsample_kernel_sizes": [24, 20, 4, 4] if sr >= 40000 else [20, 16, 4, 4],
                  "use_spectral_norm": False, "gin_channels": 256, "emb_channels": 768},
        "train": {"log_interval": 200, "seed": 1234, "learning_rate": 0.0001,
                  "betas": [0.8, 0.99], "eps": 1e-9, "lr_decay": 0.999875,
                  "segment_size": sr // 3, "c_mel": 45, "c_kl": 1.0},
        "version": "v2"
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_data, f, indent=2)

    # model_info.json
    model_info = {"model_name": name, "sample_rate": sr, "version": "v2",
                  "embedder_model": "contentvec", "f0_method": "rmvpe"}
    with open(os.path.join(exp_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)

    print("  Preprocessing complete!")


def train(name, sr, pg, pd, epochs=20, save_every=5, batch=4):
    """Run Applio training."""
    cmd = [sys.executable, "rvc/train/train.py",
           name, str(save_every), str(epochs),
           pg, pd, "-", str(batch), str(sr),
           "True", "True", "False", "False", "50", "False", "HiFi-GAN", "False"]
    env = os.environ.copy()
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = "29506"

    print(f"  Training {epochs} epochs (save every {save_every})...")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=APPLIO, env=env, timeout=7200)
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.0f}s (rc={r.returncode})")

    exp_dir = os.path.join(APPLIO, "logs", name)
    models = sorted(glob.glob(os.path.join(exp_dir, f"{name}_*.pth")))
    if models:
        print(f"  Latest model: {os.path.basename(models[-1])}")
        return models[-1]
    return None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--sr", type=int, required=True)
    p.add_argument("--pg", required=True)
    p.add_argument("--pd", required=True)
    p.add_argument("--audio", default="/Users/sarsa/Downloads/yj_voice.wav")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()

    preprocess_and_extract(args.name, args.sr, args.audio)
    model = train(args.name, args.sr, args.pg, args.pd,
                  epochs=args.epochs, save_every=args.save_every, batch=args.batch)
    if model:
        print(f"\nSUCCESS: {model}")
    else:
        print("\nFAILED: no model exported")
