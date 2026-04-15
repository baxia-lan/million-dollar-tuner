"""Resume training for TITAN 48k and SnowieV3.1 48k to 200ep.

Singer32k is already complete. TITAN is at 25ep, Snowie at 20ep.
Training resumes automatically from G_2333333/D_2333333 checkpoints.
Runs sequentially: TITAN first, then Snowie.
"""
import os
import subprocess
import sys
import time
import glob

VENV_PY = sys.executable
APPLIO_DIR = "/Users/sarsa/claude/Applio"

EXPERIMENTS = [
    {
        "name": "yj_titan48k",
        "sr": 48000,
        "pg": "rvc/models/pretraineds/TITAN_48k/G-f048k-TITAN-Medium.pth",
        "pd": "rvc/models/pretraineds/TITAN_48k/D-f048k-TITAN-Medium.pth",
        "epochs": 200,
        "save_every": 25,
        "batch": 4,
    },
    {
        "name": "yj_snowie48k",
        "sr": 48000,
        "pg": "rvc/models/pretraineds/SnowieV3.1_48k/G_SnowieV3.1_48k.pth",
        "pd": "rvc/models/pretraineds/SnowieV3.1_48k/D_SnowieV3.1_48k.pth",
        "epochs": 200,
        "save_every": 25,
        "batch": 4,
    },
]


def get_max_epoch(exp_dir, name):
    """Get the highest epoch from existing weight files (numeric sort, not alpha)."""
    files = glob.glob(os.path.join(exp_dir, f"{name}_*e_*s.pth"))
    max_ep = 0
    for f in files:
        base = os.path.basename(f)
        parts = base.split("_")
        try:
            ep = int(parts[-2].replace("e", ""))
            max_ep = max(max_ep, ep)
        except (ValueError, IndexError):
            pass
    return max_ep


def train(exp):
    name = exp["name"]
    sr = exp["sr"]
    epochs = exp["epochs"]
    save_every = exp["save_every"]
    batch = exp["batch"]
    pg = exp["pg"]
    pd = exp["pd"]

    exp_dir = os.path.join(APPLIO_DIR, "logs", name)

    # Check current state
    max_ep = get_max_epoch(exp_dir, name)
    if max_ep >= epochs:
        print(f"  Already at {max_ep}ep >= {epochs}ep, skipping")
        latest = sorted(glob.glob(os.path.join(exp_dir, f"{name}_*.pth")))[-1]
        return latest

    print(f"  Current max epoch: {max_ep}")

    # Verify pretrained models
    pg_path = os.path.join(APPLIO_DIR, pg)
    pd_path = os.path.join(APPLIO_DIR, pd)
    for p, label in [(pg_path, "G"), (pd_path, "D")]:
        if not os.path.exists(p):
            print(f"  ERROR: pretrained {label} not found: {p}")
            return None

    # Verify preprocessing
    filelist = os.path.join(exp_dir, "filelist.txt")
    if not os.path.exists(filelist):
        print(f"  ERROR: filelist not found at {filelist}")
        return None

    # Check resume checkpoint exists
    g_ckpt = os.path.join(exp_dir, "G_2333333.pth")
    d_ckpt = os.path.join(exp_dir, "D_2333333.pth")
    if os.path.exists(g_ckpt):
        import torch
        ckpt = torch.load(g_ckpt, map_location="cpu", weights_only=False)
        resume_ep = ckpt.get("iteration", "?")
        del ckpt
        print(f"  Will resume from G_2333333.pth at epoch {resume_ep}")

    cmd = [
        VENV_PY, "rvc/train/train.py",
        name,
        str(save_every),
        str(epochs),
        pg_path,
        pd_path,
        "-",  # GPU = CPU
        str(batch),
        str(sr),
        "True",   # save_only_latest — overwrite G/D each save
        "True",   # save_every_weights
        "False",  # cache_data_in_gpu
        "False",  # overtraining_detector
        "50",     # overtraining_threshold
        "False",  # cleanup
        "HiFi-GAN",  # vocoder
        "False",  # checkpointing
    ]

    env = os.environ.copy()
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = "29525"

    print(f"  Training: {name} → {epochs}ep")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=APPLIO_DIR, env=env, timeout=36000)
    elapsed = time.time() - t0

    new_max = get_max_epoch(exp_dir, name)
    if new_max > max_ep:
        models = sorted(glob.glob(os.path.join(exp_dir, f"{name}_*.pth")))
        latest = models[-1] if models else None
        print(f"  Done in {elapsed:.0f}s ({elapsed/60:.0f}min), max epoch now: {new_max}")
        return latest
    else:
        print(f"  FAILED: no new checkpoints (rc={r.returncode}, {elapsed:.0f}s)")
        return None


def main():
    print("=" * 60)
    print("  RESUME TRAINING: TITAN 48k + SnowieV3.1 48k → 200ep")
    print("  Resumes from G_2333333.pth checkpoints")
    print("=" * 60)

    results = {}
    for exp in EXPERIMENTS:
        name = exp["name"]
        print(f"\n{'='*60}")
        print(f"  {name} (sr={exp['sr']})")
        print(f"{'='*60}")

        model = train(exp)
        results[name] = model

    print(f"\n{'='*60}")
    print("  TRAINING SUMMARY")
    print(f"{'='*60}")
    for name, model in results.items():
        if model:
            print(f"  {name:<20} {os.path.basename(model)}")
        else:
            print(f"  {name:<20} FAILED")


if __name__ == "__main__":
    main()
