"""Train 3 pretrained bases to 200ep each for A/B comparison.

Same data (yj_voice 17min), same save schedule (every 25ep).
Runs sequentially to maximize CPU throughput.
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
        "name": "yj_singer32k",
        "sr": 32000,
        "pg": "rvc/models/pretraineds/SingerPreTrain_32k/f0G_SingerPreTrain.pth",
        "pd": "rvc/models/pretraineds/SingerPreTrain_32k/f0D_SingerPreTrain.pth",
        "epochs": 200,
        "save_every": 25,
        "batch": 4,
    },
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


def train(exp):
    name = exp["name"]
    sr = exp["sr"]
    epochs = exp["epochs"]
    save_every = exp["save_every"]
    batch = exp["batch"]
    pg = exp["pg"]
    pd = exp["pd"]

    exp_dir = os.path.join(APPLIO_DIR, "logs", name)

    # Check if already trained enough
    existing = sorted(glob.glob(os.path.join(exp_dir, f"{name}_*.pth")))
    if existing:
        last = os.path.basename(existing[-1])
        # Parse epoch from filename like yj_singer32k_200e_19600s.pth
        try:
            ep = int(last.split("_")[-2].replace("e", ""))
            if ep >= epochs:
                print(f"  Already at {ep}ep >= {epochs}ep target, skipping")
                return existing[-1]
        except (ValueError, IndexError):
            pass

    # Verify pretrained models exist
    pg_path = os.path.join(APPLIO_DIR, pg)
    pd_path = os.path.join(APPLIO_DIR, pd)
    if not os.path.exists(pg_path):
        print(f"  ERROR: pretrained G not found: {pg_path}")
        return None
    if not os.path.exists(pd_path):
        print(f"  ERROR: pretrained D not found: {pd_path}")
        return None

    # Verify preprocessing was done
    filelist = os.path.join(exp_dir, "filelist.txt")
    if not os.path.exists(filelist):
        print(f"  ERROR: filelist not found at {filelist}")
        print(f"  Need to run preprocessing first")
        return None

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
        "True",   # save_only_latest
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

    print(f"  Training cmd: {' '.join(cmd[-10:])}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=APPLIO_DIR, env=env, timeout=36000)
    elapsed = time.time() - t0

    # Applio exits with os._exit(2333333) on success
    models = sorted(glob.glob(os.path.join(exp_dir, f"{name}_*.pth")))
    if models:
        latest = models[-1]
        print(f"  Done in {elapsed:.0f}s ({elapsed/60:.0f}min), latest: {os.path.basename(latest)}")
        return latest
    else:
        print(f"  FAILED: no model file produced (rc={r.returncode}, {elapsed:.0f}s)")
        return None


def main():
    print("=" * 60)
    print("  3-BASE A/B TRAINING")
    print("  Same data: yj_voice 17min")
    print("  Target: 200ep, save every 25")
    print("=" * 60)

    results = {}
    for exp in EXPERIMENTS:
        name = exp["name"]
        print(f"\n{'='*60}")
        print(f"  {name} (sr={exp['sr']})")
        print(f"  Pretrained: {exp['pg']}")
        print(f"{'='*60}")

        model = train(exp)
        results[name] = model
        if model:
            print(f"  SUCCESS: {model}")
        else:
            print(f"  FAILED")

    print(f"\n{'='*60}")
    print("  TRAINING SUMMARY")
    print(f"{'='*60}")
    for name, model in results.items():
        status = os.path.basename(model) if model else "FAILED"
        print(f"  {name:<20} {status}")


if __name__ == "__main__":
    main()
