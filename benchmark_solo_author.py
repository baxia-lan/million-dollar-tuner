"""Benchmark all 3 backends on Solo Author."""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

VENV_PY = "/Users/sarsa/claude/million-dollar-tuner/.venv/bin/python"
MDT_DIR = Path("/Users/sarsa/claude/million-dollar-tuner")
APPLIO_DIR = Path("/Users/sarsa/claude/Applio")
STEMS_DIR = MDT_DIR / "stems_solo_author"
SOURCE_VOCALS = STEMS_DIR / "vocals.wav"
USER_AUDIO = Path("/Users/sarsa/Downloads/yj_voice.wav")
OUTPUT_DIR = MDT_DIR / "output" / "solo_author_benchmark"

# Best Applio checkpoint (80ep based on benchmark)
APPLIO_MODEL = str(APPLIO_DIR / "logs/yj_voice/yj_voice_80e_7840s.pth")

INFER_SCRIPT = '''
import os, sys, types
import numpy as np

APPLIO_DIR = {applio_dir!r}
sys.path = [p for p in sys.path if "rvc-clean" not in p]
if APPLIO_DIR in sys.path:
    sys.path.remove(APPLIO_DIR)
sys.path.insert(0, APPLIO_DIR)
os.chdir(APPLIO_DIR)

_fake = types.ModuleType("torchfcpe")
_fake.spawn_infer_model_from_pt = None
sys.modules["torchfcpe"] = _fake

import torch
import soundfile as sf
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.utils import load_audio_infer, load_embedding
from rvc.configs.config import Config
from rvc.infer.pipeline import Pipeline

config = Config()

cpt = torch.load({model_path!r}, map_location="cpu", weights_only=True)
tgt_sr = cpt["config"][-1]
cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
use_f0 = cpt.get("f0", 1)
version = cpt.get("version", "v1")
text_enc_hidden_dim = 768 if version == "v2" else 256
vocoder = cpt.get("vocoder", "HiFi-GAN")

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

hubert_model = load_embedding("contentvec").to(config.device).float()
hubert_model.eval()

audio = load_audio_infer({input_audio!r}, 16000)
audio_max = np.abs(audio).max() / 0.95
if audio_max > 1:
    audio /= audio_max

vc = Pipeline(tgt_sr, config)

audio_opt = vc.pipeline(
    model=hubert_model,
    net_g=net_g,
    sid=0,
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

sf.write({output_audio!r}, audio_opt, tgt_sr, format="WAV")
print(f"OK: {{len(audio_opt)/tgt_sr:.1f}}s at {{tgt_sr}}Hz")
'''


def find_rvc_pth():
    prefix = Path(sys.prefix)
    candidates = list(prefix.glob("lib/python*/site-packages/rvc.pth"))
    return candidates[0] if candidates else None


def run_applio(output_path):
    """Run Applio inference."""
    script = INFER_SCRIPT.format(
        applio_dir=str(APPLIO_DIR),
        model_path=APPLIO_MODEL,
        input_audio=str(SOURCE_VOCALS),
        output_audio=str(output_path),
    )
    fd, script_path = tempfile.mkstemp(suffix=".py", prefix="mdt_ab_")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(script)
        env = os.environ.copy()
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "29533"
        pth = find_rvc_pth()
        bak = None
        if pth and pth.exists():
            bak = pth.with_suffix(".pth.bak")
            pth.rename(bak)
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=str(APPLIO_DIR), env=env,
                capture_output=True, text=True, timeout=600,
            )
        finally:
            if pth and bak and bak.exists():
                bak.rename(pth)
        return result.returncode == 0
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def run_sovits(output_path):
    """Run so-vits-svc inference."""
    # Find the trained sovits model from previous experiments
    sovits_logs = Path("/Users/sarsa/claude/million-dollar-tuner/sovits_workdir/logs/44k")
    if not sovits_logs.exists():
        # Try alternative location
        sovits_logs = Path("/Users/sarsa/claude/million-dollar-tuner/sovits_17min/logs/44k")
    if not sovits_logs.exists():
        print("  sovits: no trained model found, skipping")
        return False

    config_path = sovits_logs.parent.parent / "configs" / "44k" / "config.json"
    if not config_path.exists():
        print(f"  sovits: config not found at {config_path}")
        return False

    cmd = [
        sys.executable, "-m", "so_vits_svc_fork", "infer",
        "-o", str(output_path),
        "-m", str(sovits_logs),
        "-c", str(config_path),
        "-fm", "dio",
        "-d", "cpu",
        "-t", "0",
        str(SOURCE_VOCALS),
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        cwd=str(sovits_logs.parent.parent),
    )
    return result.returncode == 0 and output_path.exists()


def run_seedvc(output_path):
    """Run seed-vc inference."""
    sys.path.insert(0, str(MDT_DIR))
    from mdt.vc.seedvc import SeedVCBackend

    backend = SeedVCBackend()
    # Use the user's voice as reference
    ref_dir = OUTPUT_DIR / "seedvc_ref"
    ref_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.copy2(str(USER_AUDIO), str(ref_dir / "reference.wav"))

    try:
        result = backend.infer(
            profile_dir=ref_dir,
            source_vocals=SOURCE_VOCALS,
            output_path=output_path,
            diffusion_steps=25,
            inference_cfg_rate=0.7,
        )
        return output_path.exists()
    except Exception as e:
        print(f"  seedvc error: {e}")
        return False


def evaluate(converted_path, label):
    """Run evaluation."""
    sys.path.insert(0, str(MDT_DIR))
    from mdt.vc.metrics import evaluate as eval_fn
    result = eval_fn(str(USER_AUDIO), str(SOURCE_VOCALS), str(converted_path))
    return {
        "label": label,
        "sim_to_user": result.sim_to_user,
        "f0_correlation": result.f0_correlation,
        "f0_mean_error_hz": result.f0_mean_error_hz,
    }


def mix(vocals_path, output_path):
    """Mix with instrumentals."""
    sys.path.insert(0, str(MDT_DIR))
    from mdt.vc.mixer import mix_with_stems
    mix_with_stems(
        vocals_path=str(vocals_path),
        stems_dir=str(STEMS_DIR),
        output_path=str(output_path),
        reference_vocals_path=str(SOURCE_VOCALS),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    backends = [
        ("applio_80ep", run_applio),
        ("seedvc", run_seedvc),
    ]

    for label, run_fn in backends:
        vocals_path = OUTPUT_DIR / f"{label}_vocals.wav"
        mix_path = OUTPUT_DIR / f"{label}_mix.wav"

        if vocals_path.exists():
            print(f"\n--- {label}: already converted ---")
        else:
            print(f"\n--- {label}: converting ---")
            t0 = time.time()
            ok = run_fn(vocals_path)
            elapsed = time.time() - t0
            print(f"  {elapsed:.1f}s, success={ok}")
            if not ok:
                continue

        # Evaluate
        print(f"  Evaluating {label}...")
        r = evaluate(vocals_path, label)
        results[label] = r
        print(f"  sim={r['sim_to_user']:.4f} f0={r['f0_correlation']:.4f} err={r['f0_mean_error_hz']:.1f}Hz")

        # Mix
        print(f"  Mixing {label}...")
        mix(vocals_path, mix_path)

    # sovits separately (needs special working dir)
    label = "sovits"
    vocals_path = OUTPUT_DIR / f"{label}_vocals.wav"
    mix_path = OUTPUT_DIR / f"{label}_mix.wav"
    if not vocals_path.exists():
        print(f"\n--- {label}: converting ---")
        t0 = time.time()
        ok = run_sovits(vocals_path)
        elapsed = time.time() - t0
        print(f"  {elapsed:.1f}s, success={ok}")
    else:
        ok = True
        print(f"\n--- {label}: already converted ---")

    if ok and vocals_path.exists():
        print(f"  Evaluating {label}...")
        r = evaluate(vocals_path, label)
        results[label] = r
        print(f"  sim={r['sim_to_user']:.4f} f0={r['f0_correlation']:.4f} err={r['f0_mean_error_hz']:.1f}Hz")
        print(f"  Mixing {label}...")
        mix(vocals_path, mix_path)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SOLO AUTHOR - BACKEND COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Backend':<16} {'→User':>8} {'F0corr':>8} {'F0err':>8}")
    print(f"  {'-'*40}")
    for label, r in sorted(results.items(), key=lambda x: -x[1]['sim_to_user']):
        print(f"  {label:<16} {r['sim_to_user']:>8.4f} {r['f0_correlation']:>8.4f} {r['f0_mean_error_hz']:>7.1f}Hz")

    with open(OUTPUT_DIR / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
