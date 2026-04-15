"""3-Base A/B Benchmark: compare SingerPreTrain 32k, TITAN 48k, SnowieV3.1 48k.

Same data (yj_voice 17min), section-level evaluation on dtr.wav.
Tests all available checkpoints across all 3 bases.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MDT_DIR = Path("/Users/sarsa/claude/million-dollar-tuner")
APPLIO_DIR = Path("/Users/sarsa/claude/Applio")
STEMS_DIR = MDT_DIR / "stems_dtr"
SOURCE_VOCALS = STEMS_DIR / "vocals.wav"
USER_AUDIO = Path("/Users/sarsa/Downloads/yj_voice.wav")
SECTIONS_PATH = MDT_DIR / "song_sections.json"
BENCH_DIR = Path("/Users/sarsa/Downloads/voice_benchmark/3base_ab")

# NOTE: FAISS index causes segfault on macOS — use index_rate=0.0
INDEX_RATE = 0.0

# Discover checkpoints for each base
BASES = {
    "singer32k": {
        "log_dir": "yj_singer32k",
        "sr": 32000,
    },
    "titan48k": {
        "log_dir": "yj_titan48k",
        "sr": 48000,
    },
    "snowie48k": {
        "log_dir": "yj_snowie48k",
        "sr": 48000,
    },
}


def discover_checkpoints(base_name, base_info):
    """Find all named weight files for a base."""
    log_dir = APPLIO_DIR / "logs" / base_info["log_dir"]
    prefix = base_info["log_dir"]
    checkpoints = {}
    for pth in sorted(log_dir.glob(f"{prefix}_*e_*s.pth")):
        name = pth.stem  # e.g. yj_singer32k_200e_19600s
        parts = name.split("_")
        try:
            ep = int(parts[-2].replace("e", ""))
            checkpoints[ep] = str(pth)
        except (ValueError, IndexError):
            continue
    return checkpoints


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
    index_rate={index_rate},
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


def run_inference(model_path, output_path):
    """Run Applio inference as subprocess."""
    script = INFER_SCRIPT.format(
        applio_dir=str(APPLIO_DIR),
        model_path=model_path,
        input_audio=str(SOURCE_VOCALS),
        output_audio=str(output_path),
        index_rate=INDEX_RATE,
    )

    fd, script_path = tempfile.mkstemp(suffix=".py", prefix="mdt_3base_")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(script)

        env = os.environ.copy()
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "29535"

        pth = find_rvc_pth()
        bak = None
        if pth and pth.exists():
            bak = pth.with_suffix(".pth.bak")
            pth.rename(bak)

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=str(APPLIO_DIR),
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
        finally:
            if pth and bak and bak.exists():
                bak.rename(pth)

        if result.returncode != 0:
            print(f"  FAILED (rc={result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr[-500:]}")
            return False
        return True
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def run_section_eval(converted_path, label):
    """Run section-level evaluation."""
    sys.path.insert(0, str(MDT_DIR))
    from mdt.vc.section_eval import evaluate_sections, print_report, save_report

    report = evaluate_sections(
        user_audio_path=str(USER_AUDIO),
        source_vocals_path=str(SOURCE_VOCALS),
        converted_vocals_path=str(converted_path),
        sections_path=str(SECTIONS_PATH),
        model_name=label,
    )
    save_report(report, BENCH_DIR / f"section_report_{label}.json")
    return report


def main():
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    # Discover all checkpoints
    all_checkpoints = {}
    for base_name, base_info in BASES.items():
        ckpts = discover_checkpoints(base_name, base_info)
        all_checkpoints[base_name] = ckpts
        print(f"{base_name}: {sorted(ckpts.keys())}ep available")

    # Run inference + eval for each base/checkpoint combo
    results = {}
    for base_name, ckpts in all_checkpoints.items():
        for ep in sorted(ckpts.keys()):
            label = f"{base_name}_{ep}ep"
            model_path = ckpts[ep]
            output_path = BENCH_DIR / f"{label}.wav"

            if output_path.exists():
                print(f"\n--- {label}: already converted ---")
            else:
                print(f"\n--- {label}: inference ---")
                t0 = time.time()
                ok = run_inference(model_path, str(output_path))
                elapsed = time.time() - t0
                print(f"  {elapsed:.1f}s, ok={ok}")
                if not ok:
                    continue

            # Section evaluation
            print(f"  Evaluating {label}...")
            try:
                report = run_section_eval(str(output_path), label)
                results[label] = {
                    "base": base_name,
                    "epoch": ep,
                    "rap_low_sim": report.rap_low_similarity,
                    "melodic_sim": report.melodic_similarity,
                    "melodic_pitch": report.melodic_pitch_corr,
                    "artifact": report.overall_artifact,
                    "stability": report.longform_stability,
                }
                print(f"  rap={report.rap_low_similarity:.4f} mel={report.melodic_similarity:.4f} "
                      f"pitch={report.melodic_pitch_corr:.4f} art={report.overall_artifact:.4f}")
            except Exception as e:
                print(f"  Eval error: {e}")

    # Summary tables — one per base
    print(f"\n{'='*90}")
    print(f"  3-BASE A/B COMPARISON (dtr.wav, same 17min data)")
    print(f"{'='*90}")

    for base_name in BASES:
        base_results = {k: v for k, v in results.items() if v["base"] == base_name}
        if not base_results:
            print(f"\n  {base_name}: no results")
            continue
        print(f"\n  --- {base_name} ---")
        print(f"  {'Epoch':<8} {'Rap→User':>10} {'Mel→User':>10} {'MelPitch':>10} {'Artifact':>10} {'Stability':>10}")
        print(f"  {'-'*58}")
        for label in sorted(base_results, key=lambda k: base_results[k]["epoch"]):
            r = base_results[label]
            print(f"  {r['epoch']:<8} {r['rap_low_sim']:>10.4f} {r['melodic_sim']:>10.4f} "
                  f"{r['melodic_pitch']:>10.4f} {r['artifact']:>10.4f} {r['stability']:>10.4f}")

    # Cross-base best comparison
    print(f"\n  --- BEST PER BASE (highest melodic_sim) ---")
    print(f"  {'Base':<16} {'Best':>6} {'Rap→User':>10} {'Mel→User':>10} {'MelPitch':>10} {'Artifact':>10}")
    print(f"  {'-'*60}")
    for base_name in BASES:
        base_results = {k: v for k, v in results.items() if v["base"] == base_name}
        if not base_results:
            print(f"  {base_name:<16} {'N/A':>6}")
            continue
        best = max(base_results.values(), key=lambda r: r["melodic_sim"])
        print(f"  {base_name:<16} {best['epoch']:>4}ep {best['rap_low_sim']:>10.4f} "
              f"{best['melodic_sim']:>10.4f} {best['melodic_pitch']:>10.4f} {best['artifact']:>10.4f}")

    # Save all results
    with open(BENCH_DIR / "3base_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {BENCH_DIR}")


if __name__ == "__main__":
    main()
