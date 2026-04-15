"""Batch benchmark: run inference on multiple Applio checkpoints and evaluate."""
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
BENCH_DIR = Path("/Users/sarsa/Downloads/voice_benchmark")
# NOTE: FAISS index causes segfault on macOS — use index_rate=0.0
INDEX_PATH = ""

# Checkpoints to benchmark (skip 50ep — already have results)
CHECKPOINTS = {
    "60ep": str(APPLIO_DIR / "logs/yj_voice/yj_voice_60e_5880s.pth"),
    "70ep": str(APPLIO_DIR / "logs/yj_voice/yj_voice_70e_6860s.pth"),
    "80ep": str(APPLIO_DIR / "logs/yj_voice/yj_voice_80e_7840s.pth"),
    "90ep": str(APPLIO_DIR / "logs/yj_voice/yj_voice_90e_8820s.pth"),
    "100ep": str(APPLIO_DIR / "logs/yj_voice/yj_voice_100e_9800s.pth"),
}

# Inference script template
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

MODEL_PATH = {model_path!r}
INDEX_PATH = {index_path!r}
INPUT_AUDIO = {input_audio!r}
OUTPUT_AUDIO = {output_audio!r}
PROTECT = {protect}
INDEX_RATE = {index_rate}

cpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
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

audio = load_audio_infer(INPUT_AUDIO, 16000)
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
    file_index=INDEX_PATH if INDEX_PATH else "",
    index_rate=INDEX_RATE,
    pitch_guidance=bool(use_f0),
    volume_envelope=1.0,
    version=version,
    protect=PROTECT,
    f0_autotune=False,
    f0_autotune_strength=1.0,
    proposed_pitch=False,
    proposed_pitch_threshold=155.0,
)

sf.write(OUTPUT_AUDIO, audio_opt, tgt_sr, format="WAV")
print(f"OK: {{len(audio_opt)/tgt_sr:.1f}}s at {{tgt_sr}}Hz")
'''


def find_rvc_pth():
    prefix = Path(sys.prefix)
    candidates = list(prefix.glob("lib/python*/site-packages/rvc.pth"))
    return candidates[0] if candidates else None


def run_inference(model_path, output_path, index_path="", protect=0.5, index_rate=0.0):
    """Run Applio inference as subprocess."""
    script = INFER_SCRIPT.format(
        applio_dir=str(APPLIO_DIR),
        model_path=model_path,
        index_path=index_path,
        input_audio=str(SOURCE_VOCALS),
        output_audio=str(output_path),
        protect=protect,
        index_rate=index_rate,
    )

    fd, script_path = tempfile.mkstemp(suffix=".py", prefix="mdt_bench_")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(script)

        env = os.environ.copy()
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "29530"

        # Disable rvc.pth
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
    print_report(report)
    save_report(report, BENCH_DIR / f"section_report_{label}.json")
    return report


def main():
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for label, model_path in CHECKPOINTS.items():
        if not Path(model_path).exists():
            print(f"\n--- SKIP {label}: model not found ---")
            continue

        output_path = BENCH_DIR / f"bench_{label}.wav"

        if output_path.exists():
            print(f"\n--- {label}: already converted, evaluating ---")
        else:
            print(f"\n--- {label}: running inference ---")
            t0 = time.time()
            ok = run_inference(model_path, str(output_path), index_path="", index_rate=0.0)
            elapsed = time.time() - t0
            print(f"  Inference: {elapsed:.1f}s, success={ok}")
            if not ok:
                continue

        # Evaluate
        report = run_section_eval(str(output_path), label)
        results[label] = {
            "rap_low_sim": report.rap_low_similarity,
            "melodic_sim": report.melodic_similarity,
            "melodic_pitch": report.melodic_pitch_corr,
            "artifact": report.overall_artifact,
            "stability": report.longform_stability,
        }

    # Summary table
    print(f"\n{'='*80}")
    print(f"  CHECKPOINT COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Epoch':<8} {'Rap→User':>10} {'Mel→User':>10} {'MelPitch':>10} {'Artifact':>10} {'Stability':>10}")
    print(f"  {'-'*58}")
    for label in sorted(results.keys()):
        r = results[label]
        print(f"  {label:<8} {r['rap_low_sim']:>10.4f} {r['melodic_sim']:>10.4f} "
              f"{r['melodic_pitch']:>10.4f} {r['artifact']:>10.4f} {r['stability']:>10.4f}")

    # Save summary
    with open(BENCH_DIR / "checkpoint_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {BENCH_DIR / 'checkpoint_comparison.json'}")


if __name__ == "__main__":
    main()
