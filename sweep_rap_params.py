"""Sweep inference params for rap sections only.

Extracts rap sections from vocals, runs Applio inference with different
protect/index_rate combinations, evaluates each, and picks the best config.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from itertools import product
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

MDT_DIR = Path("/Users/sarsa/claude/million-dollar-tuner")
APPLIO_DIR = Path("/Users/sarsa/claude/Applio")
SOURCE_VOCALS = MDT_DIR / "stems_dtr" / "vocals.wav"
USER_AUDIO = Path("/Users/sarsa/Downloads/yj_voice.wav")
SECTIONS_PATH = MDT_DIR / "song_sections.json"
SWEEP_DIR = Path("/Users/sarsa/Downloads/voice_benchmark/rap_sweep")

# TITAN 48k 25ep — current best base per 3-base A/B benchmark
MODEL_PATH = str(APPLIO_DIR / "logs/yj_titan48k/yj_titan48k_25e_2450s.pth")
INDEX_PATH = ""  # FAISS disabled (segfault on macOS)

# Parameters to sweep
PROTECT_VALUES = [0.0, 0.33, 0.5]
# NOTE: FAISS index causes segfault on macOS — only test index_rate=0
# If we get FAISS working later, re-add 0.5 and 0.75
INDEX_RATE_VALUES = [0.0]

SR = 48000  # TITAN 48k outputs at 48kHz

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
    file_index={index_path!r} if {index_path!r} else "",
    index_rate={index_rate},
    pitch_guidance=bool(use_f0),
    volume_envelope=1.0,
    version=version,
    protect={protect},
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


def extract_rap_sections():
    """Extract only rap sections from source vocals (with 0.5s padding)."""
    with open(SECTIONS_PATH) as f:
        sections = json.load(f)["sections"]

    y, sr = sf.read(str(SOURCE_VOCALS))
    if y.ndim > 1:
        y = y.mean(axis=1)

    rap_sections = [s for s in sections if s["type"] == "rap"]
    chunks = []
    for sec in rap_sections:
        start = max(0, sec["start"] - 0.5)
        end = min(len(y) / sr, sec["end"] + 0.5)
        s = int(start * sr)
        e = int(end * sr)
        chunks.append(y[s:e])

    # Concatenate with 0.3s silence between
    silence = np.zeros(int(0.3 * sr), dtype=np.float32)
    result = np.concatenate([x for chunk in chunks for x in [chunk, silence]][:-1])

    rap_path = SWEEP_DIR / "rap_sections_only.wav"
    sf.write(str(rap_path), result.astype(np.float32), sr)
    print(f"Extracted rap sections: {len(result)/sr:.1f}s")
    return str(rap_path), rap_sections


def run_inference(input_audio, output_path, protect, index_rate):
    """Run Applio inference as subprocess."""
    script = INFER_SCRIPT.format(
        applio_dir=str(APPLIO_DIR),
        model_path=MODEL_PATH,
        index_path=INDEX_PATH,
        input_audio=input_audio,
        output_audio=str(output_path),
        protect=protect,
        index_rate=index_rate,
    )

    fd, script_path = tempfile.mkstemp(suffix=".py", prefix="mdt_sweep_")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(script)

        env = os.environ.copy()
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "29531"

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
            print(f"  stderr: {result.stderr[-300:]}")
            return False
        return True
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def evaluate_rap(converted_path, label):
    """Evaluate converted rap audio against user voice."""
    sys.path.insert(0, str(MDT_DIR))
    from mdt.vc.metrics import load_mono, speaker_similarity, f0_correlation

    y_conv = load_mono(converted_path, SR)
    y_user = load_mono(str(USER_AUDIO), SR)

    sim = speaker_similarity(y_conv, y_user, SR)
    f0_corr, f0_err, _ = f0_correlation(y_conv, load_mono(str(SOURCE_VOCALS), SR), SR)

    # Artifact score
    from mdt.vc.section_eval import compute_artifact_score
    artifact = compute_artifact_score(y_conv, SR)

    return {
        "label": label,
        "speaker_sim": sim,
        "f0_corr": f0_corr,
        "f0_err_hz": f0_err,
        "artifact": artifact,
    }


def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    # Extract rap sections
    rap_audio_path, rap_sections = extract_rap_sections()

    results = []

    for protect, index_rate in product(PROTECT_VALUES, INDEX_RATE_VALUES):
        label = f"p{protect:.2f}_ir{index_rate:.2f}"
        output_path = SWEEP_DIR / f"rap_{label}.wav"

        if output_path.exists():
            print(f"\n--- {label}: already converted, evaluating ---")
        else:
            print(f"\n--- {label}: protect={protect}, index_rate={index_rate} ---")
            t0 = time.time()
            ok = run_inference(rap_audio_path, str(output_path), protect, index_rate)
            elapsed = time.time() - t0
            print(f"  Inference: {elapsed:.1f}s, success={ok}")
            if not ok:
                continue

        # Evaluate
        r = evaluate_rap(str(output_path), label)
        r["protect"] = protect
        r["index_rate"] = index_rate
        results.append(r)
        print(f"  sim={r['speaker_sim']:.4f} f0corr={r['f0_corr']:.4f} artifact={r['artifact']:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  RAP SECTION PARAM SWEEP (TITAN 48k 25ep)")
    print(f"{'='*70}")
    print(f"  {'protect':>8} {'idx_rate':>8} {'→User':>8} {'F0corr':>8} {'Artif':>8}")
    print(f"  {'-'*40}")
    for r in sorted(results, key=lambda x: -x["speaker_sim"]):
        print(f"  {r['protect']:>8.2f} {r['index_rate']:>8.2f} "
              f"{r['speaker_sim']:>8.4f} {r['f0_corr']:>8.4f} {r['artifact']:>8.4f}")

    # Find best (highest speaker_sim with artifact < 0.3)
    viable = [r for r in results if r["artifact"] < 0.3]
    if not viable:
        viable = results
    best = max(viable, key=lambda x: x["speaker_sim"])
    print(f"\n  BEST: protect={best['protect']}, index_rate={best['index_rate']} "
          f"(sim={best['speaker_sim']:.4f}, f0={best['f0_corr']:.4f}, art={best['artifact']:.4f})")

    with open(SWEEP_DIR / "rap_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
