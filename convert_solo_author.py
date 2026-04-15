"""Convert Solo Author vocals with Applio 100ep model."""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

APPLIO_DIR = "/Users/sarsa/claude/Applio"
MDT_DIR = "/Users/sarsa/claude/million-dollar-tuner"
MODEL_PATH = f"{APPLIO_DIR}/logs/yj_voice/yj_voice_100e_9800s.pth"
INPUT_VOCALS = f"{MDT_DIR}/stems_solo_author/vocals.wav"
OUTPUT_VOCALS = "/Users/sarsa/Downloads/solo_author_converted_vocals.wav"
OUTPUT_MIX = "/Users/sarsa/Downloads/solo_author_converted.wav"
STEMS_DIR = f"{MDT_DIR}/stems_solo_author"

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


def main():
    print("=== Converting Solo Author with Applio 100ep ===")
    print(f"Input: {INPUT_VOCALS}")
    print(f"Model: {MODEL_PATH}")

    # Step 1: Run inference
    script = INFER_SCRIPT.format(
        applio_dir=APPLIO_DIR,
        model_path=MODEL_PATH,
        input_audio=INPUT_VOCALS,
        output_audio=OUTPUT_VOCALS,
    )

    fd, script_path = tempfile.mkstemp(suffix=".py", prefix="mdt_solo_")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(script)

        env = os.environ.copy()
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "29532"

        pth = find_rvc_pth()
        bak = None
        if pth and pth.exists():
            bak = pth.with_suffix(".pth.bak")
            pth.rename(bak)

        try:
            t0 = time.time()
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=APPLIO_DIR,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
            elapsed = time.time() - t0
        finally:
            if pth and bak and bak.exists():
                bak.rename(pth)

        print(f"Inference: {elapsed:.1f}s, rc={result.returncode}")
        if result.stdout:
            print(f"  stdout: {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"  stderr: {result.stderr[-500:]}")
            return
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    if not os.path.exists(OUTPUT_VOCALS):
        print("ERROR: no output produced")
        return

    # Step 2: Mix with instrumentals
    print("\nMixing with instrumentals...")
    sys.path.insert(0, MDT_DIR)
    from mdt.vc.mixer import mix_with_stems
    mix_with_stems(
        vocals_path=OUTPUT_VOCALS,
        stems_dir=STEMS_DIR,
        output_path=OUTPUT_MIX,
        reference_vocals_path=INPUT_VOCALS,
    )

    # Step 3: Quick evaluation
    print("\nEvaluating...")
    from mdt.vc.metrics import evaluate
    result = evaluate(
        user_audio="/Users/sarsa/Downloads/yj_voice.wav",
        source_vocals=INPUT_VOCALS,
        converted_vocals=OUTPUT_VOCALS,
    )
    print(f"  Speaker sim to user: {result.sim_to_user:.4f}")
    print(f"  F0 correlation: {result.f0_correlation:.4f}")
    print(f"  F0 mean error: {result.f0_mean_error_hz:.1f} Hz")

    info = sf.info(OUTPUT_MIX)
    print(f"\nOutput: {OUTPUT_MIX} ({info.duration:.1f}s)")
    print(f"Vocals: {OUTPUT_VOCALS}")


if __name__ == "__main__":
    main()
