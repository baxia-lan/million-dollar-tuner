"""
Run Applio inference as a subprocess to avoid import/segfault issues.
"""
import os
import sys
import types

# Fix sys.path
APPLIO_DIR = "/Users/sarsa/claude/Applio"
MDT_DIR = "/Users/sarsa/claude/million-dollar-tuner"

sys.path = [p for p in sys.path if "rvc-clean" not in p]
if APPLIO_DIR in sys.path:
    sys.path.remove(APPLIO_DIR)
sys.path.insert(0, APPLIO_DIR)

os.chdir(APPLIO_DIR)

# Patch torchfcpe
fake_torchfcpe = types.ModuleType("torchfcpe")
fake_torchfcpe.spawn_infer_model_from_pt = None
sys.modules["torchfcpe"] = fake_torchfcpe

import torch
import numpy as np
import soundfile as sf
import librosa

# ---- Direct inference without VoiceConverter class ----
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "logs/yj_voice/yj_voice_5e_490s.pth"
INDEX_PATH = sys.argv[2] if len(sys.argv) > 2 else ""
INPUT_AUDIO = os.path.join(MDT_DIR, "stems_dtr", "vocals.wav")
OUTPUT_AUDIO = "/Users/sarsa/Downloads/applio_vocals.wav"

print(f"Model: {MODEL_PATH}")
print(f"Index: {INDEX_PATH}")
print(f"Input: {INPUT_AUDIO}")
print(f"Output: {OUTPUT_AUDIO}")

# Load model
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.utils import load_audio_infer, load_embedding
from rvc.configs.config import Config

config = Config()
print(f"Device: {config.device}")

# Load checkpoint
cpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
tgt_sr = cpt["config"][-1]
cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
use_f0 = cpt.get("f0", 1)
version = cpt.get("version", "v1")
text_enc_hidden_dim = 768 if version == "v2" else 256
vocoder = cpt.get("vocoder", "HiFi-GAN")

print(f"Target SR: {tgt_sr}, Version: {version}, Vocoder: {vocoder}, Use F0: {use_f0}")

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

# Load HuBERT/ContentVec
print("Loading ContentVec embedder...")
hubert_model = load_embedding("contentvec").to(config.device).float()
hubert_model.eval()

# Load audio at 16kHz
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
import time
start = time.time()

sid = 0
audio_opt = vc.pipeline(
    model=hubert_model,
    net_g=net_g,
    sid=sid,
    audio=audio,
    pitch=0,
    f0_method="rmvpe",
    file_index="",  # Skip index to avoid FAISS segfault
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
print(f"Conversion done in {elapsed:.1f}s")

# Save
sf.write(OUTPUT_AUDIO, audio_opt, tgt_sr, format="WAV")
print(f"Saved: {OUTPUT_AUDIO}")
print(f"Output shape: {audio_opt.shape}, SR: {tgt_sr}")
