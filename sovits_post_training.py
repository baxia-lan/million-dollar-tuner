"""
Post-training pipeline: inference, mix, and verify.
Run this after training completes or is stopped.
Uses the best available checkpoint.
"""
import os
import sys
import time
import json
import subprocess

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT = '/Users/sarsa/claude/million-dollar-tuner'
os.chdir(PROJECT)

PYTHON = sys.executable

VOICE_SRC = '/Users/sarsa/Downloads/yj_voice.wav'
VOCALS_SRC = 'stems_dtr/vocals.wav'
OUTPUT_VOCALS = '/Users/sarsa/Downloads/sovits_17min_vocals.wav'
OUTPUT_MIX = '/Users/sarsa/Downloads/sovits_17min_mix.wav'

CONFIG_PATH = 'configs/44k/config.json'

# ============================================================
# Find best checkpoint
# ============================================================
print("=" * 60)
print("Finding best checkpoint")
print("=" * 60)

best_g = None
best_epoch = 0
for f in sorted(os.listdir('logs/44k')):
    if f.startswith('G_') and f.endswith('.pth') and f != 'G_0.pth':
        epoch = int(f.split('_')[1].split('.')[0])
        if epoch > best_epoch:
            best_epoch = epoch
            best_g = f

if best_g:
    print(f"Best checkpoint: {best_g} (epoch {best_epoch})")
else:
    print("ERROR: No training checkpoints found!")
    sys.exit(1)

# List all checkpoints
print("\nAll checkpoints:")
for f in sorted(os.listdir('logs/44k')):
    fp = os.path.join('logs/44k', f)
    if os.path.isfile(fp):
        size_mb = os.path.getsize(fp) / 1024 / 1024
        print(f"  {f}: {size_mb:.1f} MB")

# ============================================================
# Determine speaker name from config
# ============================================================
with open(CONFIG_PATH) as f:
    cfg = json.load(f)
speakers = list(cfg.get('spk', {}).keys())
speaker_name = speakers[0] if speakers else 'speaker0'
print(f"\nSpeaker: {speaker_name}")

# ============================================================
# INFERENCE
# ============================================================
print("\n" + "=" * 60)
print("INFERENCE: Converting SUNO vocals with trained model")
print("=" * 60)
print(f"Input: {VOCALS_SRC}")
print(f"Output: {OUTPUT_VOCALS}")

t_infer_start = time.time()

infer_result = subprocess.run(
    [PYTHON, '-m', 'so_vits_svc_fork', 'infer',
     '-o', OUTPUT_VOCALS,
     '-s', speaker_name,
     '-m', 'logs/44k',
     '-c', CONFIG_PATH,
     '-fm', 'dio',
     '-d', 'cpu',
     '-t', '0',
     '-n', '0.4',
     '-p', '0.5',
     '-ch', '0.5',
     '-mc', '30',
     VOCALS_SRC],
    capture_output=True, text=True, timeout=1800,
    cwd=PROJECT,
)

infer_elapsed = time.time() - t_infer_start

stdout = infer_result.stdout
if len(stdout) > 5000:
    print("... (truncated) ...")
    print(stdout[-5000:])
else:
    print(stdout)

if infer_result.stderr:
    stderr = infer_result.stderr
    if len(stderr) > 3000:
        print("STDERR: ... (truncated) ...")
        print(stderr[-3000:])
    else:
        print("STDERR:", stderr)

print(f"\nInference took {infer_elapsed:.1f}s ({infer_elapsed/60:.1f} minutes)")
print(f"Return code: {infer_result.returncode}")

# Check if output file was created
if not os.path.exists(OUTPUT_VOCALS):
    print(f"Output not at {OUTPUT_VOCALS}, searching...")
    dl_dir = '/Users/sarsa/Downloads'
    for f in sorted(os.listdir(dl_dir)):
        if 'sovits' in f.lower() or 'vocal' in f.lower():
            found = os.path.join(dl_dir, f)
            print(f"  Found: {found}")
            if not os.path.exists(OUTPUT_VOCALS):
                OUTPUT_VOCALS = found

if not os.path.exists(OUTPUT_VOCALS):
    print("ERROR: No output file found!")
    sys.exit(1)
else:
    size_mb = os.path.getsize(OUTPUT_VOCALS) / 1024 / 1024
    print(f"Output: {OUTPUT_VOCALS} ({size_mb:.1f} MB)")

# ============================================================
# MIX
# ============================================================
print("\n" + "=" * 60)
print("MIXING with instrumentals")
print("=" * 60)

vocals, sr_v = sf.read(OUTPUT_VOCALS)
drums, sr_d = sf.read('stems_dtr/drums.wav')
bass, sr_b = sf.read('stems_dtr/bass.wav')
other, sr_o = sf.read('stems_dtr/other.wav')

print(f"Vocals: {vocals.shape}, sr={sr_v}")
print(f"Drums:  {drums.shape}, sr={sr_d}")
print(f"Bass:   {bass.shape}, sr={sr_b}")
print(f"Other:  {other.shape}, sr={sr_o}")

if len(vocals.shape) == 1:
    vocals_stereo = np.column_stack([vocals, vocals])
else:
    vocals_stereo = vocals

if sr_v != sr_d:
    import librosa
    print(f"Resampling vocals from {sr_v} to {sr_d}...")
    if len(vocals.shape) == 1:
        vocals_resampled = librosa.resample(vocals, orig_sr=sr_v, target_sr=sr_d)
        vocals_stereo = np.column_stack([vocals_resampled, vocals_resampled])
    else:
        left = librosa.resample(vocals[:, 0], orig_sr=sr_v, target_sr=sr_d)
        right = librosa.resample(vocals[:, 1], orig_sr=sr_v, target_sr=sr_d)
        vocals_stereo = np.column_stack([left, right])
    sr_v = sr_d

target_len = min(len(drums), len(bass), len(other), len(vocals_stereo))
vocals_stereo = vocals_stereo[:target_len]
drums = drums[:target_len]
bass = bass[:target_len]
other = other[:target_len]

mix = vocals_stereo * 1.2 + drums + bass + other
peak = np.max(np.abs(mix))
if peak > 0:
    mix = mix / peak * 0.95

sf.write(OUTPUT_MIX, mix, sr_d)
print(f"Mix saved: {OUTPUT_MIX} ({os.path.getsize(OUTPUT_MIX)/1024/1024:.1f} MB)")

# ============================================================
# RESEMBLYZER VERIFICATION
# ============================================================
print("\n" + "=" * 60)
print("RESEMBLYZER speaker embedding verification")
print("=" * 60)

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    import librosa

    encoder = VoiceEncoder()

    user_wav, _ = librosa.load(VOICE_SRC, sr=16000, mono=True)
    user_wav = preprocess_wav(user_wav)
    user_embed = encoder.embed_utterance(user_wav)
    print(f"User embedding: {user_embed.shape}")

    suno_wav, _ = librosa.load(VOCALS_SRC, sr=16000, mono=True)
    suno_wav = preprocess_wav(suno_wav)
    suno_embed = encoder.embed_utterance(suno_wav)
    print(f"SUNO embedding: {suno_embed.shape}")

    converted_wav, _ = librosa.load(OUTPUT_VOCALS, sr=16000, mono=True)
    converted_wav = preprocess_wav(converted_wav)
    converted_embed = encoder.embed_utterance(converted_wav)
    print(f"Converted embedding: {converted_embed.shape}")

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_user_suno = cosine_sim(user_embed, suno_embed)
    sim_user_converted = cosine_sim(user_embed, converted_embed)
    sim_suno_converted = cosine_sim(suno_embed, converted_embed)

    print(f"\nResemblyzer Cosine Similarities:")
    print(f"  User <-> SUNO (original):  {sim_user_suno:.4f}")
    print(f"  User <-> Converted:        {sim_user_converted:.4f}")
    print(f"  SUNO <-> Converted:        {sim_suno_converted:.4f}")
    print(f"\n  Improvement (User<->Conv - User<->SUNO): {sim_user_converted - sim_user_suno:+.4f}")

except ImportError as e:
    print(f"Resemblyzer not available: {e}")
except Exception as e:
    print(f"Resemblyzer error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# F0 CORRELATION
# ============================================================
print("\n" + "=" * 60)
print("F0 Correlation verification")
print("=" * 60)

try:
    import pyworld as pw
    import librosa

    source_wav, sr_src = librosa.load(VOCALS_SRC, sr=44100, mono=True)
    converted_wav_f0, _ = librosa.load(OUTPUT_VOCALS, sr=44100, mono=True)

    min_len = min(len(source_wav), len(converted_wav_f0))
    source_wav = source_wav[:min_len]
    converted_wav_f0 = converted_wav_f0[:min_len]

    source_f0, _ = pw.harvest(source_wav.astype(np.float64), 44100)
    converted_f0, _ = pw.harvest(converted_wav_f0.astype(np.float64), 44100)

    min_f0_len = min(len(source_f0), len(converted_f0))
    source_f0 = source_f0[:min_f0_len]
    converted_f0 = converted_f0[:min_f0_len]

    voiced_mask = (source_f0 > 0) & (converted_f0 > 0)
    n_voiced = voiced_mask.sum()
    n_total = len(source_f0)

    print(f"Total frames: {n_total}")
    print(f"Voiced frames (both): {n_voiced} ({n_voiced/n_total*100:.1f}%)")

    if n_voiced > 10:
        source_voiced = source_f0[voiced_mask]
        converted_voiced = converted_f0[voiced_mask]

        correlation = np.corrcoef(source_voiced, converted_voiced)[0, 1]
        mean_diff = np.mean(converted_voiced - source_voiced)
        mean_ratio = np.mean(converted_voiced / source_voiced)

        print(f"\nF0 Correlation (voiced frames): {correlation:.4f}")
        print(f"Mean F0 difference: {mean_diff:.1f} Hz")
        print(f"Mean F0 ratio (converted/source): {mean_ratio:.3f}")
        print(f"Source F0 mean: {source_voiced.mean():.1f} Hz, std: {source_voiced.std():.1f} Hz")
        print(f"Converted F0 mean: {converted_voiced.mean():.1f} Hz, std: {converted_voiced.std():.1f} Hz")
    else:
        print("Not enough voiced frames for F0 correlation")

except Exception as e:
    print(f"F0 analysis error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Best checkpoint: epoch {best_epoch}")
print(f"Inference time: {infer_elapsed:.1f}s")
print(f"Output vocals: {OUTPUT_VOCALS}")
print(f"Output mix: {OUTPUT_MIX}")
print("DONE!")
