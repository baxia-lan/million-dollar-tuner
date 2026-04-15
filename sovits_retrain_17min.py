"""
Full retrain pipeline for so-vits-svc-fork with 17-min voice dataset.
Steps: clean -> split -> preprocess -> train -> infer -> mix -> verify
"""
import os
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import shutil
import time
import json
import subprocess

import numpy as np
import soundfile as sf

PROJECT = '/Users/sarsa/claude/million-dollar-tuner'
os.chdir(PROJECT)

# Use sys.executable so the same Python that runs this script is used for subprocesses
PYTHON = sys.executable

VOICE_SRC = '/Users/sarsa/Downloads/yj_voice.wav'
VOCALS_SRC = 'stems_dtr/vocals.wav'
OUTPUT_VOCALS = '/Users/sarsa/Downloads/sovits_17min_vocals.wav'
OUTPUT_MIX = '/Users/sarsa/Downloads/sovits_17min_mix.wav'

SPEAKER = 'speaker0'
RAW_DIR = f'dataset_raw/{SPEAKER}'
DATASET_DIR = f'dataset/44k/{SPEAKER}'

# ============================================================
# STEP 1: Clean up old data
# ============================================================
print("=" * 60)
print("STEP 1: Cleaning up old data")
print("=" * 60)

# Clean dataset_raw
for d in ['dataset_raw', 'dataset/44k']:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"  Removed: {d}")

# Clean old filelists
if os.path.exists('filelists/44k'):
    shutil.rmtree('filelists/44k')
    print("  Removed: filelists/44k")

# Remove old checkpoints (keep G_0.pth and D_0.pth pretrained)
if os.path.exists('logs/44k'):
    for f in os.listdir('logs/44k'):
        fp = os.path.join('logs/44k', f)
        if os.path.isdir(fp):
            shutil.rmtree(fp)
            print(f"  Removed dir: {fp}")
        elif f not in ('G_0.pth', 'D_0.pth', 'config.json'):
            os.remove(fp)
            print(f"  Removed: {fp}")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
print("Done cleaning.\n")

# ============================================================
# STEP 2: Split 17-min recording into ~10s segments
# ============================================================
print("=" * 60)
print("STEP 2: Splitting 17-min recording into segments")
print("=" * 60)

audio, sr = sf.read(VOICE_SRC)
print(f"Source: {VOICE_SRC}")
print(f"Audio: {audio.shape}, sr={sr}, duration={len(audio)/sr:.1f}s")

if len(audio.shape) > 1:
    audio = audio.mean(axis=1)
    print(f"Converted to mono: {audio.shape}")

# Split into ~10s segments
segment_len = int(10 * sr)
segments = []
for i in range(0, len(audio), segment_len):
    seg = audio[i:i + segment_len]
    if len(seg) > sr * 2:  # at least 2 seconds
        segments.append(seg)

print(f"Created {len(segments)} segments (~10s each)")

# Save segments to dataset_raw
for i, seg in enumerate(segments):
    path = os.path.join(RAW_DIR, f'segment_{i:03d}.wav')
    sf.write(path, seg, sr)

print(f"Saved {len(segments)} segments to {RAW_DIR}")
total_dur = sum(len(s) for s in segments) / sr
print(f"Total training audio: {total_dur:.1f}s ({total_dur/60:.1f} minutes)\n")

# ============================================================
# STEP 3: Preprocessing
# ============================================================

def run_cmd(args, label, timeout=600):
    """Run a command and print output."""
    print(f"\n{'=' * 60}")
    print(f"STEP 3: {label}")
    print(f"{'=' * 60}")
    t0 = time.time()
    result = subprocess.run(
        args,
        capture_output=True, text=True, timeout=timeout,
        cwd=PROJECT,
    )
    elapsed = time.time() - t0
    if result.stdout:
        out = result.stdout
        if len(out) > 3000:
            print("... (truncated) ...")
            print(out[-3000:])
        else:
            print(out)
    if result.stderr:
        err = result.stderr
        if len(err) > 3000:
            print("STDERR: ... (truncated) ...")
            print(err[-3000:])
        else:
            print("STDERR:", err)
    print(f"{label} took {elapsed:.1f}s")
    if result.returncode != 0:
        print(f"ERROR: {label} failed with code {result.returncode}")
        sys.exit(1)
    return result

# 3a: pre-resample
run_cmd(
    [PYTHON, '-m', 'so_vits_svc_fork', 'pre-resample'],
    "pre-resample"
)

# 3b: pre-config
run_cmd(
    [PYTHON, '-m', 'so_vits_svc_fork', 'pre-config'],
    "pre-config"
)

# 3c: pre-hubert
run_cmd(
    [PYTHON, '-m', 'so_vits_svc_fork', 'pre-hubert',
     '-fm', 'dio', '-n', '1'],
    "pre-hubert",
    timeout=1200
)

# ============================================================
# STEP 4: Update config for more epochs + train
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Configure and train (400 epochs)")
print("=" * 60)

config_path = 'configs/44k/config.json'
with open(config_path) as f:
    config = json.load(f)

config['train']['epochs'] = 400
config['train']['eval_interval'] = 100
config['train']['keep_ckpts'] = 3
config['train']['batch_size'] = 6  # more data -> can use larger batch
config['train']['num_workers'] = 2

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Updated config: epochs=400, eval_interval=100, batch_size=6")
print(f"Config: {config_path}")

# Verify dataset files
n_files = len([f for f in os.listdir(DATASET_DIR) if f.endswith('.wav')])
n_hubert = len([f for f in os.listdir(DATASET_DIR) if f.endswith('.data.pt')])
print(f"Dataset: {n_files} wav files, {n_hubert} hubert files")

# Check filelists
for fl in ['filelists/44k/train.txt', 'filelists/44k/val.txt']:
    if os.path.exists(fl):
        with open(fl) as f:
            lines = f.readlines()
        print(f"  {fl}: {len(lines)} entries")

# Train
print("\nStarting training...")
t_train_start = time.time()

train_result = subprocess.run(
    [PYTHON, '-m', 'so_vits_svc_fork', 'train',
     '-c', config_path,
     '-m', 'logs/44k'],
    capture_output=True, text=True, timeout=7200,
    env={**os.environ, 'PYTORCH_ENABLE_MPS_FALLBACK': '1'},
    cwd=PROJECT,
)

train_elapsed = time.time() - t_train_start

stdout = train_result.stdout
if len(stdout) > 8000:
    print("... (truncated) ...")
    print(stdout[-8000:])
else:
    print(stdout)

if train_result.stderr:
    stderr = train_result.stderr
    if len(stderr) > 5000:
        print("STDERR: ... (truncated) ...")
        print(stderr[-5000:])
    else:
        print("STDERR:", stderr)

print(f"\nTraining took {train_elapsed:.1f}s ({train_elapsed/60:.1f} minutes)")
print(f"Return code: {train_result.returncode}")

if train_result.returncode != 0:
    print("WARNING: Training returned non-zero, checking for checkpoints anyway...")

# List checkpoints
print("\nCheckpoints:")
if os.path.exists('logs/44k'):
    for f in sorted(os.listdir('logs/44k')):
        fp = os.path.join('logs/44k', f)
        if os.path.isfile(fp):
            size_mb = os.path.getsize(fp) / 1024 / 1024
            print(f"  {f}: {size_mb:.1f} MB")

# Find best checkpoint
best_g = None
best_epoch = 0
for f in sorted(os.listdir('logs/44k')):
    if f.startswith('G_') and f.endswith('.pth') and f != 'G_0.pth':
        epoch = int(f.split('_')[1].split('.')[0])
        if epoch > best_epoch:
            best_epoch = epoch
            best_g = f

if best_g:
    print(f"\nBest checkpoint: {best_g} (epoch {best_epoch})")
else:
    print("\nWARNING: No training checkpoints found!")
    sys.exit(1)

# ============================================================
# STEP 5: Inference
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Inference - converting vocals")
print("=" * 60)

# Determine speaker name from config
with open(config_path) as f:
    cfg = json.load(f)
speakers = list(cfg.get('spk', {}).keys())
speaker_name = speakers[0] if speakers else SPEAKER
print(f"Speaker: {speaker_name}")

t_infer_start = time.time()

infer_result = subprocess.run(
    [PYTHON, '-m', 'so_vits_svc_fork', 'infer',
     '-o', OUTPUT_VOCALS,
     '-s', speaker_name,
     '-m', 'logs/44k',
     '-c', config_path,
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

# Check if output file was created (may have a different name/path)
if not os.path.exists(OUTPUT_VOCALS):
    print(f"Output not at {OUTPUT_VOCALS}, searching...")
    # so-vits-svc-fork may save with a suffix or in a different location
    dl_dir = '/Users/sarsa/Downloads'
    for f in sorted(os.listdir(dl_dir)):
        if 'sovits' in f.lower() or 'vocal' in f.lower():
            print(f"  Found: {os.path.join(dl_dir, f)}")
    # Also check the output dir pattern
    out_dir = os.path.dirname(OUTPUT_VOCALS)
    if os.path.exists(out_dir):
        for f in sorted(os.listdir(out_dir)):
            if f.startswith('sovits_17min'):
                found_path = os.path.join(out_dir, f)
                print(f"  Found: {found_path}")
                if not os.path.exists(OUTPUT_VOCALS):
                    OUTPUT_VOCALS = found_path

if not os.path.exists(OUTPUT_VOCALS):
    print("ERROR: No output file found!")
    sys.exit(1)
else:
    size_mb = os.path.getsize(OUTPUT_VOCALS) / 1024 / 1024
    print(f"Output: {OUTPUT_VOCALS} ({size_mb:.1f} MB)")

# ============================================================
# STEP 6: Mix with instrumentals
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Mixing with instrumentals")
print("=" * 60)

vocals, sr_v = sf.read(OUTPUT_VOCALS)
drums, sr_d = sf.read('stems_dtr/drums.wav')
bass, sr_b = sf.read('stems_dtr/bass.wav')
other, sr_o = sf.read('stems_dtr/other.wav')

print(f"Vocals: {vocals.shape}, sr={sr_v}")
print(f"Drums:  {drums.shape}, sr={sr_d}")
print(f"Bass:   {bass.shape}, sr={sr_b}")
print(f"Other:  {other.shape}, sr={sr_o}")

# Handle mono vocals
if len(vocals.shape) == 1:
    vocals_stereo = np.column_stack([vocals, vocals])
else:
    vocals_stereo = vocals

# Resample if needed
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

# Match lengths
target_len = min(len(drums), len(bass), len(other), len(vocals_stereo))
vocals_stereo = vocals_stereo[:target_len]
drums = drums[:target_len]
bass = bass[:target_len]
other = other[:target_len]

# Mix with slight vocal boost
mix = vocals_stereo * 1.2 + drums + bass + other
peak = np.max(np.abs(mix))
if peak > 0:
    mix = mix / peak * 0.95

sf.write(OUTPUT_MIX, mix, sr_d)
print(f"Mix saved: {OUTPUT_MIX} ({os.path.getsize(OUTPUT_MIX)/1024/1024:.1f} MB)")

# ============================================================
# STEP 7: Resemblyzer verification
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Resemblyzer speaker embedding verification")
print("=" * 60)

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    import librosa

    encoder = VoiceEncoder()

    # User voice (17-min reference)
    user_wav, _ = librosa.load(VOICE_SRC, sr=16000, mono=True)
    user_wav = preprocess_wav(user_wav)
    user_embed = encoder.embed_utterance(user_wav)
    print(f"User embedding: {user_embed.shape}")

    # Original SUNO vocals
    suno_wav, _ = librosa.load(VOCALS_SRC, sr=16000, mono=True)
    suno_wav = preprocess_wav(suno_wav)
    suno_embed = encoder.embed_utterance(suno_wav)
    print(f"SUNO embedding: {suno_embed.shape}")

    # Converted vocals
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
# STEP 8: F0 Correlation
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: F0 Correlation verification")
print("=" * 60)

try:
    import pyworld as pw
    import librosa

    source_wav, sr_src = librosa.load(VOCALS_SRC, sr=44100, mono=True)
    converted_wav, _ = librosa.load(OUTPUT_VOCALS, sr=44100, mono=True)

    min_len = min(len(source_wav), len(converted_wav))
    source_wav = source_wav[:min_len]
    converted_wav = converted_wav[:min_len]

    source_f0, _ = pw.harvest(source_wav.astype(np.float64), 44100)
    converted_f0, _ = pw.harvest(converted_wav.astype(np.float64), 44100)

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
print(f"Training data: {total_dur:.1f}s ({len(segments)} segments)")
print(f"Training time: {train_elapsed:.1f}s ({train_elapsed/60:.1f} minutes)")
print(f"Training epochs: 400 (best checkpoint: epoch {best_epoch})")
print(f"Inference time: {infer_elapsed:.1f}s")
print(f"Output vocals: {OUTPUT_VOCALS}")
print(f"Output mix: {OUTPUT_MIX}")
print("DONE!")
