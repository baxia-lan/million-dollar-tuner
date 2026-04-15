"""Split user recording into segments and preprocess for so-vits-svc-fork."""
import subprocess
import sys
import os
import time
import soundfile as sf
import numpy as np

os.chdir('/Users/sarsa/claude/million-dollar-tuner')

# ============================================================
# Step 0: Split recording into ~10-second segments
# ============================================================
print("=" * 60)
print("STEP 0: Splitting recording into segments")
print("=" * 60)

# Read the resampled file
audio, sr = sf.read('dataset/44k/user/recording.wav')
print(f"Audio: {audio.shape}, sr={sr}, duration={len(audio)/sr:.1f}s")

# If stereo, convert to mono
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)
    print(f"Converted to mono: {audio.shape}")

# Split into ~10s segments
segment_len = int(10 * sr)  # 10 seconds
segments = []
for i in range(0, len(audio), segment_len):
    seg = audio[i:i+segment_len]
    if len(seg) > sr * 2:  # at least 2 seconds
        segments.append(seg)

print(f"Created {len(segments)} segments")

# Save segments
out_dir = 'dataset/44k/user'
# Remove the original single file
orig = os.path.join(out_dir, 'recording.wav')
if os.path.exists(orig):
    os.remove(orig)

for i, seg in enumerate(segments):
    path = os.path.join(out_dir, f'segment_{i:03d}.wav')
    sf.write(path, seg, sr)
    print(f"  {path}: {len(seg)/sr:.1f}s")

# Also update dataset_raw with the same split
raw_dir = 'dataset_raw/user'
for f in os.listdir(raw_dir):
    os.remove(os.path.join(raw_dir, f))
for i, seg in enumerate(segments):
    path = os.path.join(raw_dir, f'segment_{i:03d}.wav')
    sf.write(path, seg, sr)

print(f"\nTotal segments: {len(segments)}")

# ============================================================
# Step 1: pre-config
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: pre-config")
print("=" * 60)
t0 = time.time()
result = subprocess.run(
    [sys.executable, '-m', 'so_vits_svc_fork', 'pre-config',
     '-i', 'dataset/44k', '-c', 'configs/44k/config.json'],
    capture_output=True, text=True
)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print(f"pre-config took {time.time()-t0:.1f}s")
if result.returncode != 0:
    print(f"ERROR: pre-config failed with code {result.returncode}")
    sys.exit(1)

# ============================================================
# Step 2: pre-hubert
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: pre-hubert (downloads HuBERT model if needed)")
print("=" * 60)
t0 = time.time()
result = subprocess.run(
    [sys.executable, '-m', 'so_vits_svc_fork', 'pre-hubert',
     '-i', 'dataset/44k', '-c', 'configs/44k/config.json',
     '-fm', 'dio', '-n', '1'],
    capture_output=True, text=True, timeout=600
)
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)
print(f"pre-hubert took {time.time()-t0:.1f}s")
if result.returncode != 0:
    print(f"ERROR: pre-hubert failed with code {result.returncode}")
    sys.exit(1)

print("\nAll preprocessing complete!")
