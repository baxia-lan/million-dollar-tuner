"""Run all so-vits-svc-fork preprocessing steps."""
import subprocess
import sys
import os
import time

os.chdir('/Users/sarsa/claude/million-dollar-tuner')

# Step 1: pre-resample
print("=" * 60)
print("STEP 1: pre-resample")
print("=" * 60)
t0 = time.time()
result = subprocess.run(
    [sys.executable, '-m', 'so_vits_svc_fork', 'pre-resample',
     '-i', 'dataset_raw', '-o', 'dataset/44k', '-s', '44100'],
    capture_output=True, text=True
)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print(f"pre-resample took {time.time()-t0:.1f}s")
if result.returncode != 0:
    print(f"ERROR: pre-resample failed with code {result.returncode}")
    sys.exit(1)

# Check output
for root, dirs, files in os.walk('dataset/44k'):
    for f in files:
        fp = os.path.join(root, f)
        print(f"  output: {fp} ({os.path.getsize(fp)} bytes)")

# Step 2: pre-config
print("\n" + "=" * 60)
print("STEP 2: pre-config")
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

# Step 3: pre-hubert
print("\n" + "=" * 60)
print("STEP 3: pre-hubert (this downloads HuBERT model if needed)")
print("=" * 60)
t0 = time.time()
result = subprocess.run(
    [sys.executable, '-m', 'so_vits_svc_fork', 'pre-hubert',
     '-i', 'dataset/44k', '-c', 'configs/44k/config.json',
     '-fm', 'dio'],
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
