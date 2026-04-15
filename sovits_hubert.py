"""Run pre-hubert step for so-vits-svc-fork."""
import subprocess
import sys
import os
import time

os.chdir('/Users/sarsa/claude/million-dollar-tuner')

print("=" * 60)
print("pre-hubert (downloads HuBERT model if needed)")
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
    stderr = result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr
    print("STDERR:", stderr)
elapsed = time.time() - t0
print(f"\npre-hubert took {elapsed:.1f}s")
if result.returncode != 0:
    print(f"ERROR: pre-hubert failed with code {result.returncode}")
    sys.exit(1)

# Check what was created
for root, dirs, files in os.walk('dataset/44k'):
    for f in files:
        if f.endswith(('.pt', '.npy', '.f0.npy')):
            fp = os.path.join(root, f)
            print(f"  {fp} ({os.path.getsize(fp)} bytes)")

print("\npre-hubert complete!")
