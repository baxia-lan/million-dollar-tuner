"""Train so-vits-svc-fork model on CPU with minimal config."""
import subprocess
import sys
import os
import time

os.chdir('/Users/sarsa/claude/million-dollar-tuner')

print("=" * 60)
print("TRAINING so-vits-svc-fork (CPU, 100 epochs, batch_size=4)")
print("=" * 60)

t0 = time.time()
result = subprocess.run(
    [sys.executable, '-m', 'so_vits_svc_fork', 'train',
     '-c', 'configs/44k/config.json',
     '-m', 'logs/44k'],
    capture_output=True, text=True, timeout=3600,
    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_ENABLE_MPS_FALLBACK': '1'}
)

elapsed = time.time() - t0

# Print last chunk of output
stdout = result.stdout
if len(stdout) > 5000:
    print("... (truncated) ...")
    print(stdout[-5000:])
else:
    print(stdout)

if result.stderr:
    stderr = result.stderr
    if len(stderr) > 5000:
        print("STDERR: ... (truncated) ...")
        print(stderr[-5000:])
    else:
        print("STDERR:", stderr)

print(f"\nTraining took {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
print(f"Return code: {result.returncode}")

# Check for model checkpoints
if os.path.exists('logs/44k'):
    for f in sorted(os.listdir('logs/44k')):
        fp = os.path.join('logs/44k', f)
        if os.path.isfile(fp):
            size_mb = os.path.getsize(fp) / 1024 / 1024
            print(f"  {f}: {size_mb:.1f} MB")
