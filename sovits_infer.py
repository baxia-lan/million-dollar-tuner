"""Run so-vits-svc-fork inference to convert vocals."""
import subprocess
import sys
import os
import time

os.chdir('/Users/sarsa/claude/million-dollar-tuner')

input_path = 'stems_dtr/vocals.wav'
output_path = '/Users/sarsa/Downloads/sovits_vocals.wav'

print("=" * 60)
print("INFERENCE: Converting SUNO vocals with trained model")
print("=" * 60)
print(f"Input: {input_path}")
print(f"Output: {output_path}")

t0 = time.time()
result = subprocess.run(
    [sys.executable, '-m', 'so_vits_svc_fork', 'infer',
     '-o', output_path,
     '-s', 'user',
     '-m', 'logs/44k',
     '-c', 'configs/44k/config.json',
     '-fm', 'dio',
     '-d', 'cpu',
     '-t', '0',
     '-n', '0.4',
     '-p', '0.5',
     '-ch', '0.5',
     '-mc', '30',
     input_path],
    capture_output=True, text=True, timeout=1800
)

elapsed = time.time() - t0

stdout = result.stdout
if len(stdout) > 5000:
    print("... (truncated) ...")
    print(stdout[-5000:])
else:
    print(stdout)

if result.stderr:
    stderr = result.stderr
    if len(stderr) > 3000:
        print("STDERR: ... (truncated) ...")
        print(stderr[-3000:])
    else:
        print("STDERR:", stderr)

print(f"\nInference took {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
print(f"Return code: {result.returncode}")

if os.path.exists(output_path):
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Output file: {output_path} ({size_mb:.1f} MB)")
else:
    print(f"WARNING: Output file not found at {output_path}")
    # Check if it saved with a different name
    dl_dir = '/Users/sarsa/Downloads'
    for f in sorted(os.listdir(dl_dir)):
        if 'sovits' in f.lower() or 'vocal' in f.lower():
            print(f"  Found: {os.path.join(dl_dir, f)}")
