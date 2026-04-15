"""Setup dataset structure for so-vits-svc-fork training."""
import os
import shutil

# Create dataset structure: dataset_raw/{speaker_name}/
speaker_dir = '/Users/sarsa/claude/million-dollar-tuner/dataset_raw/user'
os.makedirs(speaker_dir, exist_ok=True)
print(f'Created: {speaker_dir}')

# Copy the recording to the dataset directory
src = '/Users/sarsa/Downloads/recording.wav'
dst = os.path.join(speaker_dir, 'recording.wav')
shutil.copy2(src, dst)
print(f'Copied recording to: {dst}')
print(f'File size: {os.path.getsize(dst)} bytes')
