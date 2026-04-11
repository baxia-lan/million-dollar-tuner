#!/usr/bin/env bash
# Million Dollar Tuner — Demo Workflow
# ====================================
#
# Prerequisites:
#   pip install -e .
#   sudo apt install rubberband-cli fluidsynth ffmpeg
#
# Replace the file paths below with your actual files.

set -e

SUNO_SONG="suno_song.mp3"
MY_VOICE="my_recording.wav"
OUTPUT="my_tuned_song.wav"

echo "====================================="
echo " Million Dollar Tuner Demo"
echo "====================================="

# Step 1: Analyze the SUNO song
echo ""
echo "--- Analyzing SUNO song ---"
mdt analyze "$SUNO_SONG" --detailed

# Step 2: Separate stems (optional, for inspection)
echo ""
echo "--- Separating stems ---"
mdt separate "$SUNO_SONG" -o ./demo_stems/

# Step 3: Replace vocals with your voice
echo ""
echo "--- Replacing vocals ---"
mdt tune "$MY_VOICE" "$SUNO_SONG" -o "$OUTPUT" \
    --strength 0.85 \
    --reverb-room 0.3 \
    --reverb-wet 0.15

echo ""
echo "Done! Output: $OUTPUT"
echo ""

# Optional: Simple auto-tune (no reference song needed)
echo "--- Bonus: Simple auto-tune ---"
mdt autotune "$MY_VOICE" -o autotuned_voice.wav --strength 0.9

# Optional: Resynthesize (experimental)
echo ""
echo "--- Bonus: Resynthesis (experimental) ---"
mdt resynth "$SUNO_SONG" -o ./demo_resynth/ --midi-only
