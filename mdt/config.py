"""Shared configuration and defaults."""

from pathlib import Path

# Sample rates
OUTPUT_SR = 44100       # Output sample rate
ANALYSIS_SR = 22050     # Analysis sample rate (librosa default)

# Demucs
DEFAULT_MODEL = "htdemucs_ft"
STEM_NAMES = ["vocals", "drums", "bass", "other"]

# Pitch correction
DEFAULT_CORRECTION_STRENGTH = 0.8
MAX_PITCH_SHIFT_SEMITONES = 4
PITCH_FMIN = 65.0   # C2
PITCH_FMAX = 2093.0 # C7

# Time alignment
MAX_LENGTH_RATIO_DIFF = 0.3  # Warn if user recording differs by >30%

# Mixer
DEFAULT_REVERB_ROOM = 0.3
DEFAULT_REVERB_WET = 0.15
DEFAULT_COMPRESSOR_THRESHOLD = -20.0
DEFAULT_COMPRESSOR_RATIO = 3.0
DEFAULT_HIGHPASS_FREQ = 80.0

# Resynthesis
DEFAULT_SOUNDFONT = Path("soundfonts/GeneralUser_GS.sf2")

# Analysis
HOP_LENGTH = 512
