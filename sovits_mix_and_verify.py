"""Mix converted vocals with instrumentals and verify with resemblyzer + F0 correlation."""
import numpy as np
import soundfile as sf
import os
import time

os.chdir('/Users/sarsa/claude/million-dollar-tuner')

# ============================================================
# STEP 1: Mix converted vocals with instrumentals
# ============================================================
print("=" * 60)
print("STEP 1: Mixing converted vocals with instrumentals")
print("=" * 60)

# Load all stems
vocals, sr_v = sf.read('/Users/sarsa/Downloads/sovits_vocals.wav')
drums, sr_d = sf.read('stems_dtr/drums.wav')
bass, sr_b = sf.read('stems_dtr/bass.wav')
other, sr_o = sf.read('stems_dtr/other.wav')

print(f"Vocals: {vocals.shape}, sr={sr_v}")
print(f"Drums:  {drums.shape}, sr={sr_d}")
print(f"Bass:   {bass.shape}, sr={sr_b}")
print(f"Other:  {other.shape}, sr={sr_o}")

# Handle mono vocals - convert to stereo if needed
if len(vocals.shape) == 1:
    vocals_stereo = np.column_stack([vocals, vocals])
else:
    vocals_stereo = vocals

# All instrumentals should be at same sr; resample vocals if different
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
    print(f"Resampled vocals: {vocals_stereo.shape}")

# Ensure all same length
target_len = min(len(drums), len(bass), len(other), len(vocals_stereo))
vocals_stereo = vocals_stereo[:target_len]
drums = drums[:target_len]
bass = bass[:target_len]
other = other[:target_len]

# Mix: boost vocals slightly
mix = vocals_stereo * 1.2 + drums + bass + other

# Normalize
peak = np.max(np.abs(mix))
if peak > 0:
    mix = mix / peak * 0.95

output_mix = '/Users/sarsa/Downloads/sovits_mix.wav'
sf.write(output_mix, mix, sr_d)
print(f"Mix saved to: {output_mix} ({os.path.getsize(output_mix)/1024/1024:.1f} MB)")

# ============================================================
# STEP 2: Verify with resemblyzer speaker embeddings
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Resemblyzer speaker embedding verification")
print("=" * 60)

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    import librosa

    encoder = VoiceEncoder()

    # Load user's original recording
    user_wav, _ = librosa.load('/Users/sarsa/Downloads/recording.wav', sr=16000, mono=True)
    user_wav = preprocess_wav(user_wav)
    user_embed = encoder.embed_utterance(user_wav)
    print(f"User embedding shape: {user_embed.shape}")

    # Load the original SUNO vocals
    suno_wav, _ = librosa.load('stems_dtr/vocals.wav', sr=16000, mono=True)
    suno_wav = preprocess_wav(suno_wav)
    suno_embed = encoder.embed_utterance(suno_wav)
    print(f"SUNO embedding shape: {suno_embed.shape}")

    # Load the converted vocals
    converted_wav, _ = librosa.load('/Users/sarsa/Downloads/sovits_vocals.wav', sr=16000, mono=True)
    converted_wav = preprocess_wav(converted_wav)
    converted_embed = encoder.embed_utterance(converted_wav)
    print(f"Converted embedding shape: {converted_embed.shape}")

    # Cosine similarities
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
    print("Install with: pip install resemblyzer")

# ============================================================
# STEP 3: F0 Correlation
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: F0 Correlation verification")
print("=" * 60)

try:
    import pyworld as pw
    import librosa

    # Load source and converted at same sample rate
    source_wav, sr = librosa.load('stems_dtr/vocals.wav', sr=44100, mono=True)
    converted_wav, _ = librosa.load('/Users/sarsa/Downloads/sovits_vocals.wav', sr=44100, mono=True)

    # Ensure same length
    min_len = min(len(source_wav), len(converted_wav))
    source_wav = source_wav[:min_len]
    converted_wav = converted_wav[:min_len]

    # Extract F0
    source_f0, _ = pw.harvest(source_wav.astype(np.float64), sr)
    converted_f0, _ = pw.harvest(converted_wav.astype(np.float64), sr)

    # Ensure same length
    min_f0_len = min(len(source_f0), len(converted_f0))
    source_f0 = source_f0[:min_f0_len]
    converted_f0 = converted_f0[:min_f0_len]

    # Filter to voiced frames (both > 0)
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

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
