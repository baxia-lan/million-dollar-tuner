"""Mix converted vocals with instrumentals and verify quality."""
import os
import numpy as np
import soundfile as sf
import librosa

MDT_DIR = "/Users/sarsa/claude/million-dollar-tuner"
USER_VOICE = "/Users/sarsa/Downloads/yj_voice.wav"
SUNO_VOCALS = os.path.join(MDT_DIR, "stems_dtr", "vocals.wav")
DRUMS = os.path.join(MDT_DIR, "stems_dtr", "drums.wav")
BASS = os.path.join(MDT_DIR, "stems_dtr", "bass.wav")
OTHER = os.path.join(MDT_DIR, "stems_dtr", "other.wav")
OUTPUT_VOCALS = "/Users/sarsa/Downloads/applio_vocals.wav"
OUTPUT_MIX = "/Users/sarsa/Downloads/applio_mix.wav"

# Mix
print("=== Mixing ===")
vocals, sr_v = sf.read(OUTPUT_VOCALS)
print(f"Vocals: sr={sr_v}, shape={vocals.shape}")

drums, _ = librosa.load(DRUMS, sr=sr_v, mono=True)
bass, _ = librosa.load(BASS, sr=sr_v, mono=True)
other, _ = librosa.load(OTHER, sr=sr_v, mono=True)

if len(vocals.shape) > 1:
    vocals = vocals.mean(axis=1)

target_len = max(len(drums), len(bass), len(other), len(vocals))
def pad_to(arr, length):
    if len(arr) < length:
        return np.pad(arr, (0, length - len(arr)))
    return arr[:length]

vocals = pad_to(vocals, target_len)
drums = pad_to(drums, target_len)
bass = pad_to(bass, target_len)
other = pad_to(other, target_len)

mix = vocals * 1.0 + drums * 0.9 + bass * 0.9 + other * 0.85
peak = np.abs(mix).max()
if peak > 0:
    mix = mix / peak * 0.95

sf.write(OUTPUT_MIX, mix, sr_v)
print(f"Mix saved: {OUTPUT_MIX}")

# Verify
print("\n=== Speaker Similarity (resemblyzer) ===")
from resemblyzer import VoiceEncoder, preprocess_wav
encoder = VoiceEncoder()

user_wav = preprocess_wav(USER_VOICE)
suno_wav = preprocess_wav(SUNO_VOCALS)
output_wav = preprocess_wav(OUTPUT_VOCALS)

emb_user = encoder.embed_utterance(user_wav)
emb_suno = encoder.embed_utterance(suno_wav)
emb_output = encoder.embed_utterance(output_wav)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

sim_output_user = cosine_sim(emb_output, emb_user)
sim_output_suno = cosine_sim(emb_output, emb_suno)
sim_user_suno = cosine_sim(emb_user, emb_suno)

print(f"  Output vs User voice: {sim_output_user:.4f}")
print(f"  Output vs SUNO:       {sim_output_suno:.4f}")
print(f"  User vs SUNO:         {sim_user_suno:.4f}")

print("\n=== F0 Correlation ===")
suno_audio, _ = librosa.load(SUNO_VOCALS, sr=16000, mono=True)
output_audio, _ = librosa.load(OUTPUT_VOCALS, sr=16000, mono=True)

f0_suno, _, _ = librosa.pyin(suno_audio, fmin=50, fmax=1100, sr=16000)
f0_output, _, _ = librosa.pyin(output_audio, fmin=50, fmax=1100, sr=16000)

min_len = min(len(f0_suno), len(f0_output))
f0_suno = f0_suno[:min_len]
f0_output = f0_output[:min_len]

mask = ~(np.isnan(f0_suno) | np.isnan(f0_output))
if mask.sum() > 10:
    f0_corr = np.corrcoef(f0_suno[mask], f0_output[mask])[0, 1]
    print(f"  F0 correlation: {f0_corr:.4f}")
    print(f"  Voiced frames:  {mask.sum()} / {min_len}")
else:
    print("  Too few voiced frames")

print("\n=== DONE ===")
