import os, sys, types, time, shutil, glob, json

APPLIO_DIR = "/Users/sarsa/claude/Applio"
sys.path = [p for p in sys.path if "rvc-clean" not in p]
if APPLIO_DIR in sys.path:
    sys.path.remove(APPLIO_DIR)
sys.path.insert(0, APPLIO_DIR)
os.chdir(APPLIO_DIR)

# Patch torchfcpe
fake = types.ModuleType("torchfcpe")
fake.spawn_infer_model_from_pt = None
sys.modules["torchfcpe"] = fake

model_name = "yj_singer32k"
sample_rate = 32000
user_voice = "/Users/sarsa/Downloads/yj_voice.wav"

exp_dir = os.path.join("logs", model_name)
os.makedirs(exp_dir, exist_ok=True)

# Create dataset directory with symlink/copy
dataset_dir = os.path.join(exp_dir, "dataset")
os.makedirs(dataset_dir, exist_ok=True)
target_wav = os.path.join(dataset_dir, "yj_voice.wav")
if not os.path.exists(target_wav):
    shutil.copy2(user_voice, target_wav)

from rvc.train.preprocess.preprocess import PreProcess, save_dataset_duration

print(f"Preprocessing {model_name} at {sample_rate}Hz...")
start = time.time()
pp = PreProcess(sr=sample_rate, exp_dir=exp_dir)

audio_length = pp.process_audio(
    path=target_wav,
    idx0=0,
    sid=0,
    cut_preprocess="Automatic",
    process_effects=True,
    noise_reduction=False,
    reduction_strength=0.7,
    chunk_len=3.0,
    overlap_len=0.3,
    normalization_mode="pre",
)

save_dataset_duration(
    os.path.join(exp_dir, "model_info.json"),
    dataset_duration=audio_length,
)

elapsed = time.time() - start
n_slices = len(glob.glob(os.path.join(exp_dir, "sliced_audios", "*.wav")))
n_16k = len(glob.glob(os.path.join(exp_dir, "sliced_audios_16k", "*.wav")))
print(f"Preprocessing done in {elapsed:.1f}s")
print(f"Sliced audios: {n_slices}, 16k versions: {n_16k}")
print(f"Audio duration: {audio_length:.1f}s")
