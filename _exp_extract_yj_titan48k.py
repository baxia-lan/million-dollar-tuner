import os, sys, types, time, glob, json
import numpy as np

APPLIO_DIR = "/Users/sarsa/claude/Applio"
sys.path = [p for p in sys.path if "rvc-clean" not in p]
if APPLIO_DIR in sys.path:
    sys.path.remove(APPLIO_DIR)
sys.path.insert(0, APPLIO_DIR)
os.chdir(APPLIO_DIR)

fake = types.ModuleType("torchfcpe")
fake.spawn_infer_model_from_pt = None
sys.modules["torchfcpe"] = fake

model_name = "yj_titan48k"
sample_rate = 48000
exp_dir = os.path.join("logs", model_name)

wav_path = os.path.join(exp_dir, "sliced_audios_16k")
os.makedirs(os.path.join(exp_dir, "f0"), exist_ok=True)
os.makedirs(os.path.join(exp_dir, "f0_voiced"), exist_ok=True)
os.makedirs(os.path.join(exp_dir, "extracted"), exist_ok=True)

files = []
for f in sorted(glob.glob(os.path.join(wav_path, "*.wav"))):
    fn = os.path.basename(f)
    files.append([
        f,
        os.path.join(exp_dir, "f0", fn + ".npy"),
        os.path.join(exp_dir, "f0_voiced", fn + ".npy"),
        os.path.join(exp_dir, "extracted", fn.replace("wav", "npy")),
    ])
print(f"Files to process: {len(files)}")

print("Extracting F0 with rmvpe...")
start = time.time()

from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from rvc.lib.utils import load_audio_16k as _load_16k
import torch
from tqdm import tqdm

class SimpleFeatureInput:
    def __init__(self, device="cpu"):
        self.hop_size = 160
        self.sample_rate = 16000
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = device
        self.model = RMVPE0Predictor(
            os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
            device=self.device,
        )

    def compute_f0(self, x):
        return self.model.infer_from_audio(x, thred=0.03)

    def coarse_f0(self, f0):
        f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
        f0_mel = np.clip(
            (f0_mel - self.f0_mel_min) * (self.f0_bin - 2) / (self.f0_mel_max - self.f0_mel_min) + 1,
            1, self.f0_bin - 1,
        )
        return np.rint(f0_mel).astype(int)

    def process_file(self, file_info):
        inp_path, opt_path_coarse, opt_path_full, _ = file_info
        if os.path.exists(opt_path_coarse) and os.path.exists(opt_path_full):
            return
        try:
            np_arr = _load_16k(inp_path)
            feature_pit = self.compute_f0(np_arr)
            np.save(opt_path_full, feature_pit, allow_pickle=False)
            coarse_pit = self.coarse_f0(feature_pit)
            np.save(opt_path_coarse, coarse_pit, allow_pickle=False)
        except Exception as error:
            print(f"Error extracting F0 from {inp_path}: {error}")

fe = SimpleFeatureInput(device="cpu")
for fi in tqdm(files, desc="F0"):
    fe.process_file(fi)

f0_time = time.time() - start
print(f"F0 extraction done in {f0_time:.1f}s")

print("Extracting embeddings with contentvec...")
start = time.time()

from rvc.lib.utils import load_audio_16k, load_embedding
model = load_embedding("contentvec").to("cpu").float()
model.eval()

for fi in tqdm(files, desc="Embeddings"):
    wav_file_path, _, _, out_file_path = fi
    if os.path.exists(out_file_path):
        continue
    feats = torch.from_numpy(load_audio_16k(wav_file_path)).float()
    feats = feats.view(1, -1)
    with torch.no_grad():
        result = model(feats)["last_hidden_state"]
    feats_out = result.squeeze(0).float().cpu().numpy()
    if not np.isnan(feats_out).any():
        np.save(out_file_path, feats_out, allow_pickle=False)
    else:
        print(f"WARNING: {wav_file_path} produced NaN values; skipping.")

del model
emb_time = time.time() - start
print(f"Embedding extraction done in {emb_time:.1f}s")

model_info_path = os.path.join(exp_dir, "model_info.json")
if os.path.exists(model_info_path):
    with open(model_info_path, "r") as f:
        data = json.load(f)
else:
    data = {}
data["embedder_model"] = "contentvec"
with open(model_info_path, "w") as f:
    json.dump(data, f, indent=4)

from rvc.train.extract.preparing_files import generate_config, generate_filelist
generate_config(sample_rate, exp_dir)
generate_filelist(exp_dir, sample_rate, include_mutes=2)

n_f0 = len(glob.glob(os.path.join(exp_dir, "f0", "*.npy")))
n_emb = len(glob.glob(os.path.join(exp_dir, "extracted", "*.npy")))
print(f"F0 files: {n_f0}, Embedding files: {n_emb}")

filelist_path = os.path.join(exp_dir, "filelist.txt")
with open(filelist_path, "r") as f:
    lines = f.readlines()
print(f"Filelist entries: {len(lines)}")
