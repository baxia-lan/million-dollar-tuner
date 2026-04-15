# Million Dollar Tuner (MDT) - Handoff Document

## Project Goal

Convert SUNO AI-generated songs to sound like the user's own voice while preserving the original pitch, rhythm, and melody. Target: release-quality vocal replacement, not demo-quality.

**Input:**
- SUNO songs (WAV, any sample rate)
- User voice recordings (17 minutes available, plan to add 8-15 min targeted supplement)

**Test Songs:**
- `~/Downloads/dtr.wav` — "越界 Duck The Rope" (235s, 44.1kHz, Chinese rap + melodic)
- `~/Downloads/Solo Author.wav` (149.7s, 48kHz, English)

**Voice Data:**
- `~/Downloads/yj_voice.wav` — 17.8 minutes of voice recording

## 7-Point Specification

### 1. Data Strategy
Supplement existing 17-min data with 8-15 minutes of targeted recording (low-register, rap rhythms, bilingual coverage). NOT 45-60 min bulk recording. See `RECORDING_GUIDE.md` for the recording protocol.

### 2. Training Experiments (3-way A/B)
Three pretrained bases, same data, same evaluation:

| Base | Sample Rate | Status |
|---|---|---|
| SingerPreTrain 32k | 32000 Hz | **200ep complete** |
| TITAN 48k | 48000 Hz | **Training, at ~75ep** |
| SnowieV3.1 48k | 48000 Hz | Queued after TITAN |

All training runs in Applio (RVC), 200 epochs, save every 25, batch size 4.

**Preliminary Winner: TITAN 48k** — at 25 epochs already matches singer32k at 200 epochs on all key metrics.

### 3. Inference Experiments
- Phrase-level chunked inference (`mdt/vc/chunked_infer.py`) — split at section boundaries, convert per-chunk, crossfade reassembly
- Rap section param sweep completed: `protect` parameter has minimal impact (sim 0.841-0.842 across all values)
- No global autotune

### 4. Frontend Audio (Stem Separation)
**Winner: htdemucs_6s** — 6 stems (vocals, drums, bass, other, guitar, piano). Better separation than htdemucs/htdemucs_ft.

### 5. Section-Level Evaluation
Five metrics computed per-section (rap, melodic, intro, outro):
- `rap_low_similarity` — speaker embedding similarity in rap sections
- `melodic_similarity` — speaker embedding similarity in melodic sections
- `melodic_pitch_corr` — F0 correlation in melodic sections
- `artifact_score` — spectral discontinuity (lower = cleaner)
- `longform_stability` — 1 - std(per_section_similarity)

### 6. Post-Processing
No dependency on manual DAW post-processing. Automated mixing via `mdt/vc/mixer.py`.

### 7. YingMusic-SVC
Exploratory only. Not started.

## Architecture

```
mdt/
  cli.py                  # Click CLI: profile, convert, benchmark, tune
  config.py               # Settings
  tuning/                 # Original DSP-based approach (mostly superseded)
    voice_convert.py      # WORLD vocoder + spectral transfer
    mixer.py              # Stem mixing (older)
    ...
  vc/                     # Neural voice conversion (active)
    base.py               # VCBackend Protocol + TrainResult/InferResult/EvalResult
    registry.py           # Backend discovery: get_backend(), get_best_backend()
    applio.py             # Applio RVC backend (subprocess isolation)
    sovits.py             # so-vits-svc-fork backend (subprocess CLI)
    seedvc.py             # Seed-VC V2 (in-process, zero-shot)
    mixer.py              # Mix converted vocals with instrumental stems
    metrics.py            # resemblyzer speaker similarity + F0 correlation (pyworld)
    section_eval.py       # Section-level benchmark (5 metrics)
    chunked_infer.py      # Phrase-level inference with overlap/crossfade
  profiles/               # Voice profile management
    schema.py             # ProfileMetadata dataclass
    manager.py            # create/list/refresh/load profiles
```

## CLI Commands

```bash
mdt profile create <name> <voice.wav> [--backends applio,sovits,seedvc] [--epochs 30]
mdt profile list
mdt profile refresh <name> [--audio new_voice.wav]
mdt convert <profile> <song.wav> -o output.wav [--backend applio] [--sections sections.json]
mdt benchmark <profile> <song.wav> [-o benchmark_dir/]
```

## Key Experiment Scripts

| Script | Purpose |
|---|---|
| `benchmark_3bases.py` | 3-base A/B section-level benchmark on dtr.wav |
| `benchmark_checkpoints.py` | Sweep checkpoints for a single base |
| `benchmark_solo_author.py` | Multi-backend comparison on Solo Author |
| `sweep_rap_params.py` | Rap section protect/index_rate param sweep |
| `train_3_bases.py` | Train all 3 bases sequentially to 200ep |
| `train_remaining.py` | Resume TITAN + Snowie training |
| `convert_solo_author.py` | Convert Solo Author with best Applio model |

## Benchmark Results (3-Base A/B)

Ran on dtr.wav with 17-min voice data, section-level evaluation:

### Cross-base at comparable epochs

| Base | Ep | Mel->User | Rap->User | Pitch | Artifact |
|---|---|---|---|---|---|
| **TITAN 48k** | **25** | **0.855** | **0.798** | **0.794** | 0.173 |
| Singer32k | 200 | 0.853 | 0.796 | 0.763 | 0.170 |
| Snowie 48k | 20 | 0.828 | 0.789 | 0.757 | 0.160 |

### Singer32k Training Curve (full 200ep)

| Epoch | Mel->User | Rap->User | Pitch | Artifact |
|---|---|---|---|---|
| 25 | 0.837 | 0.789 | 0.736 | 0.166 |
| 50 | 0.843 | 0.791 | 0.787 | 0.168 |
| 100 | 0.835 | 0.790 | 0.772 | 0.167 |
| 150 | 0.851 | 0.784 | 0.761 | 0.165 |
| 200 | 0.853 | 0.796 | 0.763 | 0.170 |

### Rap Param Sweep (TITAN 25ep)

`protect` parameter has negligible effect on rap sections (0.841-0.842 speaker sim across all values).

## Current Output Files

All in `~/Downloads/voice_benchmark/3base_ab/`:
- `titan48k_25ep_mix.wav` — dtr.wav, TITAN 25ep voice (best quality)
- `singer32k_200ep_mix.wav` — dtr.wav, singer32k voice
- `solo_author_titan25ep_mix.wav` — Solo Author, TITAN voice
- `3base_comparison.json` — full benchmark data

## Critical Technical Notes

### FAISS Segfault on macOS
FAISS index causes SIGSEGV (rc=-11) on macOS Apple Silicon. **Must use `index_rate=0.0` and empty `file_index`** for all Applio inference. This is hardcoded in `mdt/vc/applio.py`.

### Applio Subprocess Isolation
All Applio operations must run via subprocess to avoid:
- `rvc.pth` module path conflicts (needs to be renamed during inference)
- `torchfcpe` import failure (fake module injected)
- FAISS segfaults
- DDP requires `MASTER_ADDR` + `MASTER_PORT` env vars

### Applio Training Success Signal
Applio `train.py` exits with `os._exit(2333333)` on success, not exit code 0.

### Training Checkpoint Storage
- `G_2333333.pth` / `D_2333333.pth` — training state (optimizer + model, ~850MB + ~450MB)
- `{name}_{epoch}e_{step}s.pth` — extracted inference weights (~57MB)
- With `save_only_latest=True`, G/D files are overwritten each save (saves disk)

All training data is in:
- `/Users/sarsa/claude/Applio/logs/yj_singer32k/` — Singer32k (200ep complete)
- `/Users/sarsa/claude/Applio/logs/yj_titan48k/` — TITAN 48k (training in progress)
- `/Users/sarsa/claude/Applio/logs/yj_snowie48k/` — Snowie 48k (queued)

### External Dependencies
- **Applio**: `/Users/sarsa/claude/Applio` — RVC training + inference engine
- **Pretrained models** in Applio's `rvc/models/pretraineds/`:
  - `SingerPreTrain_32k/` — f0G/f0D pair
  - `TITAN_48k/` — G/D pair
  - `SnowieV3.1_48k/` — G/D pair
- **contentvec** embedding model (auto-downloaded by Applio)

### Python Environment
- Python 3.14, venv at `.venv/`
- Key deps: torch, torchaudio, librosa, soundfile, pyworld, resemblyzer, demucs

## Remaining Work

### In Progress
- [ ] TITAN 48k training to 200ep (~4-5 hours remaining)
- [ ] Snowie 48k training to 200ep (starts after TITAN, ~8 hours)

### Next Steps (after training completes)
1. Re-run `benchmark_3bases.py` with full TITAN + Snowie training curves
2. Find optimal epoch for TITAN (currently 25ep is best among available)
3. Produce final mixes with TITAN best checkpoint for both songs
4. User records 8-15 min targeted audio per `RECORDING_GUIDE.md`
5. Retrain TITAN with combined 25+ min data
6. Full re-benchmark with augmented data

### Not Started
- YingMusic-SVC exploratory benchmark (low priority)
- Phrase-level chunked inference with per-section params on full songs
- SoVITS model integration for Solo Author benchmark (model not found at expected path)

## Manual Transfer Checklist

These files are too large for git and must be copied manually (e.g. `rsync`, AirDrop, external drive).

### Audio Files (~160MB)

| File | Size | Description |
|---|---|---|
| `~/Downloads/yj_voice.wav` | 90MB | User voice recording (17.8 min) |
| `~/Downloads/dtr.wav` | 41MB | SUNO song 1: "越界 Duck The Rope" (235s, 44.1kHz) |
| `~/Downloads/Solo Author.wav` | 29MB | SUNO song 2 (149.7s, 48kHz) |

Place in `~/Downloads/` on the new machine (paths referenced throughout scripts).

### Trained Models (~5GB)

Copy from old machine's `Applio/logs/` to new machine's `Applio/logs/`:

| Directory | Size | Status |
|---|---|---|
| `Applio/logs/yj_singer32k/` | ~2GB | 200ep complete, all checkpoints 5-200ep |
| `Applio/logs/yj_titan48k/` | ~2GB | Training in progress (~75ep, target 200ep) |
| `Applio/logs/yj_snowie48k/` | ~1.3GB | Queued, currently at 20ep |

Each directory contains:
- `G_2333333.pth` + `D_2333333.pth` — optimizer state for resume
- `yj_*_{epoch}e_{step}s.pth` — extracted inference weights (~57MB each)
- `filelist.txt`, `config.json` — training config
- `dataset/`, `sliced_audios/`, `extracted/`, `f0/`, `f0_voiced/` — preprocessed data

**Important:** If you only transfer the `*_{epoch}e_{step}s.pth` weight files (~57MB each), you can run inference but cannot resume training. To resume training, you need the full directory including `G_2333333.pth` + `D_2333333.pth` (~1.3GB pair).

### Pretrained Bases (~3GB)

Located in `Applio/rvc/models/pretraineds/`:

| Directory | Size | Source |
|---|---|---|
| `SingerPreTrain_32k/` | ~500MB | `f0G_SingerPreTrain.pth` + `f0D_SingerPreTrain.pth` |
| `TITAN_48k/` | ~1.3GB | `G-f048k-TITAN-Medium.pth` + `D-f048k-TITAN-Medium.pth` |
| `SnowieV3.1_48k/` | ~1.3GB | `G_SnowieV3.1_48k.pth` + `D_SnowieV3.1_48k.pth` |

Also need `Applio/rvc/models/predictors/rmvpe.pt` (~135MB) for F0 extraction.

### Benchmark Results (optional, ~600MB)

| Directory | Description |
|---|---|
| `~/Downloads/voice_benchmark/3base_ab/` | 3-base A/B comparison WAVs + JSON |
| `~/Downloads/voice_benchmark/rap_sweep/` | Rap param sweep results |
| `~/Downloads/voice_benchmark/` (root) | Earlier checkpoint benchmarks |

These can be regenerated by re-running the benchmark scripts.

### Applio Engine

The full Applio repo at `/Users/sarsa/claude/Applio/`. Either:
- Copy the entire directory (~several GB), or
- Fresh clone from `https://github.com/IAHispano/Applio.git` and re-install deps, then copy pretrained models and logs into it

### Quick rsync Commands

```bash
# From old machine to new (adjust NEW_HOST):
NEW="user@new-machine"

# Audio files
rsync -avP ~/Downloads/yj_voice.wav ~/Downloads/dtr.wav "$NEW:~/Downloads/"
rsync -avP ~/Downloads/Solo\ Author.wav "$NEW:~/Downloads/"

# Trained models (most important)
rsync -avP /Users/sarsa/claude/Applio/logs/yj_singer32k/ "$NEW:/path/to/Applio/logs/yj_singer32k/"
rsync -avP /Users/sarsa/claude/Applio/logs/yj_titan48k/ "$NEW:/path/to/Applio/logs/yj_titan48k/"
rsync -avP /Users/sarsa/claude/Applio/logs/yj_snowie48k/ "$NEW:/path/to/Applio/logs/yj_snowie48k/"

# Pretrained bases
rsync -avP /Users/sarsa/claude/Applio/rvc/models/pretraineds/ "$NEW:/path/to/Applio/rvc/models/pretraineds/"
rsync -avP /Users/sarsa/claude/Applio/rvc/models/predictors/rmvpe.pt "$NEW:/path/to/Applio/rvc/models/predictors/"

# Benchmark results (optional)
rsync -avP ~/Downloads/voice_benchmark/ "$NEW:~/Downloads/voice_benchmark/"
```

## Setup on New Machine

```bash
# 1. Clone repo
git clone https://github.com/baxia-lan/million-dollar-tuner.git
cd million-dollar-tuner
git checkout claude/vocal-tuning-replacement-AvJSj

# 2. Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install resemblyzer pyworld

# 3. Clone Applio (or copy from old machine)
cd ..
git clone https://github.com/IAHispano/Applio.git
cd Applio
pip install -r requirements.txt

# 4. Copy pretrained models + trained models + audio (see rsync commands above)

# 5. Verify setup
cd ../million-dollar-tuner
python -c "from mdt.vc.registry import list_backends; print(list_backends())"
# Should print: ['applio', 'sovits', 'seedvc']

# 6. Run benchmark (verifies everything works)
python benchmark_3bases.py

# 7. Resume training if needed (TITAN/Snowie not at 200ep)
python train_remaining.py
```
