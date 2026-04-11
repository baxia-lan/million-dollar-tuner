# Handoff: Voice Conversion

## 当前状态

Voice conversion 部分**没有做好**。信号处理方案（cepstral envelope、LPC、WORLD vocoder、频率 warping）全部在真实音频上失败——指标变了但听起来还是原声。

已经集成了 `so-vits-svc-fork`（神经网络方案），但还未在真实音频上测试过。

## 项目结构

```
mdt/
├── cli.py                  # CLI 入口，所有命令定义
├── config.py               # 全局配置
├── audio/
│   ├── io.py               # 音频读写（WAV/MP3/FLAC）
│   ├── effects.py          # Pedalboard 效果器链
│   ├── convert.py          # 格式转换/RMS匹配
│   └── mastering.py        # 母带处理（重采样/饱和/dither）
├── separation/
│   ├── separator.py        # Demucs 分离（含 spectral fallback + cache）
│   └── models.py           # StemResult 数据类
├── analysis/
│   ├── pitch.py            # PYIN 音高检测
│   ├── rhythm.py           # BPM/节拍/onset 检测
│   ├── key.py              # 调性检测（Krumhansl-Schmuckler）
│   └── features.py         # chroma/MFCC 特征
├── tuning/
│   ├── pipeline.py         # 主流水线（分离→转换→混音）
│   ├── voice_convert.py    # ★ 核心：voice conversion（需要重做）
│   ├── pitch_correct.py    # PSOLA 音高校正（autotune 命令用）
│   ├── time_align.py       # DTW 时间对齐（目前未使用）
│   └── mixer.py            # 人声+伴奏混音
├── resynthesis/            # MIDI转录/合成（实验性）
└── evaluation.py           # 质量评估指标
```

## 已完成且可用的功能

- `mdt separate` — Demucs 音轨分离 ✅（用户 Mac 上 Demucs 跑通了）
- `mdt analyze` — 调性/BPM 分析 ✅
- `mdt autotune` — 音阶校正 ✅
- stems cache 机制 ✅
- mastering 后处理 ✅
- 音频 I/O（WAV/MP3/FLAC）✅

## 需要完成的：Voice Conversion

### 当前实现（`mdt/tuning/voice_convert.py`）

两种模式：
1. **有训练模型时**：调用 `so-vits-svc-fork` 的 `Svc.infer_silence()` — **还未测试**
2. **无模型时**：WORLD vocoder 频率 warping — **不 work，听起来还是原声**

### 推荐方向

用 `so-vits-svc-fork` 做真正的 voice conversion：

```bash
# 训练（用户的 258 秒录音）
mdt train-voice ~/Downloads/recording.wav --epochs 100

# 推理
mdt tune recording.wav suno_song.wav -o result.wav --voice-model voice_model/
```

训练流程已经写好（`voice_convert.py` 的 `train_voice_model()`），调用 svc CLI：
1. `svc pre-split` — 切分长录音
2. `svc pre-resample` — 重采样到 44100Hz
3. `svc pre-config` — 生成配置
4. `svc pre-hubert` — 提取 HuBERT 特征
5. `svc train` — 训练

推理用 `so_vits_svc_fork.inference.core.Svc` 类。

### 需要注意的问题

1. **训练流程未测试** — `train_voice_model()` 的 svc CLI 调用可能有路径问题
2. **推理 API** — `Svc.__init__` 需要 `net_g_path` 和 `config_path`，需要正确找到训练产出的文件
3. **Demucs 在用户 Mac 上可用** — torchaudio patch 生效了，不需要 torchcodec
4. **Python 3.14** — 用户用的是 Python 3.14，注意兼容性

### 用户环境

- macOS, Python 3.14
- 已安装：torch, torchaudio, demucs, pyworld, so-vits-svc-fork, librosa, pedalboard
- Demucs htdemucs_ft 模型已下载并可用
- 用户有 258.8 秒的声音录音

### 关键文件

- `mdt/tuning/voice_convert.py` — voice conversion 核心，需要让 so-vits-svc 路径跑通
- `mdt/tuning/pipeline.py` — 主流水线，`voice_model_dir` 参数传递
- `mdt/cli.py` — `train-voice` 和 `tune --voice-model` 命令

### 用户的核心需求

> 给你一个 SUNO 歌曲 + 我的声音录音，输出一首完整的歌，SUNO 的音调节奏不变，只把人声音色换成我的。
