# Million Dollar Tuner (MDT)

**把你跑调的歌声变成专业级别的完美人声替换工具。**

Million Dollar Tuner 是一个命令行工具，让你可以用自己的声音替换 SUNO AI 生成歌曲中的人声。它会自动调音、对齐节奏，并将你的声音无缝混合到原始伴奏中。

## Features / 功能

### 1. Vocal Replacement（人声替换）— `mdt tune`
- 输入：你的唱歌录音 + SUNO 歌曲
- 自动将 SUNO 歌曲分离为人声和伴奏
- 分析参考人声的音高和节奏
- 自动修正你的音准（Auto-Tune）
- 自动对齐你的节奏（DTW时间拉伸）
- 混音输出完整歌曲
- 输出：你的声音 + SUNO 伴奏 = 完美歌曲

### 2. Stem Separation（音轨分离）— `mdt separate`
- 使用 Meta 的 Demucs 模型分离音轨
- 提取：人声、鼓、贝斯、其他乐器
- 支持 4 源和 6 源模型

### 3. Track Analysis（曲目分析）— `mdt analyze`
- 检测调性（Key Detection）
- 检测速度（BPM）
- 音高范围分析
- 节拍和起音检测

### 4. Auto-Tune（自动调音）— `mdt autotune`
- 简单的音阶校正（不需要参考歌曲）
- 支持自动检测调性
- 可调节修正强度（从自然到 T-Pain 效果）

### 5. Resynthesis（音乐重合成）— `mdt resynth` [实验性]
- 将每个音轨转录为 MIDI
- 使用 SoundFont 合成器重新生成音频
- 鼓和贝斯效果较好，和弦为近似值

## Installation / 安装

### System Dependencies / 系统依赖

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install rubberband-cli fluidsynth libfluidsynth3 ffmpeg

# macOS
brew install rubberband fluid-synth ffmpeg
```

### Python Package / Python 包

```bash
# 从源码安装
pip install -e .

# 或者安装依赖后直接使用
pip install -r requirements.txt
```

### GPU Support / GPU 支持

Demucs 在有 NVIDIA GPU 时会自动使用 CUDA 加速。
如果没有 GPU，会自动回退到 CPU（速度较慢但完全可用）。

```bash
# 如果需要 GPU 加速，确保安装了 CUDA 版 PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage / 使用方法

### 核心功能：人声替换

```bash
# 基本用法：用你的声音替换 SUNO 歌曲的人声
mdt tune my_voice.wav suno_song.mp3 -o result.wav

# 调节参数
mdt tune my_voice.wav suno_song.mp3 -o result.wav \
    --strength 0.9 \          # 音准修正强度 (0.0-1.0)
    --max-shift 3 \           # 最大音高偏移（半音）
    --reverb-room 0.4 \       # 混响大小
    --reverb-wet 0.2 \        # 混响湿度
    --vocal-gain 2.0          # 人声增益 (dB)

# 不加效果器（干声）
mdt tune my_voice.wav suno_song.mp3 -o result.wav --no-effects

# 指定使用 CPU
mdt tune my_voice.wav suno_song.mp3 -o result.wav --device cpu
```

### 音轨分离

```bash
# 分离歌曲为 4 个音轨
mdt separate song.mp3 -o ./stems/

# 使用 6 源模型（额外分离吉他和钢琴）
mdt separate song.mp3 -o ./stems/ --model htdemucs_6s
```

### 曲目分析

```bash
# 基本分析
mdt analyze song.mp3

# 详细分析（包含音高信息）
mdt analyze song.mp3 --detailed
```

### 自动调音（不需要参考歌曲）

```bash
# 自动检测调性并调音
mdt autotune my_voice.wav -o tuned.wav

# 指定调性
mdt autotune my_voice.wav -o tuned.wav --key "C major"

# T-Pain 效果（强修正）
mdt autotune my_voice.wav -o tuned.wav --strength 1.0
```

### 重合成（实验性）

```bash
# 转录为 MIDI 并重合成
mdt resynth song.mp3 -o ./resynth/ --soundfont path/to/soundfont.sf2

# 只导出 MIDI（不合成音频）
mdt resynth song.mp3 -o ./resynth/ --midi-only
```

## How It Works / 工作原理

### 人声替换流水线 (5 步)

```
SUNO 歌曲 ──► [Demucs 分离] ──► 人声 + 伴奏
                                   │
你的录音 ──────────────────────────┤
                                   ▼
                        [PYIN 音高检测] ──► 参考音高曲线
                                   │
                                   ▼
                        [PSOLA 音高校正] ──► 修正后的人声
                                   │
                                   ▼
                        [DTW 时间对齐] ──► 对齐后的人声
                                   │
                                   ▼
                        [混音 + 效果器] ──► 最终歌曲 🎵
```

### Key Technologies / 核心技术

| 组件 | 技术 | 说明 |
|------|------|------|
| 音轨分离 | Demucs (Meta) | 最先进的音源分离模型 |
| 音高检测 | PYIN (librosa) | 概率 YIN 算法 |
| 音高校正 | PSOLA | 时域同步叠加，保持音色 |
| 时间对齐 | DTW + pyrubberband | 动态时间规整 + 高质量时间拉伸 |
| 音频效果 | Pedalboard (Spotify) | 专业级效果器 |
| MIDI 转录 | librosa + pretty_midi | 音频到 MIDI 转换 |
| 合成引擎 | FluidSynth | SoundFont 合成器 |

## Project Structure / 项目结构

```
million-dollar-tuner/
├── mdt/
│   ├── cli.py              # CLI 命令入口
│   ├── config.py           # 全局配置
│   ├── audio/
│   │   ├── io.py           # 音频读写
│   │   ├── effects.py      # 效果器链
│   │   └── convert.py      # 格式转换
│   ├── separation/
│   │   ├── separator.py    # Demucs 封装
│   │   └── models.py       # 数据模型
│   ├── analysis/
│   │   ├── pitch.py        # 音高检测
│   │   ├── rhythm.py       # 节奏分析
│   │   ├── key.py          # 调性检测
│   │   └── features.py     # 特征提取
│   ├── tuning/
│   │   ├── pipeline.py     # 人声替换流水线
│   │   ├── pitch_correct.py # 自动调音引擎
│   │   ├── time_align.py   # DTW 时间对齐
│   │   └── mixer.py        # 混音器
│   └── resynthesis/
│       ├── transcriber.py  # MIDI 转录
│       ├── synth_engine.py # 合成引擎
│       └── controls.py     # 合成参数
├── soundfonts/             # SoundFont 文件
├── tests/                  # 测试
├── pyproject.toml          # 项目配置
└── README.md
```

## Tips / 使用建议

1. **录音质量**：在安静环境中录音，使用耳机监听 SUNO 歌曲跟唱
2. **录音长度**：尽量让你的录音长度与 SUNO 歌曲相近（差距 <30%）
3. **音准修正强度**：0.8 听起来自然，1.0 是机器人效果，0.5 保留更多原始音准
4. **GPU**：有 NVIDIA GPU 的话，音轨分离会快 5-10 倍
5. **SoundFont**：重合成功能需要 .sf2 文件，推荐 GeneralUser GS

## License

MIT
