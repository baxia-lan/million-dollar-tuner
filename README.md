# Million Dollar Tuner

用你自己的声音替换 SUNO AI 歌曲的人声，自动调音调节奏。

## 安装 (macOS)

```bash
# 1. 装系统依赖
brew install rubberband fluid-synth ffmpeg

# 2. 克隆项目
git clone <repo-url> && cd million-dollar-tuner

# 3. 装 Python 包（会自动安装所有依赖，装完就有 mdt 命令）
pip install -e .
```

装完后终端输入 `mdt --version` 验证。

## 用法

### 核心：替换人声

```bash
mdt tune 我的录音.wav suno歌曲.mp3 -o 成品.wav
```

程序会自动：分离伴奏 -> 分析音高节奏 -> 修正你的音准 -> 对齐节奏 -> 混音输出。
质量不达标会自动调参重试，最多 4 轮。

### 其他命令

```bash
mdt separate 歌曲.mp3 -o ./stems/          # 分离音轨（人声/鼓/贝斯/其他）
mdt analyze 歌曲.mp3                        # 分析调性和 BPM
mdt autotune 录音.wav -o 调音后.wav          # 简单自动调音（不需要参考歌曲）
mdt resynth 歌曲.mp3 -o ./resynth/ --midi-only  # 转 MIDI（实验性）
```

### 常用参数

```bash
mdt tune 录音.wav 歌曲.mp3 -o 成品.wav \
    --strength 0.9 \      # 音准修正力度 0.0-1.0（默认 0.8）
    --max-shift 3 \        # 最大偏移半音数（默认 4）
    --no-effects           # 不加混响压缩等效果
```

## 支持格式

输入输出都支持 WAV / MP3 / FLAC。建议全程用 WAV（无损）。

## 工作原理

```
SUNO歌曲 -> [Demucs分离] -> 伴奏
你的录音 -> [PYIN检测] -> [PSOLA调音] -> [DTW对齐] -> [混音] -> 成品
```

## 录音建议

- 戴耳机听着 SUNO 歌曲跟唱，录音长度尽量和原曲接近
- 安静环境，WAV 格式录制
- 不用唱得很准，程序会修正
