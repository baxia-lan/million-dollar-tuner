# Million Dollar Tuner

用你自己的声音替换 SUNO AI 歌曲的人声，自动调音调节奏。

## 安装 (macOS)

```bash
# 1. 装系统依赖
brew install rubberband fluid-synth ffmpeg

# 2. 克隆项目
git clone <repo-url> && cd million-dollar-tuner

# 3. 创建虚拟环境并安装
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

装完后输入 `mdt --version` 验证。

以后每次用之前先激活环境：`source .venv/bin/activate`

## 用法

### 核心：用你的声音替换 SUNO 人声

```bash
mdt tune 我的声音.wav suno歌曲.wav -o 成品.wav
```

原理：提取你的声音音色，替换掉 SUNO 歌曲里的人声音色。
SUNO 原本的音调、节奏、歌词一切不变，只是听起来像你在唱。

你的录音不需要唱同一首歌，随便说话或唱歌都行，只要能采集到你的音色。

### 常用参数

```bash
mdt tune 声音.wav 歌曲.wav -o 成品.wav \
    --blend 0.9 \          # 音色替换程度 0.0-1.0（默认 0.8）
    --no-effects           # 不加混响压缩等效果
```

### 其他命令

```bash
mdt separate 歌曲.wav -o ./stems/          # 分离音轨（人声/鼓/贝斯/其他）
mdt analyze 歌曲.wav                        # 分析调性和 BPM
mdt autotune 录音.wav -o 调音后.wav          # 简单自动调音（不需要参考歌曲）
mdt resynth 歌曲.wav -o ./resynth/ --midi-only  # 转 MIDI（实验性）
```

## 支持格式

输入输出都支持 WAV / MP3 / FLAC。建议全程用 WAV（无损）。

## 工作原理

```
SUNO歌曲 -> [Demucs分离] -> 人声 + 伴奏
                              |
你的声音 -> [提取音色] ------> [音色替换] -> 你的音色 + SUNO 原调
                                              |
                              伴奏 ---------> [混音] -> 成品
```

核心技术：频谱包络转移（Spectral Envelope Transfer）
- 从你的声音提取声道特征（共振峰/音色）
- 保留 SUNO 人声的激励信号（音高/节奏/谐波细节）
- 用你的音色包络替换 SUNO 的音色包络

## 录音建议

- 随便说话或唱歌都行，10 秒以上就够
- 安静环境，WAV 格式
- 不需要唱同一首歌，不需要唱准
