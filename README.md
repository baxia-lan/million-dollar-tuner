# Million Dollar Tuner

用你自己的声音替换 SUNO AI 歌曲的人声。

## 安装 (macOS)

```bash
brew install rubberband fluid-synth ffmpeg
git clone <repo-url> && cd million-dollar-tuner
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install so-vits-svc-fork pyworld
```

## 用法（两步）

### 第一步：训练你的声音模型（只需要做一次）

```bash
mdt train-voice 你的录音.wav --epochs 100
```

用你的声音录音训练一个神经网络模型。录音越长效果越好（建议 1 分钟以上）。
训练大约需要 10-20 分钟，完成后模型保存在 `voice_model/` 目录。

### 第二步：替换人声

```bash
mdt tune 你的录音.wav suno歌曲.wav -o 成品.wav --voice-model voice_model/
```

程序会：分离 SUNO 歌曲的人声和伴奏 → 用你的模型转换人声 → 混音输出。

### 不训练直接用（效果差）

```bash
mdt tune 你的录音.wav suno歌曲.wav -o 成品.wav
```

不带 `--voice-model` 会用 WORLD vocoder 做基础频谱转换，效果远不如训练模型。

## 其他命令

```bash
mdt separate 歌曲.wav -o ./stems/     # 分离音轨
mdt analyze 歌曲.wav                   # 分析调性和 BPM
mdt autotune 录音.wav -o 调音后.wav     # 简单自动调音
```

## 支持格式

WAV / MP3 / FLAC。建议用 WAV。

## 录音建议

- 1 分钟以上，安静环境，WAV 格式
- 说话或唱歌都行
- 不需要唱同一首歌
