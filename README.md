# Audio Extract - 音频提取工具

一个基于深度学习的音频人声分离工具，可以从视频文件中提取人声和伴奏，支持批量处理。

## 🎯 功能特性

- **视频音频提取**：从 MP4 视频文件中提取音频
- **人声分离**：使用 Demucs 深度学习模型分离人声和伴奏
- **批量处理**：支持批量处理多个视频文件
- **多格式输出**：支持 WAV 和 MP3 格式输出
- **自动设备选择**：自动检测并使用 GPU（如果可用）加速处理

## 📋 技术栈

- **Python 3.x**
- **MoviePy** - 视频处理和音频提取
- **Demucs** - 基于深度学习的音源分离
- **PyDub** - 音频格式转换和处理
- **PyTorch** - 深度学习框架
- **NumPy / SciPy** - 数值计算和音频处理

## 🚀 快速开始

### 环境要求

- Python 3.7+
- CUDA（可选，用于 GPU 加速）
- FFmpeg（用于音频格式转换）

### 安装依赖

```bash
pip install moviepy demucs pydub torch numpy scipy
```

### 使用方法

#### 1. 单个视频处理

使用 `extract_vocals.py` 处理单个视频文件：

```bash
python extract_vocals.py
```

**注意**：需要修改脚本中的 `video_path` 变量为你的视频文件路径。

**输出**：
- `separated/` - 分离的 WAV 文件（vocals.wav, drums.wav, bass.wav, other.wav）
- `mp3_output/` - 最终输出的人声和伴奏 MP3 文件

#### 2. 批量处理

使用 `batch_extract_vocals.py` 批量处理 `input_videos/` 目录下的所有视频：

```bash
python batch_extract_vocals.py
```

**输出结构**：
```
separated/
  ├── 视频名1/
  │   ├── vocals.wav
  │   ├── drums.wav
  │   ├── bass.wav
  │   └── other.wav
  └── 视频名2/
      └── ...

mp3_output/
  ├── 视频名1/
  │   ├── vocals.mp3
  │   └── accompaniment.mp3
  └── 视频名2/
      └── ...
```

#### 3. 收集人声文件

使用 `collect_vocals.py` 将所有人声文件收集到训练数据集目录：

```bash
python collect_vocals.py
```

**功能**：从 `separated/` 目录下收集所有 `vocals.wav` 文件，移动到 `datasets/my_voice/` 目录，并重命名为 `vocal_<视频名>.wav`

#### 4. 使用 Spleeter 4-stems 分离

使用 `extract_vocals_spleeter-4-stems.py` 进行 4 音轨分离：

```bash
python extract_vocals_spleeter-4-stems.py
```

## 📁 项目结构

```
audio_extract/
├── extract_vocals.py              # 单个视频处理脚本
├── batch_extract_vocals.py        # 批量处理脚本
├── extract_vocals_spleeter-4-stems.py  # Spleeter 4-stems 分离脚本
├── collect_vocals.py              # 人声文件收集脚本
├── main.py                        # 示例文件
├── input_videos/                  # 输入视频目录（需自行创建）
├── separated/                     # 分离的 WAV 文件输出目录
├── mp3_output/                    # MP3 格式输出目录
├── datasets/                      # 数据集目录
│   └── my_voice/                  # 收集的人声文件
└── rvc_demo/                      # RVC 相关演示代码
```

## ⚙️ 配置说明

### 修改输入文件路径

在各个脚本中修改以下变量：

- `extract_vocals.py`: 修改 `video_path` 变量
- `batch_extract_vocals.py`: 修改 `input_dir` 变量（默认为 `input_videos`）

### 选择模型

默认使用 `htdemucs` 模型，可以在 `separate_sources()` 函数中修改 `model_name` 参数：

```python
stems = separate_sources(audio_path, separated_dir, model_name="htdemucs")
```

可用的 Demucs 模型：
- `htdemucs` - 高质量分离（推荐）
- `htdemucs_ft` - 微调版本
- `mdx_extra` - 额外模型

### 输出格式

- **WAV 格式**：无损音频，保存在 `separated/` 目录
- **MP3 格式**：压缩音频，320kbps 比特率，保存在 `mp3_output/` 目录

## 🔧 常见问题

### 1. GPU 加速

脚本会自动检测 CUDA 是否可用。如果安装了 CUDA 和 PyTorch GPU 版本，处理速度会显著提升。

### 2. FFmpeg 未找到

如果遇到 FFmpeg 相关错误，请确保已安装 FFmpeg：

- **Windows**: 下载并添加到 PATH
- **Linux**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`

### 3. 内存不足

处理大文件时可能出现内存不足。建议：
- 使用 GPU 加速
- 分批处理视频文件
- 降低音频采样率（修改代码中的 `fps` 参数）

## 📝 注意事项

- 音频和视频文件（`.mp3`, `.mp4`, `.wav`）已被 `.gitignore` 忽略，不会提交到仓库
- 首次运行时会自动下载 Demucs 预训练模型（约几百 MB）
- 处理时间取决于视频长度和硬件配置

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

