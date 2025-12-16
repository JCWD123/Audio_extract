#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从视频文件中提取人声和背景声，并输出为两个 mp3 文件
技术栈：MoviePy + Demucs + PyDub

兼容性增强：
- 兼容 Demucs 新旧 API（避免使用 model.device，使用 next(model.parameters()).device）
- 兼容 AudioFile.read 返回 numpy 或 torch.Tensor
- 支持 CPU / GPU 自动选择
- 使用 scipy.io.wavfile.write 输出 wav，再用 pydub 导出 mp3
"""

import os
import sys
import traceback

import numpy as np
import torch
from moviepy.editor import VideoFileClip
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import AudioFile
from pydub import AudioSegment
from scipy.io.wavfile import write as wav_write


def extract_audio_from_video(video_path, output_audio_path):
    print(f"🎬 提取音频中：{video_path} -> {output_audio_path}")
    clip = VideoFileClip(video_path)
    # 写出为无损 PCM WAV，以便 Demucs 使用
    clip.audio.write_audiofile(output_audio_path, codec="pcm_s16le", fps=44100, verbose=False, logger=None)
    clip.close()


def separate_sources(audio_path, output_dir, model_name="htdemucs"):
    """
    使用 Demucs 进行音源分离，保存 wav 文件到 output_dir
    """
    print(f"🎤 使用 Demucs 模型进行音源分离（模型：{model_name}）...")
    os.makedirs(output_dir, exist_ok=True)

    # 加载预训练模型
    model = pretrained.get_model(name=model_name)
    # 放到设备（GPU 如果可用）
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device_str)
    model.eval()

    # 读取音频（AudioFile 会返回 numpy 或 torch.Tensor，视版本而定）
    audio_file = AudioFile(audio_path)
    wav = audio_file.read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)

    # 如果返回的是 numpy，转为 torch.Tensor
    if not torch.is_tensor(wav):
        wav = torch.tensor(wav, dtype=torch.float32)

    # Ensure shape is (channels, samples)
    # Some versions might return (samples, channels) — try to detect and transpose if needed
    if wav.dim() == 2 and wav.shape[0] < wav.shape[1]:
        # could be (samples, channels); we expect (channels, samples)
        # Heuristic: if first dim is smaller than second and equals number of channels, it's fine.
        pass

    # 获取实际放置模型参数的 device（兼容 BagOfModels）
    try:
        device = next(model.parameters()).device
    except StopIteration:
        # fallback
        device = torch.device(device_str)

    wav = wav.to(device)

    # 保存引用统计以便后归一化（和 Demucs 示例类似）
    # wav shape expected: (channels, samples)
    ref = wav.mean(0)
    ref_mean = ref.mean()
    ref_std = ref.std() if ref.std() != 0 else 1.0
    wav_norm = (wav - ref_mean) / ref_std

    # apply_model expects shape (batch, channels, samples)
    batch = wav_norm[None]

    # 执行分离（传入 device 变量，不使用 model.device）
    sources = apply_model(model, batch, device=device)[0]  # 返回 shape (n_sources, channels, samples)

    # 反归一化
    sources = sources * ref_std + ref_mean

    # 保存每个 stem 为 wav 文件（int16）
    stems = getattr(model, "sources", None)
    if stems is None:
        # fallback stem names
        stems = [f"stem_{i}" for i in range(sources.shape[0])]

    saved_paths = []
    for idx, src in enumerate(stems):
        audio_tensor = sources[idx]  # shape (channels, samples)
        # 转为 cpu numpy
        audio_np = audio_tensor.detach().cpu().numpy()

        # If shape is (channels, samples) -> transpose to (samples, channels) for wav write
        if audio_np.ndim == 2:
            audio_out = audio_np.T
        elif audio_np.ndim == 1:
            # mono
            audio_out = audio_np[:, None]
        else:
            raise ValueError("Unexpected audio array shape: {}".format(audio_np.shape))

        # clip to -1..1 then convert to int16
        audio_out = np.clip(audio_out, -1.0, 1.0)
        int16 = (audio_out * 32767).astype(np.int16)

        out_wav = os.path.join(output_dir, f"{src}.wav")
        wav_write(out_wav, model.samplerate, int16)
        saved_paths.append(out_wav)
        print(f"💾 已导出 WAV: {out_wav}")

    return saved_paths


def convert_to_mp3(wav_paths, output_dir, bitrate="320k"):
    print("🎧 转换为 MP3 格式...")
    os.makedirs(output_dir, exist_ok=True)
    mp3_paths = []
    for wav_path in wav_paths:
        filename = os.path.basename(wav_path)
        mp3_name = filename.replace(".wav", ".mp3")
        mp3_path = os.path.join(output_dir, mp3_name)

        # 用 pydub 从 wav 转 mp3（需要系统 ffmpeg）
        sound = AudioSegment.from_wav(wav_path)
        sound.export(mp3_path, format="mp3", bitrate=bitrate)
        mp3_paths.append(mp3_path)
        print(f"✅ 输出 MP3: {mp3_path}")

    return mp3_paths


def main():
    video_path = "input_video.mp4"       # 你的输入视频（可改为绝对路径）
    audio_path = "temp_audio.wav"
    separated_dir = "separated"
    mp3_output_dir = "mp3_output"

    if not os.path.exists(video_path):
        print(f"❗ 未找到视频文件：{video_path}\n请把测试视频放到项目目录或修改 video_path 为正确路径。")
        sys.exit(1)

    try:
        extract_audio_from_video(video_path, audio_path)
        wav_files = separate_sources(audio_path, separated_dir, model_name="htdemucs")
        convert_to_mp3(wav_files, mp3_output_dir)
        print("\n🎉 处理完成！查看 mp3_output 目录下的人声与伴奏。")
    except Exception:
        print("\n❌ 出现异常，下面是详细错误信息：")
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
