#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从视频文件中提取人声和背景声，并输出为两个 mp3 文件
技术栈：MoviePy + Demucs + PyDub

功能：
- 自动提取视频音频
- 使用 Demucs 分离音源
- 自动合并伴奏 (drums + bass + other)
- 输出两条轨道：人声 vocals.mp3 + 伴奏 accompaniment.mp3
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
    clip.audio.write_audiofile(output_audio_path, codec="pcm_s16le", fps=44100, verbose=False, logger=None)
    clip.close()


def separate_sources(audio_path, output_dir, model_name="htdemucs"):
    print(f"🎤 使用 Demucs 模型进行音源分离（模型：{model_name}）...")
    os.makedirs(output_dir, exist_ok=True)

    model = pretrained.get_model(name=model_name)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device_str).eval()

    audio_file = AudioFile(audio_path)
    wav = audio_file.read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)

    if not torch.is_tensor(wav):
        wav = torch.tensor(wav, dtype=torch.float32)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device(device_str)

    wav = wav.to(device)
    ref = wav.mean(0)
    ref_mean = ref.mean()
    ref_std = ref.std() if ref.std() != 0 else 1.0
    wav_norm = (wav - ref_mean) / ref_std

    batch = wav_norm[None]
    sources = apply_model(model, batch, device=device)[0]
    sources = sources * ref_std + ref_mean

    stems = getattr(model, "sources", None)
    if stems is None:
        stems = [f"stem_{i}" for i in range(sources.shape[0])]

    saved = {}
    for idx, src in enumerate(stems):
        audio_np = sources[idx].detach().cpu().numpy().T
        audio_out = np.clip(audio_np, -1.0, 1.0)
        int16 = (audio_out * 32767).astype(np.int16)
        out_wav = os.path.join(output_dir, f"{src}.wav")
        wav_write(out_wav, model.samplerate, int16)
        saved[src] = out_wav
        print(f"💾 已导出 WAV: {out_wav}")

    return saved


def merge_to_two_tracks(stems_dict, output_dir):
    print(f"🎚️ 合并为人声与伴奏...")
    os.makedirs(output_dir, exist_ok=True)

    vocals = AudioSegment.from_wav(stems_dict["vocals"])
    accompaniment = (
        AudioSegment.from_wav(stems_dict["drums"])
        .overlay(AudioSegment.from_wav(stems_dict["bass"]))
        .overlay(AudioSegment.from_wav(stems_dict["other"]))
    )

    vocals.export(os.path.join(output_dir, "vocals.mp3"), format="mp3", bitrate="320k")
    accompaniment.export(os.path.join(output_dir, "accompaniment.mp3"), format="mp3", bitrate="320k")

    print(f"✅ 输出人声: {os.path.join(output_dir, 'vocals.mp3')}")
    print(f"✅ 输出伴奏: {os.path.join(output_dir, 'accompaniment.mp3')}")


def main():
    video_path = "input_video2.mp4"
    temp_audio = "temp_audio.wav"
    separated_dir = "separated"
    output_dir = "mp3_output"

    if not os.path.exists(video_path):
        print(f"❗ 未找到视频文件：{video_path}")
        sys.exit(1)

    try:
        extract_audio_from_video(video_path, temp_audio)
        stems = separate_sources(temp_audio, separated_dir)
        merge_to_two_tracks(stems, output_dir)
        print("\n🎉 处理完成！已输出人声与伴奏两条音轨。")
    except Exception:
        print("\n❌ 出现异常，详细错误如下：")
        traceback.print_exc()


if __name__ == "__main__":
    main()
