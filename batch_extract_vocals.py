#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量从 input_video 文件夹中提取人声与伴奏
使用 MoviePy + Demucs + PyDub 实现
输出结构：
  separated/<视频名>/ 中间分离的WAV文件
  mp3_output/<视频名>/ 最终人声vocals.mp3与伴奏accompaniment.mp3
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


# ---------------- 基础功能函数 ----------------

def extract_audio_from_video(video_path, output_audio_path):
    """从视频中提取音频"""
    print(f"\n🎬 提取音频：{os.path.basename(video_path)} -> {output_audio_path}")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path, codec="pcm_s16le", fps=44100, verbose=False, logger=None)
    clip.close()


def separate_sources(audio_path, output_dir, model_name="htdemucs"):
    """使用 Demucs 分离音源"""
    print(f"🎤 使用 Demucs 模型进行音源分离（模型：{model_name}）...")
    os.makedirs(output_dir, exist_ok=True)

    model = pretrained.get_model(name=model_name)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device_str).eval()

    audio_file = AudioFile(audio_path)
    wav = audio_file.read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)

    if not torch.is_tensor(wav):
        wav = torch.tensor(wav, dtype=torch.float32)

    device = next(model.parameters()).device
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
    """合并为人声与伴奏"""
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


# ---------------- 主流程：批量处理 ----------------

def main():
    input_dir = "input_videos"
    separated_root = "separated"
    mp3_root = "mp3_output"
    temp_audio = "temp_audio.wav"

    if not os.path.exists(input_dir):
        print(f"❗ 未找到输入目录：{input_dir}")
        sys.exit(1)

    # 搜索所有 MP4 文件
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")]
    if not video_files:
        print(f"⚠️ 未在 {input_dir} 中找到任何 mp4 文件。")
        sys.exit(0)

    print(f"🔍 共检测到 {len(video_files)} 个视频文件待处理。")

    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        base_name = os.path.splitext(video_file)[0]
        print(f"\n==============================")
        print(f"🎞️ 正在处理：{base_name}")
        print(f"==============================")

        # 为该视频建立独立输出路径
        separated_dir = os.path.join(separated_root, base_name)
        output_dir = os.path.join(mp3_root, base_name)

        try:
            extract_audio_from_video(video_path, temp_audio)
            stems = separate_sources(temp_audio, separated_dir)
            merge_to_two_tracks(stems, output_dir)
            print(f"🎉 完成：{video_file} 的人声分离！")
        except Exception:
            print(f"❌ 处理 {video_file} 时出现错误：")
            traceback.print_exc()
        finally:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)

    print("\n✅ 全部视频处理完成！输出已保存至 mp3_output/ 与 separated/ 目录。")


if __name__ == "__main__":
    main()
