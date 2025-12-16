# rvc_infer.py
import os
import subprocess
from pathlib import Path

def convert_voice(model_path, input_path, output_path, f0method="dio"):
    """
    用 RVC 模型转换音色
    model_path: 训练好的模型文件路径
    input_path: 待转换的音频路径
    output_path: 输出文件路径
    f0method: 基频提取方式
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"未找到输入音频: {input_path}")

    os.chdir("Retrieval-based-Voice-Conversion-WebUI")

    cmd = [
        "python", "infer.py",
        "-model_path", model_path,
        "-input", input_path,
        "-output", output_path,
        "-f0method", f0method
    ]

    print("🎶 开始音色转换...")
    subprocess.run(cmd)
    print(f"✅ 音色转换完成: {output_path}")

if __name__ == "__main__":
    model_path = "logs/my_voice_model/G_50000.pth"
    input_path = "../audio_extract/output/vocal.wav"
    output_path = "../rvc_demo/output/vocal_converted.wav"

    Path("../rvc_demo/output").mkdir(parents=True, exist_ok=True)
    convert_voice(model_path, input_path, output_path)
