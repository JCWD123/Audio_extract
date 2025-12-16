# rvc_train.py
import os
import subprocess

def train_rvc_model(data_dir, exp_name="my_voice", sr=48000, f0method="dio"):
    """
    使用 RVC 的 CLI 模式训练音色模型
    data_dir: 包含样本人声音频的目录
    exp_name: 训练输出目录名称
    sr: 采样率
    f0method: 基频提取方法，可选 dio / harvest
    """

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据路径不存在: {data_dir}")

    # 克隆 RVC 主体代码（若尚未存在）
    if not os.path.exists("Retrieval-based-Voice-Conversion-WebUI"):
        subprocess.run(["git", "clone", "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"])

    os.chdir("Retrieval-based-Voice-Conversion-WebUI")

    cmd = [
        "python", "train.py",
        "-sr", str(sr),
        "-f0method", f0method,
        "-data_dir", data_dir,
        "-exp_name", exp_name,
        "-gpus", "1"
    ]
    print("🚀 开始训练 RVC 模型...")
    subprocess.run(cmd)
    print("✅ 模型训练完成！输出路径:", f"logs/{exp_name}")

if __name__ == "__main__":
    # 假设语音样本存放在 project_root/datasets/my_voice
    train_rvc_model(data_dir="../datasets/my_voice", exp_name="my_voice_model")
