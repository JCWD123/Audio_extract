#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 separated/ 目录下批量收集所有 vocals.wav 文件
并移动到 datasets/my_voice/ 目录下，命名为 vocal_<子目录名>.wav
"""

import os
import shutil

def main():
    separated_dir = "separated"
    target_dir = os.path.join("datasets", "my_voice")

    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    if not os.path.exists(separated_dir):
        print(f"❗ 未找到目录：{separated_dir}")
        return

    # 统计数量
    count = 0
    skipped = 0

    # 遍历 separated 下的所有子目录
    for subdir in os.listdir(separated_dir):
        subdir_path = os.path.join(separated_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        vocals_path = os.path.join(subdir_path, "vocals.wav")
        if os.path.exists(vocals_path):
            new_name = f"vocal_{subdir}.wav"
            dest_path = os.path.join(target_dir, new_name)

            # 如果目标文件已存在，自动重命名避免覆盖
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(new_name)
                i = 1
                while os.path.exists(os.path.join(target_dir, f"{base}_{i}{ext}")):
                    i += 1
                dest_path = os.path.join(target_dir, f"{base}_{i}{ext}")

            shutil.move(vocals_path, dest_path)
            count += 1
            print(f"✅ 已移动：{vocals_path} → {dest_path}")
        else:
            skipped += 1
            print(f"⚠️ 跳过：{subdir} 中未找到 vocals.wav")

    print(f"\n🎉 完成！共移动 {count} 个文件，跳过 {skipped} 个目录。")
    print(f"📂 输出目录：{target_dir}")


if __name__ == "__main__":
    main()
