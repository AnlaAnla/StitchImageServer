import cv2
import glob
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def resize_image(img_path: str, target_size: tuple):
    try:
        # 读取图片
        img = cv2.imread(img_path)

        # 检查图片是否成功读取
        if img is None:
            print(f"警告: 无法读取图片 {img_path}，已跳过。")
            return None

        # 缩放图片
        resized_img = cv2.resize(img, target_size)

        # 保存缩放后的图片
        cv2.imwrite(str(img_path), resized_img)

        return str(img_path)

    except Exception as e:
        # 捕获其他潜在错误
        print(f"处理图片 {img_path} 时发生错误: {e}")
        return None


def main():
    INPUT_PATTERN = r'C:\Code\ML\Project\StitchImageServer\temp\Input\*\*.jpg'

    # 目标尺寸 (宽度, 高度)
    TARGET_SIZE = (1024, 1024)

    # 使用的进程数，None表示自动使用所有可用的CPU核心
    # 你也可以手动设置为一个整数，例如 4
    MAX_WORKERS = 4

    # 获取所有图片路径
    img_paths = glob.glob(INPUT_PATTERN)

    if not img_paths:
        print(f"在模式 '{INPUT_PATTERN}' 下未找到任何图片。")
        return

    print(f"找到 {len(img_paths)} 张图片待处理。")

    # --- 3. 使用进程池并行处理 ---
    start_time = time.time()
    processed_count = 0

    # 创建一个进程池
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(resize_image, path, TARGET_SIZE): path for path in img_paths}

        # 使用 tqdm 创建进度条，并处理已完成的任务
        for future in tqdm(as_completed(futures), total=len(img_paths), desc="批量缩放图片"):
            result = future.result()
            if result:
                # 如果任务成功返回了输出路径，则计数加一
                processed_count += 1

    end_time = time.time()

    # --- 4. 打印总结 ---
    print("\n--- 任务完成 ---")
    print(f"成功处理了 {processed_count} / {len(img_paths)} 张图片。")
    print(f"总耗时: {end_time - start_time:.2f} 秒。")


# 这个保护措施对于多进程编程至关重要
if __name__ == '__main__':
    main()
