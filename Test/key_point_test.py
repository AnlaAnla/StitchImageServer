# --- START OF FILE key_point_test.py ---

import cv2
import os
import time
from pathlib import Path
import re
from tqdm import tqdm

# 导入您提供的拼接器类和拼接顺序生成器
from fry_project_classes.stitch_img_key_point import ImageStitcherKeyPoint
from fry_project_classes.get_full_stitch_order import get_full_stitch_order


def natural_sort_key(s):
    """
    提供自然排序的键，例如 '2.jpg' 会排在 '10.jpg' 之前。
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


# --- 重构后的 stitch_img 函数 ---
def stitch_img(IMAGE_DIR, OUTPUT_DIR, NUM_COLS: int, NUM_ROWS: int,
               ESTIMATE_OVERLAP_HORIZONTAL_PIXELS: int, ESTIMATE_OVERLAP_VERTICAL_PIXELS: int,
               BLEND_TYPE: str, FEATURE_DETECTOR: str, DEBUG_MODE: bool,
               BLEND_RATIO: float, LIGHT_COMPENSATION: bool, LIGHT_COMPENSATION_WIDTH: int):
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("--- 图像拼接开始 ---")
    print(f"配置: {NUM_ROWS}行 x {NUM_COLS}列")
    print(f"图片目录: {IMAGE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"水平重叠预估: {ESTIMATE_OVERLAP_HORIZONTAL_PIXELS}px, 垂直重叠预估: {ESTIMATE_OVERLAP_VERTICAL_PIXELS}px")
    print(f"特征检测器: {FEATURE_DETECTOR}, 融合模式: {BLEND_TYPE}, 融合权重: {BLEND_RATIO}")
    print(f"光照补偿: {'启用' if LIGHT_COMPENSATION else '禁用'}, 补偿宽度: {LIGHT_COMPENSATION_WIDTH}px")

    # --- 1. 加载并排序所有图片 ---
    image_paths = sorted(list(IMAGE_DIR.glob("*.jpg")), key=natural_sort_key)

    if len(image_paths) != NUM_COLS * NUM_ROWS:
        print(f"错误: 找到 {len(image_paths)} 张图片, 但预期需要 {NUM_COLS * NUM_ROWS} 张。")
        return

    # 将所有图片读入内存，并用一个字典存储
    images_dict = {}
    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        if img is None:
            print(f"错误: 无法读取图片 {path}")
            return
        # 使用从 '1' 开始的字符串作为键
        images_dict[str(i + 1)] = img

    # --- 2. 获取拼接顺序 ---
    full_stitch_order_dict = get_full_stitch_order(NUM_ROWS, NUM_COLS)
    print(f"\n--- 获取到 {len(full_stitch_order_dict)} 步拼接指令 ---")

    # --- 3. 按照指令集执行拼接 ---
    final_image = None
    progress_bar = tqdm(full_stitch_order_dict.items(), desc="执行拼接")

    for step, (round_num, img1_name, img2_name, direction, result_name) in progress_bar:
        progress_bar.set_description(f"步骤 {step}: {img1_name} + {img2_name} -> {result_name}")

        img1 = images_dict[img1_name]
        img2 = images_dict[img2_name]

        # 根据方向选择重叠像素
        overlap_pixels = 0
        if direction == 'horizontal':
            overlap_pixels = ESTIMATE_OVERLAP_HORIZONTAL_PIXELS
        elif direction == 'vertical':
            overlap_pixels = ESTIMATE_OVERLAP_VERTICAL_PIXELS
        else:
            raise ValueError(f"未知的拼接方向: {direction}")

        # 每次都创建一个新的拼接器实例
        stitcher = ImageStitcherKeyPoint(
            estimate_overlap_pixels=overlap_pixels,
            stitch_type=direction,
            blend_type=BLEND_TYPE,
            feature_detector=FEATURE_DETECTOR,
            blend_ratio=BLEND_RATIO,
            # 同样添加光照补偿参数，供底层的融合模块使用
            light_uniformity_compensation_enabled=LIGHT_COMPENSATION,
            light_uniformity_compensation_width=LIGHT_COMPENSATION_WIDTH,
            debug=DEBUG_MODE,
            debug_dir=str(OUTPUT_DIR / f'debug_{result_name}')
        )

        # 执行拼接
        stitched_image = stitcher.stitch_main(img1, img2)

        # 将新生成的图片存入字典，用于下一步拼接
        images_dict[result_name] = stitched_image
        final_image = stitched_image

        if DEBUG_MODE:
            intermediate_path = OUTPUT_DIR / f"intermediate_{result_name}.jpg"
            cv2.imwrite(str(intermediate_path), stitched_image)

    # --- 4. 保存最终结果 ---
    if final_image is not None:
        final_output_path = OUTPUT_DIR / "final_stitched_image.jpg"
        cv2.imwrite(str(final_output_path), final_image)
        print("\n--- 所有拼接任务完成！---")
        print(f"最终的全景图已保存至: {final_output_path}")
    else:
        print("\n--- 拼接失败，没有生成最终图像 ---")


def main():
    """
    主执行函数
    """
    # --- 1. 配置参数 ---

    # 图片和输出目录设置
    IMAGE_DIR = Path(r"C:\Code\ML\Project\StitchImageServer\temp\input\front_0_1")

    # 拼图网格设置
    NUM_COLS = 4
    NUM_ROWS = 6

    # ！！！关键拼接参数，您可能需要根据实际图片进行调整！！！
    # 关键点匹配对这个参数不敏感，但它仍然用于界定初始搜索区域
    ESTIMATE_OVERLAP_HORIZONTAL_PIXELS = 405
    ESTIMATE_OVERLAP_VERTICAL_PIXELS = 440

    # --- 新增和修改的参数，与原始项目对齐 ---
    BLEND_RATIO = 0.5  # 融合权重，对 'half_importance_add_weight' 等模式有效
    LIGHT_COMPENSATION = True  # 是否开启光照补偿
    LIGHT_COMPENSATION_WIDTH = 15  # 光照补偿的计算宽度 (请根据原始项目调整)

    # 可测试的融合模式列表
    blend_type_list = ["half_importance_add_weight"]

    # 可测试的特征检测器列表
    feature_detector_list = ["sift", "orb", "akaze", "brisk", "combine"]

    # 是否开启调试模式（会生成大量中间过程图片，用于分析问题）
    DEBUG_MODE = True

    for feature_detector in feature_detector_list:
        for blend_type in blend_type_list:
            base_dir_path = r"C:\Code\ML\Project\StitchImageServer\temp\output"
            # 创建更详细的输出文件夹名
            img_dir_name = f"keypoint_{feature_detector}_{blend_type}"
            OUTPUT_DIR = Path(os.path.join(base_dir_path, img_dir_name))

            print("\n" + "=" * 50)
            print(f"开始测试配置: 检测器={feature_detector}, 融合模式={blend_type}")
            print("=" * 50)

            one_config_time = time.time()
            stitch_img(
                IMAGE_DIR=IMAGE_DIR,
                OUTPUT_DIR=OUTPUT_DIR,
                NUM_COLS=NUM_COLS,
                NUM_ROWS=NUM_ROWS,
                ESTIMATE_OVERLAP_HORIZONTAL_PIXELS=ESTIMATE_OVERLAP_HORIZONTAL_PIXELS,
                ESTIMATE_OVERLAP_VERTICAL_PIXELS=ESTIMATE_OVERLAP_VERTICAL_PIXELS,
                BLEND_TYPE=blend_type,
                FEATURE_DETECTOR=feature_detector,
                DEBUG_MODE=DEBUG_MODE,
                BLEND_RATIO=BLEND_RATIO,
                LIGHT_COMPENSATION=LIGHT_COMPENSATION,
                LIGHT_COMPENSATION_WIDTH=LIGHT_COMPENSATION_WIDTH
            )
            print(f"配置 {img_dir_name} 完成, 耗时: {time.time() - one_config_time:.2f} 秒")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n总耗时: {end_time - start_time:.2f} 秒")