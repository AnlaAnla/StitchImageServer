import cv2
import os
import time
from pathlib import Path
import re
from tqdm import tqdm

# 导入您提供的拼接器类
from fry_project_classes.stitch_img_template_match import ImageStitcherTemplateMatch


def natural_sort_key(s):
    """
    提供自然排序的键，例如 '2.jpg' 会排在 '10.jpg' 之前。
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def stitch_img(IMAGE_DIR, OUTPUT_DIR, NUM_COLS: int, NUM_ROWS: int,
               ESTIMATE_OVERLAP_HORIZONTAL_PIXELS: int, ESTIMATE_OVERLAP_VERTICAL_PIXELS: int,
               BLEND_TYPE: str, LIGHT_COMPENSATION: bool,
               DEBUG_MODE: bool):
    OUTPUT_DIR.mkdir(exist_ok=True)  # 创建输出文件夹

    # --- 2. 加载并排序图片 ---

    print("--- 图像拼接开始 ---")
    print(f"配置: {NUM_ROWS}行 x {NUM_COLS}列")
    print(f"图片目录: {IMAGE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"水平重叠预估: {ESTIMATE_OVERLAP_HORIZONTAL_PIXELS}px, 垂直重叠预估: {ESTIMATE_OVERLAP_VERTICAL_PIXELS}px")
    print(f"融合模式: {BLEND_TYPE}, 光照补偿: {'启用' if LIGHT_COMPENSATION else '禁用'}")

    # --- 2. 加载并排序图片 ---
    image_paths = sorted(list(IMAGE_DIR.glob("*.jpg")), key=natural_sort_key)

    if len(image_paths) != NUM_COLS * NUM_ROWS:
        print(f"错误: 找到 {len(image_paths)} 张图片, 但预期需要 {NUM_COLS * NUM_ROWS} 张。")
        return

    # --- 3. 阶段一：水平拼接每一行 ---
    stitched_rows = []
    print("\n--- 阶段一: 水平拼接每一行 ---")

    for i in tqdm(range(NUM_ROWS), desc="处理行"):
        row_start_index = i * NUM_COLS
        row_image_paths = image_paths[row_start_index: row_start_index + NUM_COLS]

        # 加载行的第一张图片
        current_row_image = cv2.imread(str(row_image_paths[0]))
        if current_row_image is None:
            print(f"错误: 无法读取图片 {row_image_paths[0]}")
            continue

        # 依次将该行的后续图片拼接到右侧
        for j in range(1, NUM_COLS):
            # 为每次拼接实例化一个新的Stitcher对象，以隔离调试文件夹
            stitcher_h = ImageStitcherTemplateMatch(
                estimate_overlap_pixels=ESTIMATE_OVERLAP_HORIZONTAL_PIXELS,
                stitch_type="horizontal",
                blend_type=BLEND_TYPE,
                light_uniformity_compensation_enabled=LIGHT_COMPENSATION,
                light_uniformity_compensation_width=30,  # 光照补偿的计算宽度
                debug=DEBUG_MODE,
                debug_dir=str(OUTPUT_DIR / f'debug_h_row{i + 1}_col{j}vs{j + 1}')
            )

            next_image = cv2.imread(str(row_image_paths[j]))
            if next_image is None:
                print(f"错误: 无法读取图片 {row_image_paths[j]}")
                break

            current_row_image = stitcher_h.stitch_main(current_row_image, next_image)

        # 保存拼接好的行
        row_output_path = OUTPUT_DIR / f"stitched_row_{i + 1}.jpg"
        cv2.imwrite(str(row_output_path), current_row_image)
        stitched_rows.append(current_row_image)
        tqdm.write(f"第 {i + 1} 行拼接完成, 已保存至 {row_output_path}")

    # --- 4. 阶段二：垂直拼接所有行 ---
    print("\n--- 阶段二: 垂直拼接所有行 ---")
    if not stitched_rows:
        print("错误: 没有成功拼接的行，无法进行垂直拼接。")
        return

    final_image = stitched_rows[0]

    for i in tqdm(range(1, NUM_ROWS), desc="拼接行"):
        # 实例化垂直拼接器
        stitcher_v = ImageStitcherTemplateMatch(
            estimate_overlap_pixels=ESTIMATE_OVERLAP_VERTICAL_PIXELS,
            stitch_type="vertical",
            blend_type=BLEND_TYPE,
            light_uniformity_compensation_enabled=LIGHT_COMPENSATION,
            light_uniformity_compensation_width=30,
            debug=DEBUG_MODE,
            debug_dir=str(OUTPUT_DIR / f'debug_v_row{i}vs{i + 1}')
        )

        next_row_image = stitched_rows[i]
        final_image = stitcher_v.stitch_main(final_image, next_row_image)

    # --- 5. 保存最终结果 ---
    final_output_path = OUTPUT_DIR / "final_stitched_image.jpg"
    cv2.imwrite(str(final_output_path), final_image)

    print("\n--- 所有拼接任务完成！---")
    print(f"最终的全景图已保存至: {final_output_path}")


def main():
    """
    主执行函数
    """
    # --- 1. 配置参数 ---

    # 图片和输出目录设置
    IMAGE_DIR = Path(r"C:\Code\ML\Project\StitchImageServer\temp\input\_250801_1142_0030")
    # OUTPUT_DIR = Path(r"C:\Code\ML\Project\StitchImageServer\temp\output")

    # 拼图网格设置
    NUM_COLS = 4
    NUM_ROWS = 6

    # ！！！关键拼接参数，您可能需要根据实际图片进行调整！！！
    # 预估水平方向重叠的像素数。如果您的图片宽1920像素，重叠25%，则该值为 1920 * 0.25 ≈ 480
    # 预估垂直方向重叠的像素数。如果您的图片高1080像素，重叠25%，则该值为 1080 * 0.25 ≈ 270
    estimate_overlap_ratio = 0.45
    ESTIMATE_OVERLAP_HORIZONTAL_PIXELS = int(round(1024 * estimate_overlap_ratio))
    ESTIMATE_OVERLAP_VERTICAL_PIXELS = int(round(1024 * estimate_overlap_ratio))

    # 选择融合模式。'blend_half_importance_partial_HSV' 是效果最好但最慢的模式之一
    '''
    前五个都不行
    half_importance,right_first,left_first 0星
    ⭐half_importance_add_weight 2星, 49秒
    half_importance_global_brightness 0星, 49秒
    half_importance_partial_brightness 还行, 4星, 速度适中 ,99秒
    blend_half_importance_partial_HV 不错 5星, 慢, 107秒
    blend_half_importance_partial_SV 不错, 5星, 慢, 108秒
    blend_half_importance_partial_HSV 很不错, 5星, 很慢, 120秒
    ⭐blend_half_importance_partial_brightness_add_weight: 5星, 106秒
    '''

    blend_type_list = ["half_importance_add_weight",
                       "half_importance_global_brightness", "half_importance_partial_brightness",
                       "blend_half_importance_partial_HV", "blend_half_importance_partial_SV",
                       "blend_half_importance_partial_HSV", "blend_half_importance_partial_brightness_add_weight"]
    # BLEND_TYPE = 'blend_half_importance_partial_HSV'

    # 是否开启光照补偿（推荐开启以获得更好效果）
    LIGHT_COMPENSATION = True

    # 是否开启调试模式（会生成大量中间过程图片，用于分析问题）
    DEBUG_MODE = False

    for i, BLEND_TYPE in enumerate(blend_type_list):
        base_dir_path = r"C:\Code\ML\Project\StitchImageServer\temp\output"
        img_dir_name = f"{i}_{BLEND_TYPE}"
        OUTPUT_DIR = Path(os.path.join(base_dir_path, img_dir_name))

        one_img_time = time.time()
        stitch_img(IMAGE_DIR=IMAGE_DIR, OUTPUT_DIR=OUTPUT_DIR, NUM_COLS=NUM_COLS, NUM_ROWS=NUM_ROWS,
                   ESTIMATE_OVERLAP_HORIZONTAL_PIXELS=ESTIMATE_OVERLAP_HORIZONTAL_PIXELS,
                   ESTIMATE_OVERLAP_VERTICAL_PIXELS=ESTIMATE_OVERLAP_VERTICAL_PIXELS,
                   BLEND_TYPE=BLEND_TYPE, LIGHT_COMPENSATION=LIGHT_COMPENSATION,
                   DEBUG_MODE=DEBUG_MODE)
        print()
        print("_"*20)
        print(f"单个用时: {BLEND_TYPE}: {time.time() - one_img_time}")
        print("_"*20)



if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n总耗时: {end_time - start_time:.2f} 秒")
