import cv2
import os
import time
from pathlib import Path
import re
from tqdm import tqdm
import concurrent.futures

from fry_project_classes.stitch_img_template_match import ImageStitcherTemplateMatch


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


# --- 新增：用于并行处理的"任务单元"函数 ---
def stitch_single_row(row_index, row_image_paths, stitch_params):
    """
    负责拼接单一一行的图片。这个函数将在独立的进程中运行。

    Args:
        row_index (int): 当前行的索引（从0开始），用于日志和调试文件命名。
        row_image_paths (list): 这一行所有图片的路径列表。
        stitch_params (dict): 包含所有拼接所需参数的字典。

    Returns:
        tuple: 包含行索引和拼接完成的图像 (row_index, stitched_row_image)。
    """
    # 从参数字典中解包
    NUM_COLS = len(row_image_paths)
    OUTPUT_DIR = stitch_params['OUTPUT_DIR']
    ESTIMATE_OVERLAP_HORIZONTAL_PIXELS = stitch_params['ESTIMATE_OVERLAP_HORIZONTAL_PIXELS']
    BLEND_TYPE = stitch_params['BLEND_TYPE']
    LIGHT_COMPENSATION = stitch_params['LIGHT_COMPENSATION']
    DEBUG_MODE = stitch_params['DEBUG_MODE']

    # 加载行的第一张图片
    current_row_image = cv2.imread(str(row_image_paths[0]))
    if current_row_image is None:
        print(f"错误: 无法读取图片 {row_image_paths[0]}")
        return row_index, None

    # 依次将该行的后续图片拼接到右侧
    for j in range(1, NUM_COLS):
        stitcher_h = ImageStitcherTemplateMatch(
            estimate_overlap_pixels=ESTIMATE_OVERLAP_HORIZONTAL_PIXELS,
            stitch_type="horizontal",
            blend_type=BLEND_TYPE,
            light_uniformity_compensation_enabled=LIGHT_COMPENSATION,
            light_uniformity_compensation_width=30,
            debug=DEBUG_MODE,
            # 注意调试目录的命名，确保不同进程不会写入同一个文件夹
            debug_dir=str(OUTPUT_DIR / f'debug_h_row{row_index + 1}_col{j}vs{j + 1}')
        )

        next_image = cv2.imread(str(row_image_paths[j]))
        if next_image is None:
            print(f"错误: 无法读取图片 {row_image_paths[j]}")
            # 如果中间一张图片读取失败，返回当前已拼接的部分
            return row_index, current_row_image

        current_row_image = stitcher_h.stitch_main(current_row_image, next_image)

    # 返回拼接结果和行索引，以便主进程能按正确顺序排列
    return row_index, current_row_image


# --- 优化后的主拼接函数 ---
def stitch_img(IMAGE_DIR, OUTPUT_DIR, NUM_COLS: int, NUM_ROWS: int,
               ESTIMATE_OVERLAP_HORIZONTAL_PIXELS: int, ESTIMATE_OVERLAP_VERTICAL_PIXELS: int,
               BLEND_TYPE: str, LIGHT_COMPENSATION: bool,
               DEBUG_MODE: bool):
    OUTPUT_DIR.mkdir(exist_ok=True)

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

    # --- 3. 阶段一：并行水平拼接每一行 (核心优化点) ---
    print("\n--- 阶段一: 并行水平拼接每一行 ---")

    # 准备传递给每个进程的参数
    stitch_params = {
        'OUTPUT_DIR': OUTPUT_DIR,
        'ESTIMATE_OVERLAP_HORIZONTAL_PIXELS': ESTIMATE_OVERLAP_HORIZONTAL_PIXELS,
        'BLEND_TYPE': BLEND_TYPE,
        'LIGHT_COMPENSATION': LIGHT_COMPENSATION,
        'DEBUG_MODE': DEBUG_MODE
    }

    stitched_rows = [None] * NUM_ROWS  # 预先分配列表，用于按顺序存放结果

    # 使用进程池执行器
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 提交所有行的拼接任务
        futures = []
        for i in range(NUM_ROWS):
            row_start_index = i * NUM_COLS
            row_image_paths = image_paths[row_start_index: row_start_index + NUM_COLS]
            # 提交任务到进程池
            future = executor.submit(stitch_single_row, i, row_image_paths, stitch_params)
            futures.append(future)

        # 使用tqdm来显示进度条，并收集结果
        # as_completed会在任务完成时立即返回，这比直接等待所有任务更具响应性
        for future in tqdm(concurrent.futures.as_completed(futures), total=NUM_ROWS, desc="处理行"):
            try:
                row_index, result_image = future.result()
                if result_image is not None:
                    stitched_rows[row_index] = result_image
                    # 保存拼接好的行
                    row_output_path = OUTPUT_DIR / f"stitched_row_{row_index + 1}.jpg"
                    cv2.imwrite(str(row_output_path), result_image)
                    tqdm.write(f"第 {row_index + 1} 行拼接完成, 已保存至 {row_output_path}")
                else:
                    tqdm.write(f"第 {row_index + 1} 行拼接失败。")
            except Exception as exc:
                tqdm.write(f"一个行拼接任务生成了异常: {exc}")

    # 检查是否有失败的行
    if any(row is None for row in stitched_rows):
        print("错误: 存在拼接失败的行，无法进行垂直拼接。")
        return

    # --- 4. 阶段二：垂直拼接所有行 (这部分保持串行) ---
    print("\n--- 阶段二: 垂直拼接所有行 ---")

    final_image = stitched_rows[0]

    for i in tqdm(range(1, NUM_ROWS), desc="拼接行"):
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
    IMAGE_DIR = Path(r"C:\Code\ML\Project\StitchImageServer\temp\Input\_250801_1146_0034")

    # 拼图网格设置
    NUM_COLS = 4
    NUM_ROWS = 6

    # 预估重叠像素
    ESTIMATE_OVERLAP_HORIZONTAL_PIXELS = 405
    ESTIMATE_OVERLAP_VERTICAL_PIXELS = 440

    # 融合模式列表
    # 默认 half_importance_add_weight
    blend_type_list = ["half_importance_add_weight",
                       "half_importance_global_brightness", "half_importance_partial_brightness",
                       "blend_half_importance_partial_HV", "blend_half_importance_partial_SV",
                       "blend_half_importance_partial_HSV", "blend_half_importance_partial_brightness_add_weight"]

    LIGHT_COMPENSATION = True
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
        print("_" * 20)
        print(f"单个用时: {img_dir_name}: {time.time() - one_img_time}")
        print("_" * 20)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n总耗时: {end_time - start_time:.2f} 秒")