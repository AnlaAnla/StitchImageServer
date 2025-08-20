import cv2
import os
import time
from pathlib import Path
import re
from tqdm import tqdm
import cv2
# 导入您提供的拼接器类和拼接顺序生成器
from fry_project_classes.stitch_img_template_match import ImageStitcherTemplateMatch
from fry_project_classes.get_full_stitch_order import get_full_stitch_order

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
               DEBUG_MODE: bool, BLEND_RATIO: float, LIGHT_COMPENSATION_WIDTH: int):
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("--- 图像拼接开始 ---")
    print(f"配置: {NUM_ROWS}行 x {NUM_COLS}列")
    print(f"图片目录: {IMAGE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"水平重叠预估: {ESTIMATE_OVERLAP_HORIZONTAL_PIXELS}px, 垂直重叠预估: {ESTIMATE_OVERLAP_VERTICAL_PIXELS}px")
    print(f"融合模式: {BLEND_TYPE}, 权重: {BLEND_RATIO}")
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
        # 使用从 '1' 开始的字符串作为键，模仿 stitch_worker.py 的行为
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
        stitcher = ImageStitcherTemplateMatch(
            estimate_overlap_pixels=overlap_pixels,
            stitch_type=direction,
            blend_type=BLEND_TYPE,
            blend_ratio=BLEND_RATIO,
            light_uniformity_compensation_enabled=LIGHT_COMPENSATION,
            light_uniformity_compensation_width=LIGHT_COMPENSATION_WIDTH,
            debug=DEBUG_MODE,
            debug_dir=str(OUTPUT_DIR / f'debug_{result_name}')
        )

        # 执行拼接
        stitched_image = stitcher.stitch_main(img1, img2)

        # 将新生成的图片存入字典，用于下一步拼接
        images_dict[result_name] = stitched_image
        final_image = stitched_image  # 始终保留最新的拼接结果

        if DEBUG_MODE:
            # 保存每一步的中间结果
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
    # OUTPUT_DIR = Path(r"C:\Code\ML\Project\StitchImageServer\temp\output")

    # 拼图网格设置
    NUM_COLS = 4
    NUM_ROWS = 6

    # ！！！关键拼接参数，您可能需要根据实际图片进行调整！！！
    # 预估水平方向重叠的像素数。如果您的图片宽1920像素，重叠25%，则该值为 1920 * 0.25 ≈ 480
    # 预估垂直方向重叠的像素数。如果您的图片高1080像素，重叠25%，则该值为 1080 * 0.25 ≈ 270
    # estimate_overlap_ratio = 0.45
    ESTIMATE_OVERLAP_HORIZONTAL_PIXELS = 405
    ESTIMATE_OVERLAP_VERTICAL_PIXELS = 440

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
                       # "half_importance_global_brightness", "half_importance_partial_brightness",
                       # "blend_half_importance_partial_HV", "blend_half_importance_partial_SV",
                       # "blend_half_importance_partial_HSV", "blend_half_importance_partial_brightness_add_weight"
                       ]
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
                   BLEND_TYPE=BLEND_TYPE, BLEND_RATIO=0.5,
                   LIGHT_COMPENSATION=LIGHT_COMPENSATION, LIGHT_COMPENSATION_WIDTH=15,
                   DEBUG_MODE=DEBUG_MODE)
        print()
        print("_" * 20)
        print(f"单个用时: {BLEND_TYPE}: {time.time() - one_img_time}")
        print("_" * 20)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n总耗时: {end_time - start_time:.2f} 秒")
