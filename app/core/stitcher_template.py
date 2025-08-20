
import cv2
import logging
from pathlib import Path
from tqdm import tqdm

from fry_project_classes.stitch_img_template_match import ImageStitcherTemplateMatch
from fry_project_classes.get_full_stitch_order import get_full_stitch_order
from utils.utils import natural_sort_key

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def stitch_img(IMAGE_DIR: Path, OUTPUT_DIR: Path, NUM_COLS: int, NUM_ROWS: int,
               ESTIMATE_OVERLAP_HORIZONTAL_PIXELS: int, ESTIMATE_OVERLAP_VERTICAL_PIXELS: int,
               BLEND_TYPE: str, LIGHT_COMPENSATION: bool,
               # --- 新增的关键参数 ---
               BLEND_RATIO: float,
               LIGHT_COMPENSATION_WIDTH: int,
               DEBUG_MODE: bool) -> Path | None:
    """
    基于模板匹配的图像拼接函数，使用优化的拼接顺序。
    成功时返回最终图像的路径，失败时返回 None。
    """
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    logging.info("--- [模板匹配] 图像拼接开始 (优化顺序) ---")

    # 1. 加载并排序所有图片
    image_paths = sorted(list(IMAGE_DIR.glob("*.*")), key=natural_sort_key)  # 支持更多图片格式
    if len(image_paths) < NUM_COLS * NUM_ROWS:
        logging.error(f"错误: 找到 {len(image_paths)} 张图片, 但预期需要 {NUM_COLS * NUM_ROWS} 张。")
        return None

    images_dict = {}
    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        if img is None:
            logging.error(f"错误: 无法读取图片 {path}")
            return None
        images_dict[str(i + 1)] = img

    # 2. 获取拼接顺序
    full_stitch_order_dict = get_full_stitch_order(NUM_ROWS, NUM_COLS)
    logging.info(f"获取到 {len(full_stitch_order_dict)} 步拼接指令")

    # 3. 按照指令集执行拼接
    final_image = None
    iterator = tqdm(full_stitch_order_dict.items(), desc="[模板]执行拼接", total=len(full_stitch_order_dict))

    for step, (round_num, img1_name, img2_name, direction, result_name) in iterator:
        img1 = images_dict[img1_name]
        img2 = images_dict[img2_name]

        overlap_pixels = ESTIMATE_OVERLAP_HORIZONTAL_PIXELS if direction == 'horizontal' else ESTIMATE_OVERLAP_VERTICAL_PIXELS

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

        stitched_image = stitcher.stitch_main(img1, img2)
        if stitched_image is None:
            logging.error(f"步骤 {step} 拼接失败: {img1_name} + {img2_name}")
            return None

        images_dict[result_name] = stitched_image
        final_image = stitched_image

    # 4. 保存并返回结果
    if final_image is not None:
        final_output_path = OUTPUT_DIR / "final_stitched_image.jpg"
        cv2.imwrite(str(final_output_path), final_image)
        logging.info(f"--- [模板匹配] 拼接任务完成！最终图已暂存至: {final_output_path} ---")
        return final_output_path
    else:
        logging.error("拼接失败，未能生成最终图像。")
        return None