import cv2
from pathlib import Path
import concurrent.futures
import logging
from tqdm import tqdm

from fry_project_classes.stitch_img_key_point import ImageStitcherKeyPoint
from utils.utils import natural_sort_key

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def stitch_single_row_keypoint(row_index, row_image_paths, stitch_params):
    NUM_COLS = len(row_image_paths)
    OUTPUT_DIR = stitch_params['OUTPUT_DIR']
    ESTIMATE_OVERLAP_HORIZONTAL_PIXELS = stitch_params['ESTIMATE_OVERLAP_HORIZONTAL_PIXELS']
    BLEND_TYPE = stitch_params['BLEND_TYPE']
    FeatureDetector = stitch_params['FeatureDetector']
    DEBUG_MODE = stitch_params['DEBUG_MODE']

    current_row_image = cv2.imread(str(row_image_paths[0]))
    if current_row_image is None:
        logging.error(f"错误: 无法读取图片 {row_image_paths[0]}")
        return row_index, None

    for j in range(1, NUM_COLS):
        stitcher_h = ImageStitcherKeyPoint(
            estimate_overlap_pixels=ESTIMATE_OVERLAP_HORIZONTAL_PIXELS,
            stitch_type="horizontal",
            blend_type=BLEND_TYPE,
            feature_detector=FeatureDetector,
            blend_ratio=0.5,
            debug=DEBUG_MODE,
            debug_dir=str(OUTPUT_DIR / f'debug_h_row{row_index + 1}_col{j}vs{j + 1}')
        )
        next_image = cv2.imread(str(row_image_paths[j]))
        if next_image is None:
            logging.error(f"错误: 无法读取图片 {row_image_paths[j]}")
            return row_index, current_row_image
        current_row_image = stitcher_h.stitch_main(current_row_image, next_image)
    return row_index, current_row_image


def stitch_img(IMAGE_DIR: Path, OUTPUT_DIR: Path, NUM_COLS: int, NUM_ROWS: int,
               ESTIMATE_OVERLAP_HORIZONTAL_PIXELS: int, ESTIMATE_OVERLAP_VERTICAL_PIXELS: int,
               BLEND_TYPE: str, FeatureDetector: str,
               DEBUG_MODE: bool) -> Path | None:
    """
    基于关键点的图像拼接函数。
    成功时返回最终图像的路径，失败时返回 None。
    """
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    logging.info("--- [关键点] 图像拼接开始 ---")
    logging.info(f"配置: {NUM_ROWS}行 x {NUM_COLS}列, 图片目录: {IMAGE_DIR}")

    image_paths = sorted(list(IMAGE_DIR.glob("*.jpg")), key=natural_sort_key)
    if len(image_paths) != NUM_COLS * NUM_ROWS:
        logging.error(f"错误: 找到 {len(image_paths)} 张图片, 但预期需要 {NUM_COLS * NUM_ROWS} 张。")
        return None


    # --- 阶段一：并行水平拼接每一行 ---
    logging.info("--- 阶段一: 并行水平拼接每一行 ---")
    stitch_params = {
        'OUTPUT_DIR': OUTPUT_DIR,
        'ESTIMATE_OVERLAP_HORIZONTAL_PIXELS': ESTIMATE_OVERLAP_HORIZONTAL_PIXELS,
        'BLEND_TYPE': BLEND_TYPE,
        'FeatureDetector': FeatureDetector,
        'DEBUG_MODE': DEBUG_MODE
    }
    stitched_rows = [None] * NUM_ROWS
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(stitch_single_row_keypoint, i, image_paths[i * NUM_COLS: i * NUM_COLS + NUM_COLS],
                                   stitch_params) for i in range(NUM_ROWS)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=NUM_ROWS, desc="[关键点]处理行"):
            try:
                row_index, result_image = future.result()
                if result_image is not None:
                    stitched_rows[row_index] = result_image
                else:
                    logging.warning(f"第 {row_index + 1} 行拼接失败。")
            except Exception as exc:
                logging.error(f"一个行拼接任务生成了异常: {exc}")

    if any(row is None for row in stitched_rows):
        logging.error("错误: 存在拼接失败的行，无法进行垂直拼接。")
        return None

    # --- 阶段二：垂直拼接所有行 ---
    logging.info("--- 阶段二: 垂直拼接所有行 ---")
    final_image = stitched_rows[0]
    for i in tqdm(range(1, NUM_ROWS), desc="[关键点]拼接行"):
        stitcher_v = ImageStitcherKeyPoint(
            estimate_overlap_pixels=ESTIMATE_OVERLAP_VERTICAL_PIXELS, stitch_type="vertical",
            blend_type=BLEND_TYPE, feature_detector=FeatureDetector, blend_ratio=0.5,
            debug=DEBUG_MODE, debug_dir=str(OUTPUT_DIR / f'debug_v_row{i}vs{i + 1}')
        )
        next_row_image = stitched_rows[i]
        final_image = stitcher_v.stitch_main(final_image, next_row_image)

    # --- 保存并返回结果 ---
    final_output_path = OUTPUT_DIR / "final_stitched_image.jpg"
    cv2.imwrite(str(final_output_path), final_image)
    logging.info(f"--- [关键点] 拼接任务完成！最终图已暂存至: {final_output_path} ---")

    return final_output_path
