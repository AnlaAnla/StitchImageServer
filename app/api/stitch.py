import os
import shutil
import uuid
import zipfile
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

# 导入我们的核心逻辑和数据模型
from app.core import stitcher_keypoint, stitcher_template
from app.schemas import StitchingMethod, KeypointFeatureDetector, KeypointBlendType, TemplateBlendType

from utils.utils import cleanup_temp_folder

router = APIRouter(prefix="/stitch", tags=['拼图'])

TEMP_DIR = Path("_temp_work")
TEMP_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@router.post("/", response_class=FileResponse, summary="单个拼图接口")
async def stitch_single_puzzle(
        background_tasks: BackgroundTasks,
        zip_file: UploadFile = File(..., description="包含一个文件夹的ZIP压缩包，文件夹内有24张小图。"),
        # --- 通用参数 ---
        method: StitchingMethod = Form(StitchingMethod.TEMPLATE_MATCH, description="选择拼图方法"),
        num_cols: int = Form(4, description="拼图的列数"),
        num_rows: int = Form(6, description="拼图的行数"),
        overlap_h: int = Form(405, description="预估的水平重叠像素"),
        overlap_v: int = Form(440, description="预估的垂直重叠像素"),
        # --- 点匹配法 (key_point) 特定参数 ---
        kp_blend_type: KeypointBlendType = Form(KeypointBlendType.COMBINE, description="[点匹配] 融合模式"),
        kp_feature_detector: KeypointFeatureDetector = Form(KeypointFeatureDetector.SIFT,
                                                            description="[点匹配] 特征检测器"),
        # --- 模板匹配法 (template_match) 特定参数 ---
        tm_blend_type: TemplateBlendType = Form(TemplateBlendType.HALF_IMPORTANCE_ADD_WEIGHT,
                                                description="[模板匹配] 融合模式"),
        tm_light_compensation: bool = Form(True, description="[模板匹配] 是否启用光照补偿")
):
    """
    上传一个包含24张小图的文件夹的ZIP压缩包，接口会将其拼接成一张大图并返回。

    - **zip_file**: 必须是.zip格式，内部应仅包含一个文件夹，该文件夹内含所有待拼接的.jpg图片。
    - **返回**: 拼接成功后，返回拼接好的图片文件，文件名与ZIP包内的文件夹名相同。
    """
    request_id = str(uuid.uuid4())
    session_dir = TEMP_DIR / request_id
    session_dir.mkdir()
    background_tasks.add_task(cleanup_temp_folder, session_dir)

    zip_path = session_dir / zip_file.filename
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(zip_file.file, buffer)

    extracted_dir = session_dir / "extracted"
    extracted_dir.mkdir()

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extracted_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="上传的文件不是有效的ZIP格式。")

    image_dir = extracted_dir
    output_dir = session_dir / "output"

    # 根据选择的方法调用不同的拼接函数
    stitched_image_path = None
    if method == StitchingMethod.KEY_POINT:
        stitched_image_path = stitcher_keypoint.stitch_img(
            IMAGE_DIR=image_dir, OUTPUT_DIR=output_dir, NUM_COLS=num_cols, NUM_ROWS=num_rows,
            ESTIMATE_OVERLAP_HORIZONTAL_PIXELS=overlap_h, ESTIMATE_OVERLAP_VERTICAL_PIXELS=overlap_v,
            BLEND_TYPE=kp_blend_type.value, FeatureDetector=kp_feature_detector.value,
            DEBUG_MODE=False
        )
    elif method == StitchingMethod.TEMPLATE_MATCH:
        stitched_image_path = stitcher_template.stitch_img(
            IMAGE_DIR=image_dir, OUTPUT_DIR=output_dir, NUM_COLS=num_cols, NUM_ROWS=num_rows,
            ESTIMATE_OVERLAP_HORIZONTAL_PIXELS=overlap_h, ESTIMATE_OVERLAP_VERTICAL_PIXELS=overlap_v,
            BLEND_TYPE=tm_blend_type.value, LIGHT_COMPENSATION=tm_light_compensation,
            DEBUG_MODE=False
        )

    if not stitched_image_path or not stitched_image_path.exists():
        raise HTTPException(status_code=500, detail=f"图片拼接失败，请检查服务器日志（请求ID: {request_id}）。")

    # 使用原始文件夹名命名输出图片
    final_filename = f"{image_dir.name}.jpg"
    final_filepath = stitched_image_path.rename(stitched_image_path.parent / final_filename)

    return FileResponse(
        path=final_filepath,
        filename=final_filename,
        media_type='image/jpeg'
    )


@router.post("/batch", response_class=FileResponse, summary="批量拼图接口")
async def stitch_batch_puzzles(
        background_tasks: BackgroundTasks,
        zip_file: UploadFile = File(..., description="包含多个拼图文件夹的ZIP压缩包。"),
        # 参数与单个拼图接口相同
        method: StitchingMethod = Form(StitchingMethod.TEMPLATE_MATCH, description="选择拼图方法"),
        num_cols: int = Form(4, description="拼图的列数"),
        num_rows: int = Form(6, description="拼图的行数"),
        overlap_h: int = Form(405, description="预估的水平重叠像素"),
        overlap_v: int = Form(440, description="预估的垂直重叠像素"),
        kp_blend_type: KeypointBlendType = Form(KeypointBlendType.COMBINE, description="[点匹配] 融合模式"),
        kp_feature_detector: KeypointFeatureDetector = Form(KeypointFeatureDetector.SIFT,
                                                            description="[点匹配] 特征检测器"),
        tm_blend_type: TemplateBlendType = Form(TemplateBlendType.HALF_IMPORTANCE_ADD_WEIGHT,
                                                description="[模板匹配] 融合模式"),
        tm_light_compensation: bool = Form(True, description="[模板匹配] 是否启用光照补偿"),
):
    """
    上传一个包含多个拼图文件夹的ZIP压缩包，接口会处理所有文件夹，并将结果打包成一个新的ZIP返回。

    - **zip_file**: 必须是.zip格式，内部可以有多个文件夹，每个文件夹都包含待拼接的图片。
    - **返回**: 一个ZIP压缩包，里面是所有拼接好的图片，每张图片以其对应的原文件夹名命名。
    """
    request_id = str(uuid.uuid4())
    session_dir = TEMP_DIR / request_id
    session_dir.mkdir()
    background_tasks.add_task(cleanup_temp_folder, session_dir)

    zip_path = session_dir / zip_file.filename
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(zip_file.file, buffer)

    extracted_dir = session_dir / "extracted"
    extracted_dir.mkdir()
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extracted_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="上传的文件不是有效的ZIP格式。")

    puzzle_folders = [d for d in extracted_dir.iterdir() if d.is_dir()]
    if not puzzle_folders:
        raise HTTPException(status_code=400, detail="ZIP包中未找到任何拼图文件夹。")

    batch_output_dir = session_dir / "batch_output"
    batch_output_dir.mkdir()

    processed_count = 0
    failed_folders = []

    for image_dir in puzzle_folders:
        logging.info(f"--- 开始处理批量任务中的文件夹: {image_dir.name} ---")
        # 为每个子任务创建一个独立的输出目录
        single_output_dir = session_dir / "single_output"
        if single_output_dir.exists():
            shutil.rmtree(single_output_dir)  # 清理上一次循环的输出

        stitched_image_path = None
        try:
            if method == StitchingMethod.KEY_POINT:
                stitched_image_path = stitcher_keypoint.stitch_img(
                    IMAGE_DIR=image_dir, OUTPUT_DIR=single_output_dir,
                    NUM_COLS=num_cols, NUM_ROWS=num_rows,
                    ESTIMATE_OVERLAP_HORIZONTAL_PIXELS=overlap_h,
                    ESTIMATE_OVERLAP_VERTICAL_PIXELS=overlap_v,
                    BLEND_TYPE=kp_blend_type.value, FeatureDetector=kp_feature_detector.value,
                    DEBUG_MODE=False
                )
            elif method == StitchingMethod.TEMPLATE_MATCH:
                stitched_image_path = stitcher_template.stitch_img(
                    IMAGE_DIR=image_dir, OUTPUT_DIR=single_output_dir,
                    NUM_COLS=num_cols, NUM_ROWS=num_rows,
                    ESTIMATE_OVERLAP_HORIZONTAL_PIXELS=overlap_h,
                    ESTIMATE_OVERLAP_VERTICAL_PIXELS=overlap_v,
                    BLEND_TYPE=tm_blend_type.value, LIGHT_COMPENSATION=tm_light_compensation,
                    DEBUG_MODE=False
                )

            if stitched_image_path and stitched_image_path.exists():
                # 将成功的结果移到最终的批量输出目录
                target_path = batch_output_dir / f"{image_dir.name}.jpg"
                shutil.move(str(stitched_image_path), str(target_path))
                processed_count += 1
            else:
                logging.error(f"文件夹 {image_dir.name} 拼接失败。")
                failed_folders.append(image_dir.name)
        except Exception as e:
            logging.error(f"处理文件夹 {image_dir.name} 时发生严重错误: {e}")
            failed_folders.append(image_dir.name)

    if processed_count == 0:
        detail_msg = f"所有 {len(puzzle_folders)} 个文件夹都拼接失败。失败列表: {failed_folders}"
        raise HTTPException(status_code=500, detail=detail_msg)

    # 将最终结果打包成ZIP
    output_zip_path = session_dir / "stitched_results.zip"
    shutil.make_archive(str(output_zip_path.with_suffix('')), 'zip', batch_output_dir)

    if failed_folders:
        logging.warning(f"批量任务完成，但有 {len(failed_folders)} 个文件夹失败: {failed_folders}")
        # 可以在响应头中添加自定义信息来通知客户端部分失败
        # response.headers["X-Failed-Folders"] = ",".join(failed_folders)

    return FileResponse(
        path=output_zip_path,
        filename="stitched_results.zip",
        media_type='application/zip'
    )
