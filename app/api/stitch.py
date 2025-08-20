import os
import shutil
import uuid
import zipfile
import logging
from typing import List
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


@router.post("/", summary="通用拼图接口")
async def stitch_puzzle(
        background_tasks: BackgroundTasks,
        zip_file: UploadFile = File(..., description="包含一个或多个拼图文件夹的ZIP压缩包。"),
        # --- 通用参数 ---
        method: StitchingMethod = Form(StitchingMethod.TEMPLATE_MATCH, description="选择拼图方法"),
        num_cols: int = Form(4, description="拼图的列数"),
        num_rows: int = Form(6, description="拼图的行数"),
        overlap_h: int = Form(405, description="预估的水平重叠像素"),
        overlap_v: int = Form(440, description="预估的垂直重叠像素"),
        # --- 关键点匹配法 (key_point) 特定参数 ---
        kp_blend_type: KeypointBlendType = Form(KeypointBlendType.HALF_IMPORTANCE_ADD_WEIGHT,
                                                description="[关键点] 融合模式"),
        kp_feature_detector: KeypointFeatureDetector = Form(KeypointFeatureDetector.SIFT,
                                                            description="[关键点] 特征检测器"),
        kp_blend_ratio: float = Form(0.5, description="[关键点] 融合权重 (0.0-1.0)"),
        # --- 模板匹配法 (template_match) 特定参数 ---
        tm_blend_type: TemplateBlendType = Form(TemplateBlendType.HALF_IMPORTANCE_ADD_WEIGHT,
                                                description="[模板匹配] 融合模式"),
        tm_blend_ratio: float = Form(0.5, description="[模板匹配] 融合权重 (0.0-1.0)"),
        tm_light_compensation: bool = Form(True, description="[模板/关键点] 是否启用光照补偿"),
        tm_light_compensation_width: int = Form(15, description="[模板/关键点] 光照补偿宽度 (像素)")
) -> FileResponse:
    """
    上传一个包含拼图文件夹的ZIP压缩包，接口会处理所有文件夹。
    - **如果只处理了一个文件夹**：直接返回拼接好的图片文件。
    - **如果处理了多个文件夹**：将结果打包成一个新的ZIP压缩包返回。
    """
    request_id = str(uuid.uuid4())
    session_dir = TEMP_DIR / request_id
    session_dir.mkdir()
    # 确保无论成功或失败，临时文件夹最终都会被清理
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

    # 智能判断是单文件夹还是多文件夹模式
    sub_items = list(extracted_dir.iterdir())
    if len(sub_items) == 1 and sub_items[0].is_dir():
        # Case 1: ZIP内只有一个顶层文件夹，任务在其子文件夹中
        puzzle_folders = [d for d in sub_items[0].iterdir() if d.is_dir()]
        # 如果子文件夹为空，则认为顶层文件夹本身就是任务
        if not puzzle_folders:
            puzzle_folders = [sub_items[0]]
    else:
        # Case 2: ZIP内有多个文件/文件夹，任务是其中的文件夹
        puzzle_folders = [d for d in sub_items if d.is_dir()]
        # 如果没有子目录，但有图片，则认为整个解压目录是一个任务
        if not puzzle_folders and any(f.suffix.lower() in ['.jpg', '.png', '.jpeg'] for f in sub_items if f.is_file()):
            puzzle_folders = [extracted_dir]

    if not puzzle_folders:
        raise HTTPException(status_code=400, detail="ZIP包中未找到任何包含图片的拼图文件夹。")

    batch_output_dir = session_dir / "batch_output"
    batch_output_dir.mkdir()

    processed_count = 0
    failed_folders = []

    for image_dir in puzzle_folders:
        logging.info(f"--- 开始处理文件夹: {image_dir.name} ---")
        single_output_dir = session_dir / f"output_{image_dir.name}"
        stitched_image_path = None
        try:
            if method == StitchingMethod.KEY_POINT:
                stitched_image_path = stitcher_keypoint.stitch_img(
                    IMAGE_DIR=image_dir, OUTPUT_DIR=single_output_dir,
                    NUM_COLS=num_cols, NUM_ROWS=num_rows,
                    ESTIMATE_OVERLAP_HORIZONTAL_PIXELS=overlap_h,
                    ESTIMATE_OVERLAP_VERTICAL_PIXELS=overlap_v,
                    BLEND_TYPE=kp_blend_type.value, FeatureDetector=kp_feature_detector.value,
                    BLEND_RATIO=kp_blend_ratio,
                    LIGHT_COMPENSATION=tm_light_compensation,
                    LIGHT_COMPENSATION_WIDTH=tm_light_compensation_width,
                    DEBUG_MODE=False
                )
            elif method == StitchingMethod.TEMPLATE_MATCH:
                stitched_image_path = stitcher_template.stitch_img(
                    IMAGE_DIR=image_dir, OUTPUT_DIR=single_output_dir,
                    NUM_COLS=num_cols, NUM_ROWS=num_rows,
                    ESTIMATE_OVERLAP_HORIZONTAL_PIXELS=overlap_h,
                    ESTIMATE_OVERLAP_VERTICAL_PIXELS=overlap_v,
                    BLEND_TYPE=tm_blend_type.value, LIGHT_COMPENSATION=tm_light_compensation,
                    BLEND_RATIO=tm_blend_ratio,
                    LIGHT_COMPENSATION_WIDTH=tm_light_compensation_width,
                    DEBUG_MODE=False
                )

            if stitched_image_path and stitched_image_path.exists():
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

    # --- 核心修改点：根据处理成功的数量决定返回类型 ---
    successful_files = list(batch_output_dir.iterdir())

    # 如果只有一个成功的拼图结果
    if len(successful_files) == 1:
        single_result_path = successful_files[0]
        logging.info(f"检测到单个成功结果，直接返回图片: {single_result_path.name}")
        return FileResponse(
            path=single_result_path,
            filename=single_result_path.name,
            media_type='image/jpeg'
        )

    # 如果有多个成功的结果，或者即使只有一个成功但原始任务也是多个（为了保持一致性）
    else:
        output_zip_name = "stitched_results.zip"
        output_zip_path = session_dir / output_zip_name
        shutil.make_archive(str(output_zip_path.with_suffix('')), 'zip', batch_output_dir)
        logging.info(f"检测到多个成功结果，返回ZIP包: {output_zip_name}")

        if failed_folders:
            logging.warning(f"批量任务完成，但有 {len(failed_folders)} 个文件夹失败: {failed_folders}")

        return FileResponse(
            path=output_zip_path,
            filename=output_zip_name,
            media_type='application/zip'
        )


@router.post("/folder", response_class=FileResponse, summary="单个拼图接口 (直接上传文件)")
async def stitch_puzzle_from_folder(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(..., description="一个文件夹中的所有待拼接图片。"),
        output_filename_base: str = Form("stitched_result", description="输出图片的基础名称（不含扩展名）。"),
        # --- 参数与ZIP接口完全相同 ---
        method: StitchingMethod = Form(StitchingMethod.TEMPLATE_MATCH, description="选择拼图方法"),
        num_cols: int = Form(4, description="拼图的列数"),
        num_rows: int = Form(6, description="拼图的行数"),
        overlap_h: int = Form(405, description="预估的水平重叠像素"),
        overlap_v: int = Form(440, description="预估的垂直重叠像素"),
        kp_blend_type: KeypointBlendType = Form(KeypointBlendType.HALF_IMPORTANCE_ADD_WEIGHT,
                                                description="[关键点] 融合模式"),
        kp_feature_detector: KeypointFeatureDetector = Form(KeypointFeatureDetector.SIFT,
                                                            description="[关键点] 特征检测器"),
        kp_blend_ratio: float = Form(0.5, description="[关键点] 融合权重 (0.0-1.0)"),
        tm_blend_type: TemplateBlendType = Form(TemplateBlendType.HALF_IMPORTANCE_ADD_WEIGHT,
                                                description="[模板匹配] 融合模式"),
        tm_blend_ratio: float = Form(0.5, description="[模板匹配] 融合权重 (0.0-1.0)"),
        tm_light_compensation: bool = Form(True, description="[模板/关键点] 是否启用光照补偿"),
        tm_light_compensation_width: int = Form(15, description="[模板/关键点] 光照补偿宽度 (像素)")
):
    """
    上传一个文件夹内的所有图片进行拼接，直接返回拼接好的单张大图。
    此接口专为无法或不便在客户端进行ZIP压缩的场景设计。
    """
    if not files:
        raise HTTPException(status_code=400, detail="没有上传任何文件。")

    request_id = str(uuid.uuid4())
    session_dir = TEMP_DIR / request_id
    session_dir.mkdir()
    background_tasks.add_task(cleanup_temp_folder, session_dir)

    # 在会话目录中创建一个子目录来存放上传的图片，模拟一个文件夹
    image_dir = session_dir / "images"
    image_dir.mkdir()

    for upload_file in files:
        file_path = image_dir / upload_file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

    output_dir = session_dir / "output"
    stitched_image_path = None

    try:
        if method == StitchingMethod.KEY_POINT:
            stitched_image_path = stitcher_keypoint.stitch_img(
                IMAGE_DIR=image_dir, OUTPUT_DIR=output_dir, NUM_COLS=num_cols, NUM_ROWS=num_rows,
                ESTIMATE_OVERLAP_HORIZONTAL_PIXELS=overlap_h, ESTIMATE_OVERLAP_VERTICAL_PIXELS=overlap_v,
                BLEND_TYPE=kp_blend_type.value, FeatureDetector=kp_feature_detector.value, BLEND_RATIO=kp_blend_ratio,
                LIGHT_COMPENSATION=tm_light_compensation, LIGHT_COMPENSATION_WIDTH=tm_light_compensation_width,
                DEBUG_MODE=False
            )
        elif method == StitchingMethod.TEMPLATE_MATCH:
            stitched_image_path = stitcher_template.stitch_img(
                IMAGE_DIR=image_dir, OUTPUT_DIR=output_dir, NUM_COLS=num_cols, NUM_ROWS=num_rows,
                ESTIMATE_OVERLAP_HORIZONTAL_PIXELS=overlap_h, ESTIMATE_OVERLAP_VERTICAL_PIXELS=overlap_v,
                BLEND_TYPE=tm_blend_type.value, LIGHT_COMPENSATION=tm_light_compensation, BLEND_RATIO=tm_blend_ratio,
                LIGHT_COMPENSATION_WIDTH=tm_light_compensation_width, DEBUG_MODE=False
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片拼接过程中发生内部错误: {e}")

    if not stitched_image_path or not stitched_image_path.exists():
        raise HTTPException(status_code=500, detail=f"图片拼接失败，未能生成结果文件。")

    final_filename = f"{output_filename_base}.jpg"
    # 我们直接从最终的输出路径返回，不需要移动文件
    return FileResponse(
        path=stitched_image_path,
        filename=final_filename,
        media_type='image/jpeg'
    )