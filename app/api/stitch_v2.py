import os
import shutil
import uuid
import zipfile
import logging
from pathlib import Path
from typing import List  # 导入 List 类型

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


# --- 内部辅助函数，用于处理单个拼图任务，避免代码重复 ---
def _process_single_puzzle(
    image_dir: Path,
    output_dir: Path,
    method: StitchingMethod,
    params: dict
) -> Path | None:
    """
    处理单个拼图任务的核心逻辑。

    :param image_dir: 包含所有小图的输入目录。
    :param output_dir: 存放拼接结果的输出目录。
    :param method: 拼图方法。
    :param params: 包含所有拼图参数的字典。
    :return: 成功则返回拼接后图片的路径，否则返回 None。
    """
    output_dir.mkdir(exist_ok=True)
    stitched_image_path = None
    try:
        if method == StitchingMethod.KEY_POINT:
            stitched_image_path = stitcher_keypoint.stitch_img(
                IMAGE_DIR=image_dir, OUTPUT_DIR=output_dir,
                NUM_COLS=params["num_cols"], NUM_ROWS=params["num_rows"],
                ESTIMATE_OVERLAP_HORIZONTAL_PIXELS=params["overlap_h"],
                ESTIMATE_OVERLAP_VERTICAL_PIXELS=params["overlap_v"],
                BLEND_TYPE=params["kp_blend_type"].value,
                FeatureDetector=params["kp_feature_detector"].value,
                DEBUG_MODE=False
            )
        elif method == StitchingMethod.TEMPLATE_MATCH:
            stitched_image_path = stitcher_template.stitch_img(
                IMAGE_DIR=image_dir, OUTPUT_DIR=output_dir,
                NUM_COLS=params["num_cols"], NUM_ROWS=params["num_rows"],
                ESTIMATE_OVERLAP_HORIZONTAL_PIXELS=params["overlap_h"],
                ESTIMATE_OVERLAP_VERTICAL_PIXELS=params["overlap_v"],
                BLEND_TYPE=params["tm_blend_type"].value,
                LIGHT_COMPENSATION=params["tm_light_compensation"],
                DEBUG_MODE=False
            )
    except Exception as e:
        logging.error(f"处理文件夹 {image_dir.name} 时发生错误: {e}")
        return None

    return stitched_image_path


# --- 新的单个拼图接口 (直接上传图片文件夹) ---
@router.post("/from-folder", response_class=FileResponse, summary="单个拼图接口 (从文件夹上传)")
async def stitch_single_from_folder(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(..., description="一个文件夹中的所有待拼接图片。"),
        output_filename_base: str = Form(..., description="输出图片的基础名称（不含扩展名），例如 'puzzle_A'。"),
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
    上传一个文件夹内的所有图片进行拼接，直接返回拼接好的单张大图。

    - **files**: 选择一个文件夹中的所有图片进行上传。
    - **output_filename_base**: 为你的拼图任务命名，这个名字将作为返回图片的文件名。
    - **返回**: 拼接成功后，返回拼接好的图片文件。
    """
    if not files:
        raise HTTPException(status_code=400, detail="没有上传任何文件。")

    request_id = str(uuid.uuid4())
    session_dir = TEMP_DIR / request_id
    session_dir.mkdir()
    background_tasks.add_task(cleanup_temp_folder, session_dir)

    # 创建用于存放上传图片的临时目录
    image_dir = session_dir / "images"
    image_dir.mkdir()

    # 保存所有上传的文件
    for upload_file in files:
        file_path = image_dir / upload_file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

    output_dir = session_dir / "output"

    # 将所有参数打包到一个字典中，方便传递
    params = locals()

    # 调用核心处理函数
    stitched_image_path = _process_single_puzzle(image_dir, output_dir, method, params)

    if not stitched_image_path or not stitched_image_path.exists():
        raise HTTPException(status_code=500, detail=f"图片拼接失败，请检查服务器日志（请求ID: {request_id}）。")

    # 使用用户提供的基础名称命名输出图片
    final_filename = f"{output_filename_base}.jpg"
    final_filepath = stitched_image_path.rename(stitched_image_path.parent / final_filename)

    return FileResponse(
        path=final_filepath,
        filename=final_filename,
        media_type='image/jpeg'
    )


# --- 新的批量拼图接口 (上传一个拼图文件夹，返回ZIP) ---
@router.post("/batch/from-folder", response_class=FileResponse, summary="批量拼图接口 (单文件夹上传, ZIP返回)")
async def stitch_batch_from_folder(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(..., description="一个文件夹中的所有待拼接图片。"),
        output_filename_base: str = Form(..., description="输出图片的基础名称（不含扩展名），例如 'puzzle_A'。"),
        # --- 参数与单个拼图接口相同 ---
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
    上传一个文件夹内的所有图片进行拼接，将结果打包成一个ZIP压缩文件返回。

    - **files**: 选择一个文件夹中的所有图片进行上传。
    - **output_filename_base**: 为你的拼图任务命名，这个名字将作为ZIP包内图片的文件名。
    - **返回**: 一个ZIP压缩包，里面包含拼接好的单张图片。
    """
    if not files:
        raise HTTPException(status_code=400, detail="没有上传任何文件。")

    request_id = str(uuid.uuid4())
    session_dir = TEMP_DIR / request_id
    session_dir.mkdir()
    background_tasks.add_task(cleanup_temp_folder, session_dir)

    image_dir = session_dir / "images"
    image_dir.mkdir()

    for upload_file in files:
        file_path = image_dir / upload_file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

    single_output_dir = session_dir / "single_output"
    params = locals()

    stitched_image_path = _process_single_puzzle(image_dir, single_output_dir, method, params)

    if not stitched_image_path or not stitched_image_path.exists():
        raise HTTPException(status_code=500, detail=f"图片拼接失败，请检查服务器日志（请求ID: {request_id}）。")

    # --- 将单个结果打包成ZIP ---
    batch_output_dir = session_dir / "batch_output"
    batch_output_dir.mkdir()

    # 将成功的结果移到最终的批量输出目录
    target_path = batch_output_dir / f"{output_filename_base}.jpg"
    shutil.move(str(stitched_image_path), str(target_path))

    # 将最终结果打包成ZIP
    output_zip_path = session_dir / "stitched_result.zip"
    shutil.make_archive(str(output_zip_path.with_suffix('')), 'zip', batch_output_dir)

    return FileResponse(
        path=output_zip_path,
        filename=f"{output_filename_base}_stitched.zip",
        media_type='application/zip'
    )
