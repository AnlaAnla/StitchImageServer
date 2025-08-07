import time

import requests
import os
import shutil
import zipfile
from PIL import Image, ImageDraw

# --- 配置 ---
# 请确保你的 FastAPI 服务器正在运行，并修改此处的地址和端口
BASE_URL = "http://127.0.0.1:7745/api"  # 假设您的API前缀是 /api, 如果不是，请修改
STITCH_API_PREFIX = "/stitch"
SINGLE_FOLDER_URL = f"{BASE_URL}{STITCH_API_PREFIX}/from-folder"
BATCH_FOLDER_URL = f"{BASE_URL}{STITCH_API_PREFIX}/batch/from-folder"

# 用于存放自动生成的测试图片的临时目录
TEST_DATA_DIR = "temp_api_test_data"


# --- 测试函数 1：新的单个拼图接口 (从文件夹上传) ---
def single_puzzle_from_folder_api(image_folder_path: str):
    """
    测试 /stitch/from-folder 接口 (单个拼图)
    """
    print(f"--- 1. 开始测试: 单个拼图接口 (从文件夹上传) ---")
    print(f"使用文件夹: {image_folder_path}")

    # 1. 准备请求数据
    # 从文件夹路径推断输出文件名
    output_filename_base = os.path.basename(image_folder_path)
    form_data = {
        'output_filename_base': output_filename_base,
        'method': 'template_match',
        'num_cols': 4,
        'num_rows': 6,
        'overlap_h': 405,
        'overlap_v': 440,
        'tm_blend_type': 'half_importance_add_weight',
        'tm_light_compensation': True,
    }

    # 2. 准备要上传的文件列表
    # 'files' 是 FastAPI 接口中定义的参数名
    # requests 要求格式为: [('field_name', ('filename', file_object, 'content_type')), ...]
    files_to_send = []
    file_objects = []
    try:
        for filename in os.listdir(image_folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(image_folder_path, filename)
                f = open(file_path, 'rb')
                file_objects.append(f)
                files_to_send.append(('files', (filename, f, 'image/jpeg')))

        if not files_to_send:
            print("❌ 错误: 在文件夹中未找到可上传的图片。")
            return

        # 3. 发送 POST 请求
        print(f"向服务器 {SINGLE_FOLDER_URL} 发送 {len(files_to_send)} 个文件...")
        response = requests.post(SINGLE_FOLDER_URL, data=form_data, files=files_to_send, timeout=60)

        # 4. 处理响应
        print(f"服务器响应状态码: {response.status_code}")
        if response.status_code == 200:
            content_type = response.headers.get('content-type')
            print(f"响应内容类型: {content_type}")
            if 'image/jpeg' in content_type:
                output_filename = f"stitched_single_{output_filename_base}.jpg"
                with open(output_filename, "wb") as f:
                    f.write(response.content)
                print(f"✅ 成功! 拼接后的大图已保存为: {output_filename}")
            else:
                print(f"❌ 失败! 期望得到 'image/jpeg'，但收到了 '{content_type}'")
        else:
            print(f"❌ 请求失败! 错误信息: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ 请求异常! 无法连接到服务器: {e}")
    finally:
        # 确保所有打开的文件都被关闭
        for f in file_objects:
            f.close()

    print("-" * 50 + "\n")


# --- 测试函数 2：新的批量拼图接口 (从文件夹上传, ZIP返回) ---
def batch_puzzle_from_folder_api(image_folder_path: str):
    """
    测试 /stitch/batch/from-folder 接口 (批量拼图)
    """
    print(f"--- 2. 开始测试: 批量拼图接口 (单文件夹上传, ZIP返回) ---")
    print(f"使用文件夹: {image_folder_path}")

    # 1. 准备请求数据
    output_filename_base = os.path.basename(image_folder_path)
    form_data = {
        'output_filename_base': output_filename_base,
        'method': 'template_match',
        'num_cols': 4,
        'num_rows': 6,
        'overlap_h': 405,
        'overlap_v': 440,
    }

    # 2. 准备文件列表 (与单个接口的逻辑相同)
    files_to_send = []
    file_objects = []
    try:
        for filename in os.listdir(image_folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(image_folder_path, filename)
                f = open(file_path, 'rb')
                file_objects.append(f)
                files_to_send.append(('files', (filename, f, 'image/jpeg')))

        if not files_to_send:
            print("❌ 错误: 在文件夹中未找到可上传的图片。")
            return

        # 3. 发送 POST 请求
        print(f"向服务器 {BATCH_FOLDER_URL} 发送 {len(files_to_send)} 个文件...")
        response = requests.post(BATCH_FOLDER_URL, data=form_data, files=files_to_send, timeout=60)

        # 4. 处理响应
        print(f"服务器响应状态码: {response.status_code}")
        if response.status_code == 200:
            content_type = response.headers.get('content-type')
            print(f"响应内容类型: {content_type}")
            if 'application/zip' in content_type:
                output_filename = f"stitched_batch_{output_filename_base}.zip"
                with open(output_filename, "wb") as f:
                    f.write(response.content)
                print(f"✅ 成功! 包含拼接结果的ZIP包已保存为: {output_filename}")
                # (可选) 解压并检查结果
                try:
                    extract_dir = "batch_results_unzipped"
                    if os.path.exists(extract_dir):
                        shutil.rmtree(extract_dir)
                    os.makedirs(extract_dir, exist_ok=True)
                    with zipfile.ZipFile(output_filename, 'r') as zf:
                        zf.extractall(extract_dir)
                    print(f"  - 结果已自动解压到 '{extract_dir}' 文件夹，包含文件: {os.listdir(extract_dir)}")
                except Exception as e:
                    print(f"  - 解压返回的ZIP文件时出错: {e}")
            else:
                print(f"❌ 失败! 期望得到 'application/zip'，但收到了 '{content_type}'")
        else:
            print(f"❌ 请求失败! 错误信息: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ 请求异常! 无法连接到服务器: {e}")
    finally:
        for f in file_objects:
            f.close()

    print("-" * 50 + "\n")


if __name__ == "__main__":
    t1 = time.time()
    single_puzzle_from_folder_api(r"C:\Code\ML\Project\StitchImageServer\temp\Input\_250801_1043_0001")
    # batch_puzzle_from_folder_api(test_folder_2)
    t2 = time.time()
    print("cost: ", t2 - t1)
