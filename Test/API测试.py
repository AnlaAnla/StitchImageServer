import requests
import os
import io
import zipfile
from PIL import Image, ImageDraw

# --- 配置 ---
# 请确保你的 FastAPI 服务器正在运行，并修改此处的地址和端口
BASE_URL = "http://127.0.0.1:7745/api"
SINGLE_STITCH_URL = f"{BASE_URL}/stitch"


# --- 测试函数 1：单个拼图接口 ---

def single_puzzle_api(zipfile_path):
    """
    测试 /stitch/ 接口 (单个拼图)
    使用 模板匹配法 (template_match)
    """
    print("--- 1. 开始测试: 单个拼图接口 (/stitch) ---")

    # 1. 准备请求数据
    form_data = {
        'method': 'template_match',
        'num_cols': 4,
        'num_rows': 6,
        'overlap_h': 405,
        'overlap_v': 440,
        'tm_blend_type': 'half_importance_add_weight',
        'tm_light_compensation': True,
    }

    try:
        # 打开文件
        with open(zipfile_path, "rb") as f:
            files = {
                'zip_file': (os.path.basename(zipfile_path), f, 'application/zip')
            }
            # 4. 发送 POST 请求
            print("向服务器发送请求...")
            response = requests.post(SINGLE_STITCH_URL, data=form_data, files=files)  # 设置较长超时

            # 5. 处理响应
            print(f"服务器响应状态码: {response.status_code}")

        if response.status_code == 200:
            # 检查响应头，确认是图片
            content_type = response.headers.get('content-type')
            print(f"响应内容类型: {content_type}")

            if 'image/jpeg' in content_type:
                # 将返回的图片内容保存到本地文件
                output_filename = "stitched_single_result.jpg"
                with open(output_filename, "wb") as f:
                    f.write(response.content)
                print(f"✅ 成功! 拼接后的大图已保存为: {output_filename}")
            else:
                print(f"❌ 失败! 期望得到 'image/jpeg'，但收到了 '{content_type}'")
        else:
            # 如果接口返回错误，打印错误详情
            print(f"❌ 请求失败! 错误信息: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ 请求异常! 无法连接到服务器: {e}")

    print("-" * 40 + "\n")


# --- 测试函数 2：批量拼图接口 ---


if __name__ == "__main__":
    # 依次运行两个测试函数
    single_puzzle_api(r"C:\Code\ML\Project\StitchImageServer\temp\Input\front_0_1.zip")
