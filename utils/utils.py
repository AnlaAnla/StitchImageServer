import re
from pathlib import Path
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def natural_sort_key(s):
    """
    提供自然排序的键，例如 '2.jpg' 会排在 '10.jpg' 之前。
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def cleanup_temp_folder(folder_path: Path):
    """在后台删除指定的临时文件夹"""
    try:
        if folder_path.exists() and folder_path.is_dir():
            shutil.rmtree(folder_path)
            logging.info(f"已清理临时文件夹: {folder_path}")
    except Exception as e:
        logging.error(f"清理临时文件夹 {folder_path} 时出错: {e}")
