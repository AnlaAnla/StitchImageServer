import cv2
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Union, Dict
import re


class FryImageWrite:
    """
    图片创建和文字添加类
    支持英文和中文文字添加，以及基本的图形绘制
    """

    def __init__(self, width: int = 1920, height: int = 1080,
                 background_color: Tuple[int, int, int] = (255, 255, 255)):
        """
        初始化图片画布
        
        参数:
        width: 图片宽度
        height: 图片高度
        background_color: 背景颜色，BGR格式
        """
        self.width = width
        self.height = height
        self.image = np.full((height, width, 3), background_color, dtype=np.uint8)

    def add_text(self, text: str, position: Tuple[int, int],
                 font_size: int = 32, color: Tuple[int, int, int] = (0, 0, 0),
                 font_path: str = "simhei.ttf", thickness: int = 2) -> None:
        """
        统一的文字添加方法，自动判断中英文并对齐文字基线
        
        参数:
        text: 文本内容
        position: 文本位置 (x, y)
        font_size: 字体大小
        color: 字体颜色 (B, G, R)
        font_path: 中文字体文件路径
        thickness: 英文字体粗细
        """
        x, y = position

        if self.is_chinese(text):
            # 中文文字处理
            try:
                # 创建临时PIL Image来计算文字尺寸
                font = ImageFont.truetype(font_path, font_size)
                # 获取文字的尺寸信息
                bbox = font.getbbox(text)
                # 调整y坐标，使文字基线对齐
                adjusted_y = y + bbox[3] - font_size
                self.add_text_cn(text, (x, y), font_size, color, font_path)
            except Exception as e:
                print(f"字体加载失败: {e}")
                self.add_text_cn(text, (x, y), font_size, color, font_path)
        else:
            # 英文文字处理
            font_scale = font_size / 32
            # OpenCV文字基准点在左下角，需要向上偏移一个字体高度
            # 估算字体高度并调整y坐标
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                        font_scale, thickness)[0]
            adjusted_y = y + text_size[1]  # 加上文字高度使基线对齐
            self.add_text_en(text, (x, adjusted_y), font_scale, color, thickness)

    def add_text_cn(self, text: str, position: Tuple[int, int],
                    font_size: int = 32, color: Tuple[int, int, int] = (0, 0, 0),
                    font_path: str = "simhei.ttf") -> None:
        """
        添加中文文本
        
        参数:
        text: 文本内容
        position: 文本位置 (x, y)
        font_size: 字体大小
        color: 字体颜色 (B, G, R)
        font_path: 字体文件路径
        """
        img_pil = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"字体加载失败: {e}")
            font = ImageFont.load_default()

        color_rgb = (color[2], color[1], color[0])

        # 获取文字的bbox
        bbox = font.getbbox(text)
        # 计算文字的垂直中心位置
        text_height = bbox[3] - bbox[1]
        y_offset = text_height // 2
        x, y = position
        adjusted_position = (x, y - y_offset)

        draw.text(adjusted_position, text, font=font, fill=color_rgb)

        self.image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def add_text_en(self, text: str, position: Tuple[int, int],
                    font_scale: float = 1.0, color: Tuple[int, int, int] = (0, 0, 0),
                    thickness: int = 2, font_face: int = cv2.FONT_HERSHEY_SIMPLEX) -> None:
        """
        添加英文文本
        """
        # 获取文字大小
        text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
        x, y = position

        # 确保x和y是整数
        x = int(x)
        y = int(y)

        # 调整y坐标，使文字垂直居中
        adjusted_y = y - text_size[1] // 2

        # 确保adjusted_y也是整数
        adjusted_y = int(adjusted_y)

        # 使用整数坐标调用putText
        cv2.putText(self.image, text, (x, adjusted_y), font_face, font_scale,
                    color, thickness, cv2.LINE_AA)

    def is_chinese(self, text: str) -> bool:
        """
        判断字符串是否包含中文
        
        参数:
        text: 需要判断的文本
        
        返回:
        bool: 是否包含中文
        """
        pattern = re.compile(r'[\u4e00-\u9fff]')
        return bool(pattern.search(text))

    def add_dict_info(self, info_dict: Dict[str, str],
                      start_position: Tuple[int, int] = (50, 50),
                      line_spacing: int = 30,
                      label_value_spacing: int = 150,
                      font_size: int = 24,
                      label_color: Tuple[int, int, int] = (0, 0, 0),
                      value_color_map: Dict[str, Tuple[int, int, int]] = None) -> None:
        """
        添加字典信息到图片
        
        参数:
        info_dict: 信息字典，键为标签，值为内容
        start_position: 起始位置 (x, y)
        line_spacing: 行间距
        label_value_spacing: 标签和值之间的间距
        font_size: 字体大小
        label_color: 标签颜色
        value_color_map: 值的颜色映射字典，比如 {"PASS": (0, 255, 0), "FAIL": (0, 0, 255)}
        """
        if value_color_map is None:
            value_color_map = {}

        x, y = start_position

        for idx, (key, value) in enumerate(info_dict.items()):
            current_y = y + idx * line_spacing

            # 添加标签
            self.add_text(f"{key}:", (x, current_y), font_size, label_color)

            # 确定值的颜色
            value_color = value_color_map.get(str(value), label_color)

            # 添加值
            self.add_text(str(value), (x + label_value_spacing, current_y),
                          font_size, value_color)

    def add_rectangle(self, start_point: Tuple[int, int],
                      end_point: Tuple[int, int],
                      color: Tuple[int, int, int] = (0, 0, 0),
                      thickness: int = 2) -> None:
        """
        添加矩形
        
        参数:
        start_point: 起始点 (x, y)
        end_point: 结束点 (x, y)
        color: 颜色 (B, G, R)
        thickness: 线条粗细，-1表示填充
        """
        cv2.rectangle(self.image, start_point, end_point, color, thickness)

    def add_line(self, start_point: Tuple[int, int],
                 end_point: Tuple[int, int],
                 color: Tuple[int, int, int] = (0, 0, 0),
                 thickness: int = 2) -> None:
        """
        添加直线
        
        参数:
        start_point: 起始点 (x, y)
        end_point: 结束点 (x, y)
        color: 颜色 (B, G, R)
        thickness: 线条粗细
        """
        cv2.line(self.image, start_point, end_point, color, thickness)

    def save_image(self, file_path: str) -> bool:
        """
        保存图片
        
        参数:
        file_path: 保存路径
        
        返回:
        bool: 是否保存成功
        """
        try:
            cv2.imwrite(file_path, self.image)
            return True
        except Exception as e:
            print(f"保存图片失败: {e}")
            return False

    def get_image(self) -> np.ndarray:
        """
        获取图片数组
        
        返回:
        numpy.ndarray: 图片数组
        """
        return self.image.copy()

    def create_report(self, title: str, info_dict: Dict[str, str],
                      value_color_map: Dict[str, Tuple[int, int, int]] = None) -> None:
        """
        创建一个标准格式的报告
        
        参数:
        title: 报告标题
        info_dict: 信息字典，键为标签，值为内容
        value_color_map: 值的颜色映射字典，比如 {"PASS": (0, 255, 0), "FAIL": (0, 0, 255)}
        """
        # 设置默认的颜色映射
        if value_color_map is None:
            value_color_map = {
                "PASS": (0, 255, 0),  # 绿色
                "FAIL": (0, 0, 255),  # 红色
                "Good": (0, 255, 0),  # 绿色
                "Bad": (0, 0, 255)  # 红色
            }

        # 计算边距和间距
        margin = min(self.width, self.height) // 20  # 动态边距
        content_width = self.width - 2 * margin
        content_height = self.height - 2 * margin

        # 计算标题大小和位置
        title_font_size = min(content_width // len(title) if len(title) > 0 else content_width,
                              content_height // 8)
        title_font_size = min(72, max(36, title_font_size))  # 限制标题字体大小范围

        # 标题位置居中
        title_x = self.width // 2
        title_y = margin + title_font_size

        # 获取标题的大小来调整内容区域
        if self.is_chinese(title):
            font = ImageFont.truetype("simhei.ttf", title_font_size)
            title_width = font.getbbox(title)[2]
        else:
            title_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX,
                                          title_font_size / 32, 2)[0][0]

        # 添加标题（居中）
        self.add_text(title, (title_x - title_width // 2, title_y),
                      title_font_size, (0, 0, 0))

        # 计算内容区域的位置和大小
        content_start_y = title_y + title_font_size + margin
        content_area_height = self.height - content_start_y - margin

        # 计算内容的字体大小和间距
        item_count = len(info_dict)
        font_size = min(32, max(18, int(content_area_height / (item_count * 2))))
        line_spacing = max(font_size * 1.8, min(content_area_height / item_count, 50))

        # 计算最长标签的宽度来决定label_value_spacing
        max_label_len = max(len(str(key)) for key in info_dict.keys())
        label_value_spacing = max(150, max_label_len * font_size // 2)

        # 添加边框
        border_margin = margin // 2
        self.add_rectangle(
            (border_margin, border_margin),
            (self.width - border_margin, self.height - border_margin),
            (0, 0, 0), 2
        )

        # 添加内容区域的分隔线
        self.add_line(
            (border_margin, title_y + title_font_size // 2 + margin // 2),
            (self.width - border_margin, title_y + title_font_size // 2 + margin // 2),
            (0, 0, 0), 1
        )

        # 添加信息字典
        self.add_dict_info(
            info_dict,
            start_position=(margin * 1.5, content_start_y),
            line_spacing=int(line_spacing),
            label_value_spacing=label_value_spacing,
            font_size=font_size,
            value_color_map=value_color_map
        )


# 测试代码
def test_fry_image_write():
    """
    测试FryImageWrite类的各项功能
    """
    # 创建实例
    image_writer = FryImageWrite(800, 600)

    # 测试信息字典
    info_dict = {
        "产品型号": "ABC-123",
        "序列号": "SN20240101001",
        "检测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "测试结果": "PASS",
        "Product": "Camera",
        "Status": "FAIL",
        "温度": "25摄氏度",
        "Quality": "Good"
    }

    # 定义值的颜色映射
    value_color_map = {
        "PASS": (0, 255, 0),  # 绿色
        "FAIL": (0, 0, 255),  # 红色
        "Good": (0, 255, 0)  # 绿色
    }

    # 添加标题
    image_writer.add_text("测试报告", (300, 30), 48, (0, 0, 0))
    # 添加边框
    image_writer.add_rectangle((30, 80), (770, 580), (0, 0, 0), 2)

    # 添加信息字典
    image_writer.add_dict_info(
        info_dict,
        start_position=(50, 100),
        line_spacing=40,
        label_value_spacing=200,
        font_size=28,
        value_color_map=value_color_map
    )

    # 保存图片
    image_writer.save_image("test_output.jpg")


def test_fry_image_write2():
    """
    测试FryImageWrite类的各项功能
    """
    # 创建实例
    image_writer = FryImageWrite(1280, 800)  # 使用更大的尺寸以获得更好的效果

    # 测试信息字典
    info_dict = {
        "产品型号": "ABC-123",
        "序列号": "SN20240101001",
        "检测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "测试结果": "PASS",
        "Product": "Camera",
        "Status": "FAIL",
        "温度": "25摄氏度",
        "Quality": "Good"
    }

    # 一行代码创建完整报告
    image_writer.create_report("测试报告", info_dict)

    # 保存图片
    image_writer.save_image("test_output.jpg")


# 运行测试
if __name__ == "__main__":
    test_fry_image_write2()
