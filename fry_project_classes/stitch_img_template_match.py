import math
from pathlib import Path

import cv2
import numpy as np
import os
import time


def fry_cv2_imread(filename, flags=cv2.IMREAD_COLOR):
    try:
        with open(filename, 'rb') as f:
            chunk = f.read()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        img = cv2.imdecode(chunk_arr, flags)
        if img is None:
            error_info = f"Warning: Unable to decode image: {filename}"
            print("警告", error_info)
        return img
    except IOError as e:
        error_info = f"IOError: Unable to read file: {filename}"
        print("错误", error_info)
        print("错误", f"Error details: {str(e)}")

        return None


def fry_cv2_imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1].lower()
        result, encoded_img = cv2.imencode(ext, img, params)

        if result:
            with open(filename, 'wb') as f:
                encoded_img.tofile(f)
            return True
        else:
            print("警告", f"Warning: Unable to encode image: {filename}")
            return False
    except Exception as e:
        print("错误", f"Error: Unable to write file: {filename}")
        print("错误", f"Error details: {str(e)}")
        return False


# 覆盖 OpenCV 的原始函数
cv2.imread = fry_cv2_imread
cv2.imwrite = fry_cv2_imwrite

from fry_project_classes.blend_type_mixin import BlendTypeMixin


class ImageStitcherTemplateMatch(BlendTypeMixin):
    def __init__(self, estimate_overlap_pixels=800, center_ratio=0.8,
                 stitch_type="vertical",
                 blend_type='half_importance',
                 blend_ratio: float = 0.3,
                 debug=False, debug_dir='debug_output',
                 light_uniformity_compensation_enabled=False,
                 light_uniformity_compensation_width=15,
                 debug_draw_line_enabled=False
                 ):
        """
        初始化拼图器
        
        参数:
        overlap_pixels: 重叠区域像素数，默认800像素（预估值）
        center_ratio: 中心区域比例，默认0.8
        debug: 是否开启调试模式，默认False
        debug_dir: 调试图片保存目录，默认'debug_output'
        use_weight_blend: 是否使用加权融合，默认True
        """

        print("警告", f"拼图方法：区域")

        self.estimate_overlap_pixels = estimate_overlap_pixels
        self.estimate_non_overlap_pixels = None
        self.center_ratio = center_ratio
        self.blend_ratio = blend_ratio

        self.blend_type = blend_type
        self.stitch_type = stitch_type

        self.debug_draw_line_enabled = debug_draw_line_enabled

        self.debug = debug
        self.init_debug = debug_dir

        self.light_uniformity_compensation_enabled = light_uniformity_compensation_enabled
        self.light_uniformity_compensation_width = light_uniformity_compensation_width

        if self.debug:
            self.debug_dir = f"{self.init_debug}_{self.stitch_type}_{self.blend_type}"
            os.makedirs(self.debug_dir, exist_ok=True)
            os.makedirs(self.debug_dir, exist_ok=True)

        # 模板匹配参数
        self.best_y = -1
        self.best_x = -1
        self.best_score = -1
        self.template_score = -1

    def save_debug_image(self, img, name, normalize=False):
        """
        保存调试图片
        
        参数:
        img: 要保存的图片
        name: 图片名称
        normalize: 是否需要归一化处理（对于模板匹配结果图等）
        """
        try:
            if self.debug:
                save_path = os.path.join(self.debug_dir, f"{name}.jpg")
                if normalize:
                    # 归一化到0-255范围
                    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                    cv2.imwrite(save_path, img_normalized)
                else:
                    cv2.imwrite(save_path, img)

                now_info = f"Debug: Saved {save_path}"

                # print("信息", now_info)

                return True, f"save_debug_image 成功: {save_path}"
            else:
                return False, "debug mode is not enabled"
        except Exception as e:
            error_info = f"save_debug_image出现bug: {str(e)}"
            print("警告", error_info)

            return False, error_info

    def visualize_template_match(self, template, search_region, best_y, best_x):
        """
        可视化模板匹配结果
        
        参数:
        template: 模板图片
        search_region: 搜索区域
        best_y, best_x: 最佳匹配位置
        """
        try:
            if self.debug:
                vis_img = search_region.copy()
                h, w = template.shape[:2]
                cv2.rectangle(vis_img, (best_x, best_y),
                              (best_x + w, best_y + h), (0, 255, 0), 2)
                is_ok, msg = self.save_debug_image(vis_img, 'tp_140_template_match_visualization')
                return is_ok, msg
            else:
                return False, "debug mode is not enabled"
        except Exception as e:
            error_info = f"visualize_template_match 出现bug: {str(e)}"
            print("警告", error_info)

            return False, msg

    def split_image(self, img, is_left_top=True):
        """
        分割图片为重叠区域和非重叠区域
        """
        height, width = img.shape[:2]
        overlap_width = min(self.estimate_overlap_pixels, width // 2)
        non_overlap_width = width - overlap_width

        if is_left_top:
            non_overlap_region = img[:, :non_overlap_width]
            overlap_region = img[:, non_overlap_width:]
            if self.debug:
                self.save_debug_image(img, 'split_220_left_top_original')
                self.save_debug_image(non_overlap_region, 'split_240_left_top_non_overlap')
                self.save_debug_image(overlap_region, 'split_260_left_top_overlap')
        else:
            overlap_region = img[:, :overlap_width]
            non_overlap_region = img[:, overlap_width:]
            if self.debug:
                self.save_debug_image(img, 'split_320_right_bottom_original')
                self.save_debug_image(overlap_region, 'split_340_right_bottom_overlap')
                self.save_debug_image(non_overlap_region, 'split_360_right_bottom_non_overlap')

        return overlap_region, non_overlap_region

    def get_center_region(self, img):
        """
        获取图片的中心区域
        """
        height, width = img.shape[:2]
        margin_y = int(height * (1 - self.center_ratio) / 2)
        margin_x = int(width * (1 - self.center_ratio) / 2)

        center_region = img[margin_y:height - margin_y, margin_x:width - margin_x]

        if self.debug:
            self.save_debug_image(center_region, 'center_120_template_center_region')

        return center_region, margin_x, margin_y

    def template_matching(self, template, search_region):
        """
        模板匹配
        """
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        best_y, best_x = max_loc[1], max_loc[0]

        now_info = f"匹配结果: 最大值{max_val}, 位置{best_x, best_y}"

        print("警告", now_info)

        self.best_y = best_y
        self.best_x = best_x
        self.best_score = max_val
        self.template_score = max_val

        # 如果存在父对象且父对象有 update_match_score 方法，发送匹配分数
        # if hasattr(self, 'parent') and hasattr(self.parent, 'update_match_score'):
        #     self.parent.update_match_score(max_val)

        if self.debug:
            is_ok1, opt_msg1 = self.save_debug_image(result, 'tp_120_template_matching_result', normalize=True)
            is_ok2, opt_msg2 = self.visualize_template_match(template, search_region, best_y, best_x)
            is_ok3, opt_msg3 = self.save_debug_image(template, 'tp_220_template')
            is_ok4, opt_msg4 = self.save_debug_image(search_region, 'tp_240_search_region')

            with open(os.path.join(self.debug_dir, 'tp_320_matching_info.txt'), 'w') as f:
                f.write(f"Best match position: ({best_x}, {best_y})\n")
                f.write(f"Match score: {max_val}\n")

        return best_x, best_y, max_val

    def pad_image(self, img: np.ndarray, target_width: int = None, target_height: int = None) -> np.ndarray:
        """
        将图片填充到目标尺寸

        参数:
        img: 输入图片
        target_width: 目标宽度，如果为None则保持原宽度
        target_height: 目标高度，如果为None则保持原高度

        返回:
        填充后的图片
        """
        if self.debug:
            self.save_debug_image(img, 'pad_120_before_padding')

        current_height, current_width = img.shape[:2]
        target_width = target_width if target_width is not None else current_width
        target_height = target_height if target_height is not None else current_height

        if current_width == target_width and current_height == target_height:
            return img

        # 创建黑色背景
        padded_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        # 将原图放在中心位置
        y_offset = (target_height - current_height) // 2
        x_offset = (target_width - current_width) // 2

        padded_img[y_offset:y_offset + current_height,
        x_offset:x_offset + current_width] = img

        if self.debug:
            self.save_debug_image(padded_img, 'pad_220_after_padding')
            with open(os.path.join(self.debug_dir, 'pad_320_padding_info.txt'), 'w') as f:
                f.write(f"Original size: {current_width}x{current_height}\n")
                f.write(f"Target size: {target_width}x{target_height}\n")
                f.write(f"Padding offsets: x={x_offset}, y={y_offset}\n")

        return padded_img

    def stitch_horizontal(self, left_img, right_img):
        """
        水平拼接两张图片
        水平拼接的图片的高必须一样
        """
        # self.debug_dir = f"{self.init_debug}_horizontal_{self.blend_type}"
        # os.makedirs(self.debug_dir, exist_ok=True)

        # 确保两张图片高度相同
        max_height = max(left_img.shape[0], right_img.shape[0])
        left_img = self.pad_image(left_img, target_height=max_height)
        right_img = self.pad_image(right_img, target_height=max_height)

        # 1. 分割图片，使用预估的重叠像素
        left_overlap, left_non_overlap = self.split_image(left_img, is_left_top=True)
        right_overlap, right_non_overlap = self.split_image(right_img, is_left_top=False)

        # 2. 获取左图重叠区域的中心部分作为模板
        template, template_offset_x, template_offset_y = self.get_center_region(left_overlap)

        # 计算模板在左图中的位置
        self.estimate_non_overlap_pixels = left_img.shape[1] - self.estimate_overlap_pixels
        template_in_left_x = self.estimate_non_overlap_pixels + template_offset_x
        template_in_left_y = template_offset_y

        # 3. 在右图重叠区域中进行模板匹配
        best_x, best_y, max_val = self.template_matching(template, right_overlap)
        template_score = max_val

        # 计算模板在右图中的位置
        template_in_right_x = best_x
        template_in_right_y = best_y

        # 真实的重叠区域的x和y 这个还不好算

        # 04、计算最后拼接的图片的尺寸
        left_width_contribution = template_in_left_x  # 左图取到模板的左上角，不包含模板
        right_width_contribution = right_img.shape[1] - template_in_right_x  # 右图取到模板的左上角，包含模板
        stitch_img_width = left_width_contribution + right_width_contribution
        stitch_img_height = max(left_img.shape[0], right_img.shape[0])

        # 计算右图相对于左图的y方向的偏移量
        y_offset_right2left = template_in_left_y - template_in_right_y

        # 计算真正的重叠区域：右图的左边+模板宽度+左图的右边
        real_overlap_width = template_in_right_x + template.shape[1] + (
                    left_img.shape[1] - template_in_left_x - template.shape[1])

        if self.debug:
            with open(os.path.join(self.debug_dir, 'h_320_alignment_info.txt'), 'w') as f:
                f.write(f"template_in_left_x: {template_in_left_x}\n")
                f.write(f"template_in_left_y: {template_in_left_y}\n")
                f.write(f"template_in_right_x: {template_in_right_x}\n")
                f.write(f"template_in_right_y: {template_in_right_y}\n")
                f.write(f"left_width_contribution: {left_width_contribution}\n")
                f.write(f"right_width_contribution: {right_width_contribution}\n")
                f.write(f"stitch_img_width: {stitch_img_width}\n")
                f.write(f"stitch_img_height: {stitch_img_height}\n")
                f.write(f"real_overlap_width: {real_overlap_width}\n")

        if self.blend_type == 'half_importance':
            blend_stitch_img = self.blend_half_importance(left_img, right_img, stitch_img_width, stitch_img_height,
                                                          y_offset_right2left, real_overlap_width,
                                                          light_uniformity_compensation=self.light_uniformity_compensation_enabled,
                                                          light_uniformity_compensation_width=self.light_uniformity_compensation_width)
            if self.debug_draw_line_enabled:
                blend_stitch_img_visualize = self.blend_half_importance(left_img, right_img, stitch_img_width,
                                                                        stitch_img_height, y_offset_right2left,
                                                                        real_overlap_width, visualize=True)
                if self.debug:
                    self.save_debug_image(blend_stitch_img_visualize, 'h_500_final_result_horizontal_visualize')
        elif self.blend_type == 'right_first':
            # 右边优先的拼接方式
            blend_stitch_img = self.blend_right_first(left_img, right_img, stitch_img_width, stitch_img_height,
                                                      y_offset_right2left)
        elif self.blend_type == 'left_first':
            blend_stitch_img = self.blend_left_first(left_img, right_img,
                                                     stitch_img_width,
                                                     stitch_img_height,
                                                     y_offset_right2left,
                                                     real_overlap_width)
        elif self.blend_type == 'half_importance_add_weight':
            blend_stitch_img = self.blend_half_importance_add_weight(left_img, right_img, stitch_img_width,
                                                                     stitch_img_height, y_offset_right2left,
                                                                     real_overlap_width,
                                                                     blend_ratio=self.blend_ratio)
        elif self.blend_type == 'half_importance_global_brightness':
            blend_stitch_img = self.blend_half_importance_global_brightness(left_img, right_img, stitch_img_width,
                                                                            stitch_img_height, y_offset_right2left,
                                                                            real_overlap_width,
                                                                            light_uniformity_compensation=self.light_uniformity_compensation_enabled,
                                                                            light_uniformity_compensation_width=self.light_uniformity_compensation_width)
            if self.debug_draw_line_enabled:
                blend_stitch_img_visualize = self.blend_half_importance_global_brightness(left_img, right_img,
                                                                                          stitch_img_width,
                                                                                          stitch_img_height,
                                                                                          y_offset_right2left,
                                                                                          real_overlap_width,
                                                                                          visualize=True)
                if self.debug:
                    self.save_debug_image(blend_stitch_img_visualize, 'h_500_final_result_horizontal_visualize')

        elif self.blend_type == 'half_importance_partial_brightness':
            blend_stitch_img = self.blend_half_importance_partial_brightness(left_img, right_img, stitch_img_width,
                                                                             stitch_img_height, y_offset_right2left,
                                                                             real_overlap_width,
                                                                             light_uniformity_compensation=self.light_uniformity_compensation_enabled,
                                                                             light_uniformity_compensation_width=self.light_uniformity_compensation_width)
            if self.debug_draw_line_enabled:
                blend_stitch_img_visualize = self.blend_half_importance_partial_brightness(left_img, right_img,
                                                                                           stitch_img_width,
                                                                                           stitch_img_height,
                                                                                           y_offset_right2left,
                                                                                           real_overlap_width,
                                                                                           visualize=True)
                if self.debug:
                    self.save_debug_image(blend_stitch_img_visualize, 'h_500_final_result_horizontal_visualize')


        elif self.blend_type == 'blend_half_importance_partial_HV':
            blend_stitch_img = self.blend_half_importance_partial_HV(left_img, right_img, stitch_img_width,
                                                                     stitch_img_height, y_offset_right2left,
                                                                     real_overlap_width,
                                                                     light_uniformity_compensation=self.light_uniformity_compensation_enabled,
                                                                     light_uniformity_compensation_width=self.light_uniformity_compensation_width)
            if self.debug_draw_line_enabled:
                blend_stitch_img_visualize = self.blend_half_importance_partial_HV(left_img, right_img,
                                                                                   stitch_img_width, stitch_img_height,
                                                                                   y_offset_right2left,
                                                                                   real_overlap_width, visualize=True)
                if self.debug:
                    self.save_debug_image(blend_stitch_img_visualize, 'h_500_final_result_horizontal_visualize')

        elif self.blend_type == 'blend_half_importance_partial_SV':
            blend_stitch_img = self.blend_half_importance_partial_SV(left_img, right_img, stitch_img_width,
                                                                     stitch_img_height, y_offset_right2left,
                                                                     real_overlap_width,
                                                                     light_uniformity_compensation=self.light_uniformity_compensation_enabled,
                                                                     light_uniformity_compensation_width=self.light_uniformity_compensation_width)
            if self.debug_draw_line_enabled:
                blend_stitch_img_visualize = self.blend_half_importance_partial_SV(left_img, right_img,
                                                                                   stitch_img_width, stitch_img_height,
                                                                                   y_offset_right2left,
                                                                                   real_overlap_width, visualize=True)
                if self.debug:
                    self.save_debug_image(blend_stitch_img_visualize, 'h_500_final_result_horizontal_visualize')

        elif self.blend_type == 'blend_half_importance_partial_HSV':
            blend_stitch_img = self.blend_half_importance_partial_HSV(left_img, right_img, stitch_img_width,
                                                                      stitch_img_height, y_offset_right2left,
                                                                      real_overlap_width,
                                                                      light_uniformity_compensation=self.light_uniformity_compensation_enabled,
                                                                      light_uniformity_compensation_width=self.light_uniformity_compensation_width)
            if self.debug_draw_line_enabled:
                blend_stitch_img_visualize = self.blend_half_importance_partial_HSV(left_img, right_img,
                                                                                    stitch_img_width, stitch_img_height,
                                                                                    y_offset_right2left,
                                                                                    real_overlap_width, visualize=True)
                if self.debug:
                    self.save_debug_image(blend_stitch_img_visualize, 'h_500_final_result_horizontal_visualize')



        elif self.blend_type == 'blend_half_importance_partial_brightness_add_weight':
            blend_stitch_img = self.blend_half_importance_partial_brightness_add_weight(
                left_img=left_img,
                right_img=right_img,
                stitch_img_width=stitch_img_width,
                stitch_img_height=stitch_img_height,
                y_offset_right2left=y_offset_right2left,
                real_overlap_width=real_overlap_width,
                light_uniformity_compensation=self.light_uniformity_compensation_enabled,
                light_uniformity_compensation_width=self.light_uniformity_compensation_width,
                add_weight_rate=self.blend_ratio
            )
            if self.debug_draw_line_enabled:
                blend_stitch_img_visualize = self.blend_half_importance_partial_brightness_add_weight(left_img=left_img,
                                                                                                      right_img=right_img,
                                                                                                      stitch_img_width=stitch_img_width,
                                                                                                      stitch_img_height=stitch_img_height,
                                                                                                      y_offset_right2left=y_offset_right2left,
                                                                                                      real_overlap_width=real_overlap_width,
                                                                                                      add_weight_rate=self.blend_ratio,
                                                                                                      visualize=True)
                if self.debug:
                    self.save_debug_image(blend_stitch_img_visualize, 'h_500_final_result_horizontal_visualize')


        else:
            # 左边优先的拼接方式
            blend_stitch_img = None

        if self.debug:
            self.save_debug_image(blend_stitch_img, 'h_500_final_result_horizontal')

        # return result
        return blend_stitch_img

    def stitch_vertical(self, top_img, bottom_img):
        """
        垂直拼接两张图片
        垂直拼接的图片的宽必须一样
        """
        # self.debug_dir = f"{self.init_debug}_vertical_{self.blend_type}"
        # os.makedirs(self.debug_dir, exist_ok=True)

        # 确保两张图片宽度相同
        max_width = max(top_img.shape[1], bottom_img.shape[1])
        top_img = self.pad_image(top_img, target_width=max_width)
        bottom_img = self.pad_image(bottom_img, target_width=max_width)

        if self.debug:
            self.save_debug_image(top_img, 'v_120_top_original')
            self.save_debug_image(bottom_img, 'v_140_bottom_original')

        # 将图片逆时针旋转90度
        top_rotated = cv2.rotate(top_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        bottom_rotated = cv2.rotate(bottom_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.debug:
            self.save_debug_image(top_rotated, 'v_220_top_rotated')
            self.save_debug_image(bottom_rotated, 'v_240_bottom_rotated')

        # 进行水平拼接
        result_rotated = self.stitch_horizontal(top_rotated, bottom_rotated)

        # 将结果顺时针旋转90度
        result = cv2.rotate(result_rotated, cv2.ROTATE_90_CLOCKWISE)

        if self.debug:
            self.save_debug_image(result, 'v_500_final_result_vertical')

        return result

    def stitch_main(self, left_img, right_img):
        if self.stitch_type == 'horizontal':
            # 确保两张图片高度相同
            max_height = max(left_img.shape[0], right_img.shape[0])
            left_img = self.pad_image(left_img, target_height=max_height)
            right_img = self.pad_image(right_img, target_height=max_height)

            return self.stitch_horizontal(left_img, right_img)
        elif self.stitch_type == 'vertical':
            # 确保两张图片宽度相同
            max_width = max(left_img.shape[1], right_img.shape[1])
            left_img = self.pad_image(left_img, target_width=max_width)
            right_img = self.pad_image(right_img, target_width=max_width)

            return self.stitch_vertical(left_img, right_img)
        else:
            raise ValueError('Invalid stitch type, must be one of horizontal or vertical')


def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root_path = r"D:\_241231_fry_gitlab\LA_ai_main_CV_OpenCV\740_project\_250115_Stitch_Image_TemplateMatch\test_images"
    root_path_obj = Path(root_path).absolute()

    stitch_type = "vertical"
    debug_dir_str = str(root_path_obj / f'debug_{timestamp}_{stitch_type}')
    debug_dir_obj = Path(debug_dir_str).absolute()
    estimate_overlap_ratio = 0.45
    estimate_overlap_pixels = int(round(1024 * estimate_overlap_ratio))

    # 使用加权融合
    # stitcher_weight = ImageStitcher(
    #     estimate_overlap_pixels=estimate_overlap_pixels, 
    #     center_ratio=0.8, 
    #     debug=True,
    #     debug_dir=debug_dir+"_weight",
    #     use_weight_blend=True
    # )

    # 读取图片
    left_img_name = r"20250123_162407_0001.jpg"
    right_img_name = r"20250123_162409_0002.jpg"
    bottom_img_name = r"20250123_162422_0007.jpg"

    left_img_path = str(root_path_obj / left_img_name)
    right_img_path = str(root_path_obj / right_img_name)
    bottom_img_path = str(root_path_obj / bottom_img_name)

    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    top_img = left_img
    bottom_img = cv2.imread(bottom_img_path)

    start_time = time.time()

    # 使用简单拼接
    stitcher_simple = ImageStitcherTemplateMatch(
        estimate_overlap_pixels=estimate_overlap_pixels,
        center_ratio=0.8,
        blend_type='half_importance_add_weight',
        # blend_type='left_first',
        stitch_type=stitch_type,
        blend_ratio=0.5,
        debug=True,
        debug_dir=debug_dir_str,
    )

    # 竖直拼接 - 简单拼接
    result_img = stitcher_simple.stitch_main(left_img, bottom_img)
    save_final_image_path = str(debug_dir_obj / 'result_img.jpg')
    cv2.imwrite(save_final_image_path, result_img)

    end_time = time.time()

    print(f"拼接图片耗时：{end_time - start_time:.2f}秒")


if __name__ == '__main__':
    main()
