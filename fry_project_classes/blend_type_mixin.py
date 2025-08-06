import cv2
import numpy as np
import os
from fry_project_classes.fry_image_write_V03_250401 import FryImageWrite


class BlendTypeMixin:
    def blend_right_first(self, left_img: np.ndarray, right_img: np.ndarray,
                          stitch_img_width: int, stitch_img_height: int,
                          y_offset_right2left: int):
        """"右边优先的拼接方式"""
        # self.debug_dir = f"{self.init_debug}_{self.stitch_type}_right_first"
        # os.makedirs(self.debug_dir, exist_ok=True)
        # if self.debug:
        #     if data_center_algo_inner_signals_obj is not None:
        #         data_center_algo_inner_signals_obj.log_info_signal.emit("警告",f"混合模式：右图优先模式")

        # 05、实现简单拼接逻辑
        stitch_init_img = np.zeros((stitch_img_height, stitch_img_width, 3), dtype=np.uint8)
        # 左图直接放进去
        stitch_init_img[:left_img.shape[0], :left_img.shape[1]] = left_img
        # 先高后宽
        # 右图纠正位置之后放进去
        if y_offset_right2left > 0:
            stitch_init_img[y_offset_right2left:, stitch_img_width - right_img.shape[1]:] = \
                right_img[:right_img.shape[0] - y_offset_right2left, :]
        else:
            stitch_init_img[:right_img.shape[0] + y_offset_right2left, stitch_img_width - right_img.shape[1]:] = \
                right_img[abs(y_offset_right2left):, :]

        # return result
        return stitch_init_img

    def blend_left_first(self, left_img: np.ndarray, right_img: np.ndarray,
                         stitch_img_width: int, stitch_img_height: int,
                         y_offset_right2left: int, real_overlap_width: int):
        """"右边优先的拼接方式"""
        # self.debug_dir = f"{self.init_debug}_{self.stitch_type}_right_first"
        # os.makedirs(self.debug_dir, exist_ok=True)
        # if self.debug:
        #     if data_center_algo_inner_signals_obj is not None:
        #         data_center_algo_inner_signals_obj.log_info_signal.emit("警告",f"混合模式：左图优先模式")

        # 05、实现简单拼接逻辑
        stitch_init_img = np.zeros((stitch_img_height, stitch_img_width, 3), dtype=np.uint8)
        # 左图直接放进去
        stitch_init_img[:left_img.shape[0], :left_img.shape[1]] = left_img

        left_width = left_img.shape[1]
        # 先高后宽
        # 右图纠正位置之后放进去
        if y_offset_right2left > 0:
            stitch_init_img[y_offset_right2left:, left_width:] = \
                right_img[:right_img.shape[0] - y_offset_right2left, real_overlap_width:]
        else:
            stitch_init_img[:right_img.shape[0] + y_offset_right2left, left_width:] = \
                right_img[abs(y_offset_right2left):, real_overlap_width:]

        # return result
        return stitch_init_img

    def blend_half_importance(self, left_img: np.ndarray, right_img: np.ndarray,
                              stitch_img_width: int, stitch_img_height: int,
                              y_offset_right2left: int, real_overlap_width: int,
                              light_uniformity_compensation=False,
                              light_uniformity_compensation_width=15,
                              visualize=False):
        """"左右一样重要性的拼接方式"""
        # self.debug_dir = f"{self.init_debug}_{self.stitch_type}_half_importance"
        # os.makedirs(self.debug_dir, exist_ok=True)
        # if self.debug:
        #     if data_center_algo_inner_signals_obj is not None:
        #         data_center_algo_inner_signals_obj.log_info_signal.emit("警告",f"混合模式：左右半拼模式")

        real_overlap_width_half = int(real_overlap_width / 2)
        # 05、实现简单拼接逻辑
        stitch_init_img = np.zeros((stitch_img_height, stitch_img_width, 3), dtype=np.uint8)
        # 左图直接放进去
        stitch_init_img[:left_img.shape[0], :left_img.shape[1]] = left_img

        # 右图纠正位置之后放进去
        right_img_start_x = stitch_img_width - right_img.shape[1] + real_overlap_width_half

        if y_offset_right2left > 0:
            # _250609_1448_ 右图向下放一点
            stitch_init_img[y_offset_right2left:, right_img_start_x:] = \
                right_img[:right_img.shape[0] - y_offset_right2left, real_overlap_width_half:]
        else:
            # _250609_1448_ 右图向上放一点
            stitch_init_img[:right_img.shape[0] + y_offset_right2left, right_img_start_x:] = \
                right_img[abs(y_offset_right2left):, real_overlap_width_half:]

        # 可视化操作部分
        if visualize:
            green = (0, 255, 0)
            red = (0, 0, 255)
            blue = (0, 0, 255)
            color3 = (255, 255, 0)
            cv2.rectangle(stitch_init_img, (0, 0), (left_img.shape[1], left_img.shape[0]), color=green)  # 左上图

            cv2.rectangle(stitch_init_img, (0, 0), (left_img.shape[1] - real_overlap_width, left_img.shape[0]),
                          color=color3)

            # 右下图
            if y_offset_right2left > 0:
                cv2.rectangle(stitch_init_img, (right_img_start_x, 0),
                              (stitch_init_img.shape[1], stitch_init_img.shape[0]), color=red)
                pass
            else:
                cv2.rectangle(stitch_init_img, (right_img_start_x, 0),
                              (stitch_init_img.shape[1], stitch_init_img.shape[0]), color=blue)
                pass

        # return result
        return stitch_init_img

    def blend_half_importance_partial_brightness(self, left_img: np.ndarray, right_img: np.ndarray,
                                                 stitch_img_width: int, stitch_img_height: int,
                                                 y_offset_right2left: int, real_overlap_width: int,
                                                 light_uniformity_compensation=False,
                                                 light_uniformity_compensation_width=15,
                                                 visualize=False):
        """"左右一样重要性的拼接方式"""

        real_overlap_width_half = int(real_overlap_width / 2)
        # 05、实现简单拼接逻辑
        stitch_init_img = np.zeros((stitch_img_height, stitch_img_width, 3), dtype=np.uint8)

        right_img_start_x = stitch_img_width - right_img.shape[1] + real_overlap_width_half

        # 如果进行图像补偿
        if light_uniformity_compensation:
            stitch_x_in_left = left_img.shape[1] - real_overlap_width_half
            stitch_x_in_right = real_overlap_width_half
            adjusted_left_img, adjusted_right_img = self.adjust_partial_brightness_for_stitching(left_img, right_img,
                                                                                                 real_overlap_width=real_overlap_width,
                                                                                                 y_offset_right2left=y_offset_right2left,
                                                                                                 block_size=light_uniformity_compensation_width,
                                                                                                 use_saturation_correct=False,
                                                                                                 use_hue_correct=False,
                                                                                                 weight_ratio=2.0
                                                                                                 )
            if self.debug:
                self.save_debug_image(adjusted_left_img, 'light_compensation_720_left_img_adjusted')
                self.save_debug_image(adjusted_right_img, 'light_compensation_720_right_img_adjusted')
            left_img = adjusted_left_img
            right_img = adjusted_right_img
            pass

        # 左图直接放进去
        stitch_init_img[:left_img.shape[0], :left_img.shape[1]] = left_img

        # 右图纠正位置之后放进去
        if y_offset_right2left > 0:
            stitch_init_img[y_offset_right2left:, right_img_start_x:] = \
                right_img[:right_img.shape[0] - y_offset_right2left, real_overlap_width_half:]
        else:
            stitch_init_img[:right_img.shape[0] + y_offset_right2left, right_img_start_x:] = \
                right_img[abs(y_offset_right2left):, real_overlap_width_half:]

        # 可视化操作部分
        if visualize:
            green = (0, 255, 0)
            red = (0, 0, 255)
            blue = (0, 0, 255)
            color3 = (255, 255, 0)
            cv2.rectangle(stitch_init_img, (0, 0), (left_img.shape[1], left_img.shape[0]), color=green)  # 左上图

            cv2.rectangle(stitch_init_img, (0, 0), (left_img.shape[1] - real_overlap_width, left_img.shape[0]),
                          color=color3)

            # 右下图
            if y_offset_right2left > 0:
                cv2.rectangle(stitch_init_img, (right_img_start_x, 0),
                              (stitch_init_img.shape[1], stitch_init_img.shape[0]), color=red)
                pass
            else:
                cv2.rectangle(stitch_init_img, (right_img_start_x, 0),
                              (stitch_init_img.shape[1], stitch_init_img.shape[0]), color=blue)
                pass

        # return result
        return stitch_init_img

    def blend_half_importance_partial_HV(self, left_img: np.ndarray, right_img: np.ndarray,
                                         stitch_img_width: int, stitch_img_height: int,
                                         y_offset_right2left: int, real_overlap_width: int,
                                         light_uniformity_compensation=False,
                                         light_uniformity_compensation_width=15,
                                         visualize=False):
        """"左右一样重要性的拼接方式"""

        real_overlap_width_half = int(real_overlap_width / 2)
        # 05、实现简单拼接逻辑
        stitch_init_img = np.zeros((stitch_img_height, stitch_img_width, 3), dtype=np.uint8)

        right_img_start_x = stitch_img_width - right_img.shape[1] + real_overlap_width_half

        # 如果进行图像补偿
        if light_uniformity_compensation:
            stitch_x_in_left = left_img.shape[1] - real_overlap_width_half
            stitch_x_in_right = real_overlap_width_half
            adjusted_left_img, adjusted_right_img = self.adjust_partial_brightness_for_stitching(left_img, right_img,
                                                                                                 real_overlap_width=real_overlap_width,
                                                                                                 y_offset_right2left=y_offset_right2left,
                                                                                                 block_size=light_uniformity_compensation_width,
                                                                                                 use_saturation_correct=False,
                                                                                                 use_hue_correct=True
                                                                                                 )
            if self.debug:
                self.save_debug_image(adjusted_left_img, 'light_compensation_720_left_img_adjusted')
                self.save_debug_image(adjusted_right_img, 'light_compensation_720_right_img_adjusted')
            left_img = adjusted_left_img
            right_img = adjusted_right_img
            pass

        # 左图直接放进去
        stitch_init_img[:left_img.shape[0], :left_img.shape[1]] = left_img

        # 右图纠正位置之后放进去
        if y_offset_right2left > 0:
            stitch_init_img[y_offset_right2left:, right_img_start_x:] = \
                right_img[:right_img.shape[0] - y_offset_right2left, real_overlap_width_half:]
        else:
            stitch_init_img[:right_img.shape[0] + y_offset_right2left, right_img_start_x:] = \
                right_img[abs(y_offset_right2left):, real_overlap_width_half:]

        # 可视化操作部分
        if visualize:
            green = (0, 255, 0)
            red = (0, 0, 255)
            blue = (0, 0, 255)
            color3 = (255, 255, 0)
            cv2.rectangle(stitch_init_img, (0, 0), (left_img.shape[1], left_img.shape[0]), color=green)  # 左上图

            cv2.rectangle(stitch_init_img, (0, 0), (left_img.shape[1] - real_overlap_width, left_img.shape[0]),
                          color=color3)

            # 右下图
            if y_offset_right2left > 0:
                cv2.rectangle(stitch_init_img, (right_img_start_x, 0),
                              (stitch_init_img.shape[1], stitch_init_img.shape[0]), color=red)
                pass
            else:
                cv2.rectangle(stitch_init_img, (right_img_start_x, 0),
                              (stitch_init_img.shape[1], stitch_init_img.shape[0]), color=blue)
                pass

        # return result
        return stitch_init_img

    def blend_half_importance_partial_SV(self, left_img: np.ndarray, right_img: np.ndarray,
                                         stitch_img_width: int, stitch_img_height: int,
                                         y_offset_right2left: int, real_overlap_width: int,
                                         light_uniformity_compensation=False,
                                         light_uniformity_compensation_width=15,
                                         visualize=False):
        """"左右一样重要性的拼接方式"""

        real_overlap_width_half = int(real_overlap_width / 2)
        # 05、实现简单拼接逻辑
        stitch_init_img = np.zeros((stitch_img_height, stitch_img_width, 3), dtype=np.uint8)

        right_img_start_x = stitch_img_width - right_img.shape[1] + real_overlap_width_half

        # 如果进行图像补偿
        if light_uniformity_compensation:
            stitch_x_in_left = left_img.shape[1] - real_overlap_width_half
            stitch_x_in_right = real_overlap_width_half
            adjusted_left_img, adjusted_right_img = self.adjust_partial_brightness_for_stitching(left_img, right_img,
                                                                                                 real_overlap_width=real_overlap_width,
                                                                                                 y_offset_right2left=y_offset_right2left,
                                                                                                 block_size=light_uniformity_compensation_width,
                                                                                                 use_saturation_correct=True,
                                                                                                 use_hue_correct=False
                                                                                                 )
            if self.debug:
                self.save_debug_image(adjusted_left_img, 'light_compensation_720_left_img_adjusted')
                self.save_debug_image(adjusted_right_img, 'light_compensation_720_right_img_adjusted')
            left_img = adjusted_left_img
            right_img = adjusted_right_img
            pass

        # 左图直接放进去
        stitch_init_img[:left_img.shape[0], :left_img.shape[1]] = left_img

        # 右图纠正位置之后放进去
        if y_offset_right2left > 0:
            stitch_init_img[y_offset_right2left:, right_img_start_x:] = \
                right_img[:right_img.shape[0] - y_offset_right2left, real_overlap_width_half:]
        else:
            stitch_init_img[:right_img.shape[0] + y_offset_right2left, right_img_start_x:] = \
                right_img[abs(y_offset_right2left):, real_overlap_width_half:]

        # 可视化操作部分
        if visualize:
            green = (0, 255, 0)
            red = (0, 0, 255)
            blue = (0, 0, 255)
            color3 = (255, 255, 0)
            cv2.rectangle(stitch_init_img, (0, 0), (left_img.shape[1], left_img.shape[0]), color=green)  # 左上图

            cv2.rectangle(stitch_init_img, (0, 0), (left_img.shape[1] - real_overlap_width, left_img.shape[0]),
                          color=color3)

            # 右下图
            if y_offset_right2left > 0:
                cv2.rectangle(stitch_init_img, (right_img_start_x, 0),
                              (stitch_init_img.shape[1], stitch_init_img.shape[0]), color=red)
                pass
            else:
                cv2.rectangle(stitch_init_img, (right_img_start_x, 0),
                              (stitch_init_img.shape[1], stitch_init_img.shape[0]), color=blue)
                pass

        # return result
        return stitch_init_img

    def blend_half_importance_partial_HSV(self, left_img: np.ndarray, right_img: np.ndarray,
                                          stitch_img_width: int, stitch_img_height: int,
                                          y_offset_right2left: int, real_overlap_width: int,
                                          light_uniformity_compensation=False,
                                          light_uniformity_compensation_width=15,
                                          visualize=False):
        """"左右一样重要性的拼接方式"""

        real_overlap_width_half = int(real_overlap_width / 2)
        # 05、实现简单拼接逻辑
        stitch_init_img = np.zeros((stitch_img_height, stitch_img_width, 3), dtype=np.uint8)

        right_img_start_x = stitch_img_width - right_img.shape[1] + real_overlap_width_half

        # 如果进行图像补偿
        if light_uniformity_compensation:
            stitch_x_in_left = left_img.shape[1] - real_overlap_width_half
            stitch_x_in_right = real_overlap_width_half
            adjusted_left_img, adjusted_right_img = self.adjust_partial_brightness_for_stitching(left_img, right_img,
                                                                                                 real_overlap_width=real_overlap_width,
                                                                                                 y_offset_right2left=y_offset_right2left,
                                                                                                 block_size=light_uniformity_compensation_width,
                                                                                                 use_saturation_correct=True,
                                                                                                 use_hue_correct=True
                                                                                                 )
            if self.debug:
                self.save_debug_image(adjusted_left_img, 'light_compensation_720_left_img_adjusted')
                self.save_debug_image(adjusted_right_img, 'light_compensation_720_right_img_adjusted')
            left_img = adjusted_left_img
            right_img = adjusted_right_img
            pass

        # 左图直接放进去
        stitch_init_img[:left_img.shape[0], :left_img.shape[1]] = left_img

        # 右图纠正位置之后放进去
        if y_offset_right2left > 0:
            stitch_init_img[y_offset_right2left:, right_img_start_x:] = \
                right_img[:right_img.shape[0] - y_offset_right2left, real_overlap_width_half:]
        else:
            stitch_init_img[:right_img.shape[0] + y_offset_right2left, right_img_start_x:] = \
                right_img[abs(y_offset_right2left):, real_overlap_width_half:]

        # 可视化操作部分
        if visualize:
            green = (0, 255, 0)
            red = (0, 0, 255)
            blue = (0, 0, 255)
            color3 = (255, 255, 0)
            cv2.rectangle(stitch_init_img, (0, 0), (left_img.shape[1], left_img.shape[0]), color=green)  # 左上图

            cv2.rectangle(stitch_init_img, (0, 0), (left_img.shape[1] - real_overlap_width, left_img.shape[0]),
                          color=color3)

            # 右下图
            if y_offset_right2left > 0:
                cv2.rectangle(stitch_init_img, (right_img_start_x, 0),
                              (stitch_init_img.shape[1], stitch_init_img.shape[0]), color=red)
                pass
            else:
                cv2.rectangle(stitch_init_img, (right_img_start_x, 0),
                              (stitch_init_img.shape[1], stitch_init_img.shape[0]), color=blue)
                pass

        # return result
        return stitch_init_img

    def adjust_partial_brightness_for_stitching(self, left_img, right_img,
                                                real_overlap_width,
                                                y_offset_right2left,
                                                block_size=32,
                                                use_saturation_correct=False,
                                                use_hue_correct=False,
                                                weight_ratio=2.0
                                                ):
        """
        对拼接图像的左右子图的亮度进行调整

        参数:
        left_img: 左图像 (numpy array)
        right_img: 右图像 (numpy array)
        stitch_x_in_left: 做左图上面的拼接位置的x
        stitch_x_in_right: 在右图上面的拼接位置的x
        block_size:

        返回:
        亮度调整后的左右图像

        算法思路：
        我们有两张图片left_img，right_img进行水平拼接，
        因为两张图片有x方向重叠区域 real_overlap_width
        两张图片也有y方向的偏移 y_offset_right2left
        所以两张图片是有真正的重叠区域的，真正的重叠区域的尺寸大概是 width = real_overlap_width，height = img_height-abs(y_offset_right2left)
        我是可以分别在左图和右图把重叠区域弄出来的，分别为 left_overlap_region,right_overlap_region，并且他们尺寸是一致的
        因为拍照的时候的光照是不均匀的，所以左右图的重叠区域是有亮度差异的
        我现在的需求就是解决这种亮度差异
        我的算法思路如下：
        我可以把left_overlap_region和right_overlap_region分成多个block_sizexblock_size的区域，比如32x32的区域
        对于每个区域，我都可以算出亮度均值，也就是hsv里面的v的均值，
        然后把 left_overlap_region和right_overlap_region 都朝着这个均值调整
        这样我们就得到了 亮度调整后的 left_overlap_region和right_overlap_region
        再把 亮度调整后的 left_overlap_region和right_overlap_region 分别放回 adjusted_left_img 和 adjusted_right_right
        那么我就就得到了亮度调整的 adjusted_left_img 和 adjusted_right_right
        我就就可以用这两张图片进行拼接了

        算法优化：
        现在已经实现了基础的算法，但是还可以优化
        如下代码中的两图的重叠有效区域为：left_effective 和 right_effective
        然后再他们的基础上面进行网格的亮度的矫正，然后替换回原图
        调整左右图的时候，分别是加上的 left_v_diff 和 right_v_diff
        我觉得这里加上 left_v_diff * left_weight 和 right_v_diff * right_weight 效果会更好
        那这两个 left_weight 和 right_weight 如何产生
        对于左 left_effective，我希望这个 left_weight 是 0%到 200% 之间线性变化，
        因为这样0%的区域刚好接近了左图的原图区域，100%的区域刚好接近了和右图的拼接区域
        对于右 right_effective，我希望这个 right_weight 是 200%到 0% 之间线性变化，
        因为这样0%的区域刚好接近了右图的原图区域，100%的区域刚好接近了和左图的拼接区域

        再次优化：
        在计算 left_block 和 right_block 的时候，如果左图或者右图里面的纯黑元素 也就是 rgb为(0,0,0)的元素超过10%
        那么本次不进行计算，或者说直接把 left_v_diff 和 right_v_diff 设置为0
        这是为了避免一张图是黑的，从而影响另一张图，这样会给另一张图加上黑斑

        色调和饱和度方面的优化：
        色调（H: hue），饱和度（S: saturation），亮度（V: value）。
        如下代码中，其实只做了左右拼接图片亮度方面的调整，没有做色调和饱和度方面的调整
        如果加上色调和饱和度方面的调整，整个拼图应该会更加优雅
        色调和饱和度的调整思路，和亮度是一样的
        """
        assert left_img.shape[0] == right_img.shape[0], "左右图尺寸的高必须一样"

        if self.debug:
            print("警告",
                  f"real_overlap_width : {real_overlap_width};  y_offset_right2left : {y_offset_right2left};  block_size : {block_size};  use_saturation_correct : {use_saturation_correct};  use_hue_correct : {use_hue_correct};  weight_ratio : {weight_ratio};  ")

        img_height = left_img.shape[0]

        # 创建左右图像的副本
        adjusted_left_img = left_img.copy()
        adjusted_right_img = right_img.copy()

        if self.debug:
            self.save_debug_image(adjusted_left_img, 'light_compensation_120_left_img_origin')
            self.save_debug_image(adjusted_right_img, 'light_compensation_120_right_img_origin')

        # 提取重叠区域
        left_overlap = left_img[:, -real_overlap_width:]
        right_overlap = right_img[:, :real_overlap_width]

        if self.debug:
            self.save_debug_image(left_overlap, 'light_compensation_140_left_overlap')
            self.save_debug_image(right_overlap, 'light_compensation_140_right_overlap')

        # 考虑y偏移，确定实际有效的重叠区域
        effective_height = img_height - abs(y_offset_right2left)

        if y_offset_right2left > 0:
            left_effective = left_overlap[y_offset_right2left:, :]
            right_effective = right_overlap[:effective_height, :]
        else:
            left_effective = left_overlap[:effective_height, :]
            right_effective = right_overlap[abs(y_offset_right2left):, :]

        if self.debug:
            self.save_debug_image(left_effective, 'light_compensation_160_left_effective')
            self.save_debug_image(right_effective, 'light_compensation_160_right_effective')

        # 计算分块数量
        num_blocks_y = effective_height // block_size
        num_blocks_x = real_overlap_width // block_size

        # 转换为HSV颜色空间
        left_hsv = cv2.cvtColor(adjusted_left_img, cv2.COLOR_BGR2HSV)
        right_hsv = cv2.cvtColor(adjusted_right_img, cv2.COLOR_BGR2HSV)

        # 创建左右图像的权重矩阵 - 横向渐变
        left_weights = np.zeros((real_overlap_width, 1))
        right_weights = np.zeros((real_overlap_width, 1))

        # 左图权重从0%到200%线性变化（从左到右）
        for i in range(real_overlap_width):
            left_weights[i] = i / (real_overlap_width - 1) * weight_ratio

        # 右图权重从200%到0%线性变化（从左到右）
        for i in range(real_overlap_width):
            right_weights[i] = 2.0 - i / (real_overlap_width - 1) * weight_ratio

        if self.debug:
            # 可视化权重矩阵
            left_weight_vis = (left_weights * 127.5).astype(np.uint8)
            right_weight_vis = (right_weights * 127.5).astype(np.uint8)
            left_weight_img = np.tile(left_weight_vis, (effective_height, 1))
            right_weight_img = np.tile(right_weight_vis, (effective_height, 1))
            self.save_debug_image(left_weight_img, 'light_compensation_180_left_weights')
            self.save_debug_image(right_weight_img, 'light_compensation_180_right_weights')

        # 定义黑色像素检测阈值（RGB值之和小于指定值认为是黑色）
        black_threshold = 1  # 调整这个值以定义"黑色"
        max_black_percentage = 0.05  # 最大可接受的黑色像素比例

        # 对每个块进行亮度、色调和饱和度调整
        for y in range(num_blocks_y):
            for x in range(num_blocks_x):
                # 计算当前块的位置
                y_start = y * block_size
                y_end = y_start + block_size
                x_start = x * block_size
                x_end = x_start + block_size

                # 提取左右图像对应的块
                left_block = left_effective[y_start:y_end, x_start:x_end]
                right_block = right_effective[y_start:y_end, x_start:x_end]

                # 检测黑色像素比例
                left_black_pixels = np.sum(np.sum(left_block, axis=2) <= black_threshold)
                right_black_pixels = np.sum(np.sum(right_block, axis=2) <= black_threshold)

                left_black_percentage = left_black_pixels / (left_block.shape[0] * left_block.shape[1])
                right_black_percentage = right_black_pixels / (right_block.shape[0] * right_block.shape[1])

                # 如果任一块中黑色像素比例超过阈值，则跳过调整
                if left_black_percentage > max_black_percentage or right_black_percentage > max_black_percentage:
                    # if self.debug:
                    #     log_info_signal_function("信息",
                    #                              f"跳过块调整，黑色像素比例: 左={left_black_percentage:.2f}, 右={right_black_percentage:.2f}")
                    continue

                # 转换块到HSV颜色空间
                left_block_hsv = cv2.cvtColor(left_block, cv2.COLOR_BGR2HSV)
                right_block_hsv = cv2.cvtColor(right_block, cv2.COLOR_BGR2HSV)

                # 计算各通道均值
                left_h_mean = np.mean(left_block_hsv[:, :, 0])
                right_h_mean = np.mean(right_block_hsv[:, :, 0])
                left_s_mean = np.mean(left_block_hsv[:, :, 1])
                right_s_mean = np.mean(right_block_hsv[:, :, 1])
                left_v_mean = np.mean(left_block_hsv[:, :, 2])
                right_v_mean = np.mean(right_block_hsv[:, :, 2])

                # 计算目标值
                # 对于色调H通道，使用角度平均方法
                h_diff = ((right_h_mean - left_h_mean + 90) % 180) - 90
                target_h_mean = (left_h_mean + h_diff / 2) % 180

                # 对于饱和度S和亮度V通道，使用普通平均
                target_s_mean = (left_s_mean + right_s_mean) / 2
                target_v_mean = (left_v_mean + right_v_mean) / 2

                # 计算调整值
                # 色调使用角度差异
                left_h_diff = ((target_h_mean - left_h_mean + 90) % 180) - 90
                right_h_diff = ((target_h_mean - right_h_mean + 90) % 180) - 90

                # 饱和度和亮度使用普通差异
                left_s_diff = target_s_mean - left_s_mean
                right_s_diff = target_s_mean - right_s_mean
                left_v_diff = target_v_mean - left_v_mean
                right_v_diff = target_v_mean - right_v_mean

                # 获取当前块的平均权重
                avg_left_weight = np.mean(left_weights[x_start:x_end])
                avg_right_weight = np.mean(right_weights[x_start:x_end])

                # 亮度较低区域饱和度调整应减小
                v_factor_left = min(1.0, left_v_mean / 128)
                v_factor_right = min(1.0, right_v_mean / 128)

                # 根据权重调整差异值
                weighted_left_h_diff = left_h_diff * avg_left_weight
                weighted_right_h_diff = right_h_diff * avg_right_weight
                weighted_left_s_diff = left_s_diff * avg_left_weight * v_factor_left
                weighted_right_s_diff = right_s_diff * avg_right_weight * v_factor_right
                weighted_left_v_diff = left_v_diff * avg_left_weight
                weighted_right_v_diff = right_v_diff * avg_right_weight

                # 计算左图中的实际位置
                actual_left_x_start = left_img.shape[1] - real_overlap_width + x_start
                actual_left_x_end = actual_left_x_start + block_size

                # 计算右图中的实际位置
                actual_right_x_start = x_start
                actual_right_x_end = x_start + block_size

                # 计算y方向的实际位置，考虑y偏移
                if y_offset_right2left > 0:
                    actual_left_y_start = y_start + y_offset_right2left
                    actual_left_y_end = actual_left_y_start + block_size
                    actual_right_y_start = y_start
                    actual_right_y_end = y_start + block_size
                else:
                    actual_left_y_start = y_start
                    actual_left_y_end = y_start + block_size
                    actual_right_y_start = y_start + abs(y_offset_right2left)
                    actual_right_y_end = actual_right_y_start + block_size

                # 防止越界
                actual_left_y_end = min(actual_left_y_end, left_hsv.shape[0])
                actual_right_y_end = min(actual_right_y_end, right_hsv.shape[0])

                # 调整左图HSV
                # 色调H通道
                if use_hue_correct:
                    left_h_channel = left_hsv[actual_left_y_start:actual_left_y_end,
                                     actual_left_x_start:actual_left_x_end,
                                     0]
                    left_h_channel = (left_h_channel + weighted_left_h_diff) % 180
                    left_hsv[actual_left_y_start:actual_left_y_end, actual_left_x_start:actual_left_x_end,
                    0] = left_h_channel

                # 饱和度S通道
                if use_saturation_correct:
                    left_hsv[actual_left_y_start:actual_left_y_end, actual_left_x_start:actual_left_x_end, 1] = np.clip(
                        left_hsv[actual_left_y_start:actual_left_y_end, actual_left_x_start:actual_left_x_end,
                        1] + weighted_left_s_diff,
                        0, 255
                    )

                # 亮度V通道
                left_hsv[actual_left_y_start:actual_left_y_end, actual_left_x_start:actual_left_x_end, 2] = np.clip(
                    left_hsv[actual_left_y_start:actual_left_y_end, actual_left_x_start:actual_left_x_end,
                    2] + weighted_left_v_diff,
                    0, 255
                )

                # 调整右图HSV
                # 色调H通道
                if use_hue_correct:
                    right_h_channel = right_hsv[actual_right_y_start:actual_right_y_end,
                                      actual_right_x_start:actual_right_x_end, 0]
                    right_h_channel = (right_h_channel + weighted_right_h_diff) % 180
                    right_hsv[actual_right_y_start:actual_right_y_end, actual_right_x_start:actual_right_x_end,
                    0] = right_h_channel

                # 饱和度S通道
                if use_saturation_correct:
                    right_hsv[actual_right_y_start:actual_right_y_end, actual_right_x_start:actual_right_x_end,
                    1] = np.clip(
                        right_hsv[actual_right_y_start:actual_right_y_end, actual_right_x_start:actual_right_x_end,
                        1] + weighted_right_s_diff,
                        0, 255
                    )

                # 亮度V通道
                right_hsv[actual_right_y_start:actual_right_y_end, actual_right_x_start:actual_right_x_end,
                2] = np.clip(
                    right_hsv[actual_right_y_start:actual_right_y_end, actual_right_x_start:actual_right_x_end,
                    2] + weighted_right_v_diff,
                    0, 255
                )

        # 转换回BGR
        adjusted_left_img = cv2.cvtColor(left_hsv, cv2.COLOR_HSV2BGR)
        adjusted_right_img = cv2.cvtColor(right_hsv, cv2.COLOR_HSV2BGR)

        if self.debug:
            self.save_debug_image(adjusted_left_img, 'light_compensation_320_result_left')
            self.save_debug_image(adjusted_right_img, 'light_compensation_320_result_right')

            # 可视化调整前后的差异
            left_diff = cv2.absdiff(left_img, adjusted_left_img)
            right_diff = cv2.absdiff(right_img, adjusted_right_img)
            self.save_debug_image(left_diff, 'light_compensation_330_left_diff')
            self.save_debug_image(right_diff, 'light_compensation_330_right_diff')

        return adjusted_left_img, adjusted_right_img

    def blend_half_importance_partial_brightness_add_weight(self, left_img: np.ndarray, right_img: np.ndarray,
                                                            stitch_img_width: int, stitch_img_height: int,
                                                            y_offset_right2left: int, real_overlap_width: int,
                                                            light_uniformity_compensation=False,
                                                            light_uniformity_compensation_width=15,
                                                            add_weight_rate=0.1,
                                                            visualize=False):
        """"左右一样重要性的拼接方式"""

        real_overlap_width_half = int(real_overlap_width / 2)
        # 05、实现简单拼接逻辑
        stitch_init_img = np.zeros((stitch_img_height, stitch_img_width, 3), dtype=np.uint8)

        right_img_start_x = stitch_img_width - right_img.shape[1] + real_overlap_width_half

        # 如果进行图像补偿
        if light_uniformity_compensation:
            stitch_x_in_left = left_img.shape[1] - real_overlap_width_half
            stitch_x_in_right = real_overlap_width_half
            adjusted_left_img, adjusted_right_img = self.adjust_partial_brightness_for_stitching(
                left_img, right_img,
                real_overlap_width=real_overlap_width,
                y_offset_right2left=y_offset_right2left,
                block_size=light_uniformity_compensation_width)

            if self.debug:
                self.save_debug_image(adjusted_left_img, 'light_compensation_720_left_img_adjusted')
                self.save_debug_image(adjusted_right_img, 'light_compensation_720_right_img_adjusted')
            left_img = adjusted_left_img
            right_img = adjusted_right_img
            pass

        # 左图直接放进去
        # stitch_init_img[:left_img.shape[0], :left_img.shape[1]] = left_img
        #
        # # 右图纠正位置之后放进去
        # if y_offset_right2left > 0:
        #     stitch_init_img[y_offset_right2left:, right_img_start_x:] = \
        #         right_img[:right_img.shape[0] - y_offset_right2left,real_overlap_width_half:]
        # else:
        #     stitch_init_img[:right_img.shape[0] + y_offset_right2left, right_img_start_x:] = \
        #         right_img[abs(y_offset_right2left):,real_overlap_width_half:]

        stitch_init_img = self.blend_half_importance_add_weight(
            left_img=left_img,
            right_img=right_img,
            stitch_img_width=stitch_img_width,
            stitch_img_height=stitch_img_height,
            y_offset_right2left=y_offset_right2left,
            real_overlap_width=real_overlap_width,
            blend_ratio=add_weight_rate
        )

        # 可视化操作部分
        if visualize:
            green = (0, 255, 0)
            red = (0, 0, 255)
            blue = (0, 0, 255)
            color3 = (255, 255, 0)
            cv2.rectangle(stitch_init_img, (0, 0), (left_img.shape[1], left_img.shape[0]), color=green)  # 左上图

            cv2.rectangle(stitch_init_img, (0, 0), (left_img.shape[1] - real_overlap_width, left_img.shape[0]),
                          color=color3)

            # 右下图
            if y_offset_right2left > 0:
                cv2.rectangle(stitch_init_img, (right_img_start_x, 0),
                              (stitch_init_img.shape[1], stitch_init_img.shape[0]), color=red)
                pass
            else:
                cv2.rectangle(stitch_init_img, (right_img_start_x, 0),
                              (stitch_init_img.shape[1], stitch_init_img.shape[0]), color=blue)
                pass

        # return result
        return stitch_init_img

    def blend_half_importance_global_brightness(self, left_img: np.ndarray, right_img: np.ndarray,
                                                stitch_img_width: int, stitch_img_height: int,
                                                y_offset_right2left: int, real_overlap_width: int,
                                                light_uniformity_compensation=False,
                                                light_uniformity_compensation_width=15,
                                                visualize=False):
        """"左右一样重要性的拼接方式"""
        # self.debug_dir = f"{self.init_debug}_{self.stitch_type}_half_importance"
        # os.makedirs(self.debug_dir, exist_ok=True)
        # if self.debug:
        #     if data_center_algo_inner_signals_obj is not None:
        #         data_center_algo_inner_signals_obj.log_info_signal.emit("警告",f"混合模式：左右半拼模式")

        real_overlap_width_half = int(real_overlap_width / 2)
        # 05、实现简单拼接逻辑
        stitch_init_img = np.zeros((stitch_img_height, stitch_img_width, 3), dtype=np.uint8)
        # 左图直接放进去
        stitch_init_img[:left_img.shape[0], :left_img.shape[1]] = left_img

        # 右图纠正位置之后放进去
        right_img_start_x = stitch_img_width - right_img.shape[1] + real_overlap_width_half

        # 如果进行图像补偿
        if light_uniformity_compensation:
            stitch_x_in_left = left_img.shape[1] - real_overlap_width_half
            stitch_x_in_right = real_overlap_width_half
            adjusted_right_img = self.adjust_global_brightness_for_stitching(left_img, right_img,
                                                                             stitch_x_in_left=stitch_x_in_left,
                                                                             stitch_x_in_right=stitch_x_in_right,
                                                                             half_test_width=light_uniformity_compensation_width)
            if self.debug:
                self.save_debug_image(right_img, 'light_compensation_820_right_img_origin')
                self.save_debug_image(adjusted_right_img, 'light_compensation_820_right_img_adjusted')
            right_img = adjusted_right_img
            pass

        if y_offset_right2left > 0:
            stitch_init_img[y_offset_right2left:, right_img_start_x:] = \
                right_img[:right_img.shape[0] - y_offset_right2left, real_overlap_width_half:]
        else:
            stitch_init_img[:right_img.shape[0] + y_offset_right2left, right_img_start_x:] = \
                right_img[abs(y_offset_right2left):, real_overlap_width_half:]

        # 可视化操作部分
        if visualize:
            green = (0, 255, 0)
            red = (0, 0, 255)
            blue = (0, 0, 255)
            color3 = (255, 255, 0)
            cv2.rectangle(stitch_init_img, (0, 0), (left_img.shape[1], left_img.shape[0]), color=green)  # 左上图

            cv2.rectangle(stitch_init_img, (0, 0), (left_img.shape[1] - real_overlap_width, left_img.shape[0]),
                          color=color3)

            # 右下图
            if y_offset_right2left > 0:
                cv2.rectangle(stitch_init_img, (right_img_start_x, 0),
                              (stitch_init_img.shape[1], stitch_init_img.shape[0]), color=red)
                pass
            else:
                cv2.rectangle(stitch_init_img, (right_img_start_x, 0),
                              (stitch_init_img.shape[1], stitch_init_img.shape[0]), color=blue)
                pass

        # return result
        return stitch_init_img

    def adjust_global_brightness_for_stitching(self, left_img, right_img, stitch_x_in_left, stitch_x_in_right,
                                               half_test_width=15):
        """
        调整拼接处的亮度差异

        参数:
        left_img: 左图像 (numpy array)
        right_img: 右图像 (numpy array)
        stitch_x: 拼接位置的x坐标
        test_width: 测试区域宽度，默认50像素

        返回:
        调整后的右图像
        """
        # 获取测试区域

        # 去掉上下 10%

        assert left_img.shape[0] == right_img.shape[0], "左右图尺寸的高必须一样"

        img_height = left_img.shape[0]
        img_width = left_img.shape[1]
        # img_height_ignore = img_height//7
        # _250609_1106_ 这里是高方向不参与计算的尺寸大小
        img_height_ignore = img_height // 6

        # stitch_x_in_right = img_width - stitch_x_in_left

        # left_test = left_img[img_height_ignore:-img_height_ignore, stitch_x_in_left-test_width:stitch_x_in_left + test_width]
        # right_test = right_img[img_height_ignore:-img_height_ignore, stitch_x_in_right-test_width:stitch_x_in_right + test_width]

        # _250609_1106_  对于左图，宽方向只取了 half_test_width 的宽度
        left_test = left_img[img_height_ignore:-img_height_ignore, stitch_x_in_left - half_test_width:stitch_x_in_left]
        # _250609_1107_ 对于右图，宽方向也只取了 half_test_width 的宽度
        right_test = right_img[img_height_ignore:-img_height_ignore,
                     stitch_x_in_right:stitch_x_in_right + half_test_width]

        if self.debug:
            self.save_debug_image(left_test, 'light_compensation_140_left_test')
            self.save_debug_image(right_test, 'light_compensation_140_right_test')

        # 转换为HSV颜色空间
        left_hsv = cv2.cvtColor(left_test, cv2.COLOR_BGR2HSV)
        right_hsv = cv2.cvtColor(right_test, cv2.COLOR_BGR2HSV)

        # _250609_1117_亮度差异是在中间的一小部分上面的

        # 计算V通道（亮度）的平均差值
        left_v = left_hsv[:, :, 2].mean()
        right_v = right_hsv[:, :, 2].mean()
        v_diff = left_v - right_v

        # 创建右图像的副本
        adjusted_right = right_img.copy()

        # 计算需要调整的区域宽度（右图像左侧1/3）
        adjust_width = right_img.shape[1] // 3

        # 创建渐变权重矩阵
        gradient = np.linspace(1, 0, adjust_width)
        gradient = np.tile(gradient, (right_img.shape[0], 1))

        # 转换右图像为HSV
        right_img_hsv = cv2.cvtColor(adjusted_right, cv2.COLOR_BGR2HSV)

        # 调整V通道
        v_adjustment = gradient * v_diff
        # _250609_1119_ 只调整右图的左边部分
        right_img_hsv[:, :adjust_width, 2] = np.clip(
            right_img_hsv[:, :adjust_width, 2] + v_adjustment, 0, 255
        )

        # 转换回BGR
        adjusted_right = cv2.cvtColor(right_img_hsv, cv2.COLOR_HSV2BGR)

        if self.debug:
            # 创建一张图片，用来显示各种信息
            info_dict = {
                'half_test_width': half_test_width,
                'left_v': left_v,
                'right_v': right_v,
                'v_diff': v_diff,
                'adjust_width': adjust_width,
            }

            # 创建实例
            fry_image_write = FryImageWrite()  # 使用更大的尺寸以获得更好的效果

            # 一行代码创建完整报告
            fry_image_write.create_report("局部亮度调整", info_dict)

            self.save_debug_image(fry_image_write.get_image(), 'light_compensation_160_detail_info')

        return adjusted_right

    def blend_half_importance_add_weight(self, left_img: np.ndarray, right_img: np.ndarray,
                                         stitch_img_width: int, stitch_img_height: int,
                                         y_offset_right2left: int, real_overlap_width: int,
                                         blend_ratio: float = 0.3):
        """重叠区域使用渐变权重的拼接方式
        1. 先用 blend_half_importance 得到基础拼接结果
        2. 提取左右图像的重叠区域
        3. 对重叠区域进行权重融合
        4. 将融合后的重叠区域覆盖到基础拼接结果上
        """

        inner_debug_model = False

        # if self.debug:
        #     log_info_signal_function("警告",f"混合模式：左右权重融合模式")

        # if self.debug:
        #     log_info_signal_function("信息",f"blend_ratio 的值为：{blend_ratio}")

        if blend_ratio > 1:
            blend_ratio = 1
        if blend_ratio < 0.03:
            blend_ratio = 0.03

        if self.debug:
            print("信息", f"blend_ratio 的值处理后为：{blend_ratio}")

        # 1. 获取基础拼接结果
        base_result = self.blend_half_importance(left_img, right_img,
                                                 stitch_img_width, stitch_img_height,
                                                 y_offset_right2left, real_overlap_width)

        if self.debug:
            self.save_debug_image(base_result, 'b_410_base_result_before_weight_blend')

        # 2. 获取重叠区域
        left_width = left_img.shape[1]
        left_overlap = left_img[:, left_img.shape[1] - real_overlap_width:left_img.shape[1]]
        right_overlap = right_img[:, :real_overlap_width]

        if self.debug:
            self.save_debug_image(left_overlap, 'b_422_left_overlap_region')
            self.save_debug_image(right_overlap, 'b_424_right_overlap_region')

        # 4. 根据y偏移计算有效的融合区域
        correct_right_overlap = np.zeros((stitch_img_height, real_overlap_width, 3), dtype=np.uint8)
        if y_offset_right2left > 0:
            correct_right_overlap[y_offset_right2left:, :] = \
                right_overlap[:right_overlap.shape[0] - y_offset_right2left, :]
        else:
            correct_right_overlap[:right_img.shape[0] + y_offset_right2left, :] = \
                right_overlap[abs(y_offset_right2left):, :]

        if self.debug:
            self.save_debug_image(correct_right_overlap, 'b_432_correct_right_overlap')
            self.save_debug_image(left_overlap, 'b_434_left_overlap_region')

        # 获取左图和右图真实融合区域
        real_blend_width = int(round(real_overlap_width * blend_ratio))
        real_blend_start_x = int(round((real_overlap_width - real_blend_width) / 2))
        left_overlap_blend_area = left_overlap[:, real_blend_start_x:real_blend_start_x + real_blend_width]
        right_overlap_blend_area = correct_right_overlap[:, real_blend_start_x:real_blend_start_x + real_blend_width]

        if self.debug:
            self.save_debug_image(left_overlap_blend_area, 'b_442_left_overlap_blend_area')
            self.save_debug_image(right_overlap_blend_area, 'b_444_right_overlap_blend_area')

        # 3. 创建权重矩阵

        weights = np.linspace(1, 0, real_blend_width)
        weights = weights.reshape(1, -1, 1)  # 适应图像的维度 (1, width, 1)

        # 5. 执行权重融合
        blended_overlap = np.round(left_overlap_blend_area * weights + right_overlap_blend_area * (1 - weights)).astype(
            np.uint8)

        if self.debug:
            self.save_debug_image(blended_overlap, 'b_446_blended_overlap_region')

        # 把混合区域放回原重叠区域
        final_blended_overlap = left_overlap.copy()
        final_blended_overlap[:, real_blend_start_x:real_blend_start_x + real_blend_width] = blended_overlap[:, :]
        final_blended_overlap[:, real_blend_start_x + real_blend_width:] = correct_right_overlap[:,
                                                                           real_blend_start_x + real_blend_width:]

        if self.debug:
            self.save_debug_image(blended_overlap, 'b_448_final_blended_overlap')

        # 6. 将融合结果放回原图
        # 计算重叠区域在结果图中的位置
        blended_overlap_width = final_blended_overlap.shape[1]
        base_result[:, left_width - blended_overlap_width:left_width] = final_blended_overlap[:, :]

        if self.debug:
            self.save_debug_image(base_result, 'b_450_final_result_with_weight_blend')

            # 保存调试信息
            with open(os.path.join(self.debug_dir, 'b_520_weight_blend_info.txt'), 'w') as f:
                f.write(f"blend_ratio: {blend_ratio}\n")
                f.write(f"Overlap width: {real_overlap_width}\n")
                f.write(f"real_blend_width: {real_blend_width}\n")
                f.write(f"Y offset: {y_offset_right2left}\n")
                f.write(f"Left overlap shape: {left_overlap.shape}\n")
                f.write(f"Right overlap shape: {right_overlap.shape}\n")
                f.write(f"Blended overlap shape: {blended_overlap.shape}\n")
                f.write(f"real_blend_width: {real_blend_width}\n")
                f.write(f"real_blend_start_x: {real_blend_start_x}\n")

        return base_result
