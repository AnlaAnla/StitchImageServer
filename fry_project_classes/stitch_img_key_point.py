import sys
import logging
from pathlib import Path

import cv2
import numpy as np
import os
import time
from typing import Optional, Tuple, List
from dataclasses import dataclass
from fry_project_classes.blend_type_mixin import BlendTypeMixin

data_center_algo_inner_signals_obj = None


@dataclass
class FeatureMatchResult:
    """特征匹配结果的数据类"""
    keypoints1: List[cv2.KeyPoint]
    keypoints2: List[cv2.KeyPoint]
    matches: List[cv2.DMatch]
    transform_matrix: Optional[np.ndarray]
    match_score: float
    offset_x: int
    offset_y: int


class ImageStitcherKeyPoint(BlendTypeMixin):
    """基于特征点的图像拼接器"""

    def __init__(self, estimate_overlap_pixels=800,
                 center_ratio=0.8,
                 stitch_type="vertical",
                 blend_type='half_importance',
                 debug=False,
                 debug_dir='debug_output',
                 min_matches=10,
                 feature_detector='akaze',
                 blend_ratio: float = 0.3,
                 combine_detectors=False):
        """
        初始化拼图器

        参数:
        estimate_overlap_pixels: 预估重叠区域像素数
        center_ratio: 中心区域比例
        stitch_type: 拼接方式 ('vertical' 或 'horizontal')
        blend_type: 融合方式 ('half_importance', 'right_first', 'half_importance_add_weight')
        debug: 是否开启调试模式
        debug_dir: 调试图片保存目录
        min_matches: 最小匹配点数
        feature_detector: 特征检测器类型 ('akaze', 'sift', 'orb', 'brisk', 'combine')
        combine_detectors: 是否组合使用多个检测器
        """

        if data_center_algo_inner_signals_obj is not None:
            print("警告", f"拼图方法：关键点")

        self.estimate_overlap_pixels = estimate_overlap_pixels
        self.estimate_non_overlap_pixels = None
        self.center_ratio = center_ratio
        self.blend_type = blend_type
        self.stitch_type = stitch_type
        self.debug = debug
        self.init_debug = debug_dir
        self.min_matches = min_matches
        self.blend_ratio = blend_ratio

        if self.debug:
            self.debug_dir = f"{self.init_debug}_{self.stitch_type}_{self.blend_type}"
            os.makedirs(self.debug_dir, exist_ok=True)

        # 创建特征检测器和描述符计算器
        self.detector = cv2.AKAZE_create()
        # 创建特征匹配器
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # 匹配得分和变换矩阵
        self.best_score = -1
        self.match_score = -1
        self.transform_matrix = None

        # 预估的重叠区域大小
        self.real_overlap_width = None

        # 初始化特征检测器
        self.feature_detector = feature_detector.lower()
        self.combine_detectors = combine_detectors
        self.detectors = self._init_feature_detectors()

    def _init_feature_detectors(self):
        """初始化特征检测器"""
        detectors = {}

        try:
            # AKAZE检测器
            detectors['akaze'] = cv2.AKAZE_create()

            # SIFT检测器 (需要opencv-contrib-python)
            detectors['sift'] = cv2.SIFT_create()

            # ORB检测器
            detectors['orb'] = cv2.ORB_create(nfeatures=2000,
                                              scaleFactor=1.2,
                                              nlevels=8)

            # BRISK检测器
            detectors['brisk'] = cv2.BRISK_create()

        except Exception as e:
            error_info = f"Warning: Some detectors could not be initialized: {str(e)}"
            if data_center_algo_inner_signals_obj is not None:
                print("警告", error_info)

        if not detectors:
            raise ValueError("No feature detectors could be initialized")

        return detectors

    def _get_detector_and_matcher(self, detector_name):
        """获取特征检测器和对应的特征匹配器"""
        if detector_name not in self.detectors:
            raise ValueError(f"Unsupported detector: {detector_name}")

        detector = self.detectors[detector_name]

        # 根据检测器类型选择合适的特征匹配器
        if detector_name in ['sift', 'surf']:
            # L2范数更适合SIFT和SURF
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            # 汉明距离更适合二进制描述符(AKAZE, ORB, BRISK)
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        return detector, matcher

    def detect_and_match_features(self, img1: np.ndarray, img2: np.ndarray) -> FeatureMatchResult:
        """使用选定的特征检测器检测并匹配特征点"""
        try:
            if self.feature_detector == 'combine':
                return self._detect_and_match_combined(img1, img2)
            else:
                return self._detect_and_match_single(img1, img2, self.feature_detector)

        except Exception as e:

            if data_center_algo_inner_signals_obj is not None:
                print("警告",
                                                                        f"Feature detection and matching failed: {str(e)}")

            raise

    def _detect_and_match_single(self, img1: np.ndarray, img2: np.ndarray,
                                 detector_name: str) -> FeatureMatchResult:
        """使用单个检测器进行特征检测和匹配"""
        detector, matcher = self._get_detector_and_matcher(detector_name)

        # 检测特征点和计算描述符
        keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
        keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

        if self.debug:
            # 绘制特征点
            img1_kp = cv2.drawKeypoints(img1, keypoints1, None, (255, 0, 0))
            img2_kp = cv2.drawKeypoints(img2, keypoints2, None, (255, 0, 0))
            self.save_debug_image(img1_kp, f'keypoints_img1_{detector_name}')
            self.save_debug_image(img2_kp, f'keypoints_img2_{detector_name}')

        feature_match_result = self._match_features(keypoints1, keypoints2, descriptors1, descriptors2,
                                                    matcher, detector_name, img1=img1, img2=img2)

        return feature_match_result

    def _detect_and_match_combined(self, img1: np.ndarray, img2: np.ndarray) -> FeatureMatchResult:
        """组合使用多个检测器进行特征检测和匹配"""
        all_results = []

        # 对每个检测器分别进行特征检测和匹配
        for detector_name, detector in self.detectors.items():
            try:
                # 使用当前检测器进行特征检测和匹配
                matcher = self._get_detector_and_matcher(detector_name)[1]
                kp1, desc1 = detector.detectAndCompute(img1, None)
                kp2, desc2 = detector.detectAndCompute(img2, None)

                if desc1 is not None and desc2 is not None:
                    # 执行特征匹配
                    matches = matcher.match(desc1, desc2)
                    matches = sorted(matches, key=lambda x: x.distance)

                    # 选择最佳匹配
                    good_matches = matches[:min(50, len(matches))]

                    if len(good_matches) >= self.min_matches:
                        if self.debug:
                            # 绘制当前检测器的特征点和匹配结果
                            img1_kp = cv2.drawKeypoints(img1, kp1, None, (255, 0, 0))
                            img2_kp = cv2.drawKeypoints(img2, kp2, None, (255, 0, 0))
                            self.save_debug_image(img1_kp, f'match_120_keypoints_img1_{detector_name}')
                            self.save_debug_image(img2_kp, f'match_140_keypoints_img2_{detector_name}')

                            match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                            self.save_debug_image(match_img, f'match_160_feature_matches_{detector_name}')

                        # 计算变换矩阵
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        transform_matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)

                        # 计算匹配得分
                        match_score = np.sum(mask) / len(mask)

                        # 添加到结果列表
                        all_results.append({
                            'keypoints1': kp1,
                            'keypoints2': kp2,
                            'matches': good_matches,
                            'transform_matrix': transform_matrix,
                            'match_score': match_score,
                            'detector_name': detector_name
                        })

                        if self.debug:
                            with open(os.path.join(self.debug_dir, f'match_220_match_info_{detector_name}.txt'),
                                      'w') as f:
                                f.write(f"Number of keypoints in img1: {len(kp1)}\n")
                                f.write(f"Number of keypoints in img2: {len(kp2)}\n")
                                f.write(f"Number of matches: {len(matches)}\n")
                                f.write(f"Number of good matches: {len(good_matches)}\n")
                                f.write(f"Match score: {match_score}\n")
                                if transform_matrix is not None:
                                    f.write(f"Transform matrix:\n{transform_matrix}\n")

            except Exception as e:

                if data_center_algo_inner_signals_obj is not None:
                    print("警告",
                                                                            f"Warning: Detection failed for {detector_name}: {str(e)}")

                continue

        if not all_results:
            raise ValueError("No successful feature detection and matching results")

        # 选择得分最高的结果
        best_result = max(all_results, key=lambda x: x['match_score'])

        # 计算最佳结果的偏移量
        offset_x = int(round(best_result['transform_matrix'][0, 2]))
        offset_y = int(round(best_result['transform_matrix'][1, 2]))

        # 更新类的属性
        self.match_score = best_result['match_score']
        self.transform_matrix = best_result['transform_matrix']

        if self.debug:
            with open(os.path.join(self.debug_dir, 'match_240_best_detector_info.txt'), 'w') as f:
                f.write(f"Best detector: {best_result['detector_name']}\n")
                f.write(f"Best match score: {best_result['match_score']}\n")

        return FeatureMatchResult(
            keypoints1=best_result['keypoints1'],
            keypoints2=best_result['keypoints2'],
            matches=best_result['matches'],
            transform_matrix=best_result['transform_matrix'],
            match_score=best_result['match_score'],
            offset_x=offset_x,
            offset_y=offset_y
        )

    def find_homography(self, kp1, kp2, good_matches):
        """计算单应性矩阵"""
        if len(good_matches) < 4:
            return None, None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if self.debug:
            # 保存匹配信息
            with open(os.path.join(self.debug_dir, 'match_320_homography_info.txt'), 'w') as f:
                f.write(f"Homography matrix:\n{H}\n")
                f.write(f"Number of inliers: {np.sum(mask)}\n")

        return H, mask

    def _match_features(self, keypoints1, keypoints2, descriptors1, descriptors2,
                        matcher, detector_name, img1=None, img2=None, is_overlap=True) -> FeatureMatchResult:
        """特征点匹配的通用处理"""
        if descriptors1 is None or descriptors2 is None:
            raise ValueError(f"No descriptors found using {detector_name}")

        # 1. 执行特征匹配
        matches = matcher.match(descriptors1, descriptors2)

        # 2. 计算距离统计
        distances = np.array([m.distance for m in matches])
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        # 3. 筛选好的匹配
        threshold = mean_dist - 0.7 * std_dist
        good_matches = [m for m in matches if m.distance < threshold]
        if len(good_matches) < self.min_matches:
            good_matches = matches[:min(50, len(matches))]

        if self.debug and img1 is not None and img2 is not None:
            # 在重叠区域上显示匹配
            overlap_match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2,
                                                good_matches, None,
                                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            self.save_debug_image(overlap_match_img, f'match_420_overlap_matches_{detector_name}')

        # 计算变换矩阵
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # RANSAC
        transform_matrix, mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0
        )

        # 筛选内点
        inliers = mask.ravel() == 1
        good_matches = [good_matches[i] for i in range(len(good_matches)) if inliers[i]]

        # 使用内点重新计算变换矩阵
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        transform_matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        # 计算得分和偏移
        match_score = np.sum(mask) / len(mask)
        offset_x = int(round(transform_matrix[0, 2]))
        offset_y = int(round(transform_matrix[1, 2]))

        # 这两个之间有16和11的位移非常好解释

        if self.debug:
            with open(os.path.join(self.debug_dir, f'match_520_match_info_{detector_name}.txt'), 'w') as f:
                f.write(f"Transform matrix:\n{transform_matrix}\n")
                f.write(f"Offsets: ({offset_x}, {offset_y})\n")
                f.write(f"Match score: {match_score}\n")
                f.write(f"Number of initial matches: {len(matches)}\n")
                f.write(f"Number of good matches: {len(good_matches)}\n")

        return FeatureMatchResult(
            keypoints1=keypoints1,
            keypoints2=keypoints2,
            matches=good_matches,
            transform_matrix=transform_matrix,
            match_score=match_score,
            offset_x=offset_x,
            offset_y=offset_y
        )

    def save_debug_image(self, img, name, normalize=False):
        """保存调试图片"""
        try:
            if self.debug:
                save_path = os.path.join(self.debug_dir, f"{name}.jpg")
                if normalize:
                    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                    cv2.imwrite(save_path, img_normalized)
                else:
                    cv2.imwrite(save_path, img)

                if data_center_algo_inner_signals_obj is not None:
                    print("信息",
                                                                            f"Debug: Saved {save_path}")

                return True, f"save_debug_image 成功: {save_path}"
            else:
                return False, "debug mode is not enabled"
        except Exception as e:
            msg = f"save_debug_image出现bug: {str(e)}"
            if data_center_algo_inner_signals_obj is not None:
                print("警告",
                                                                        msg)
            return False, msg

    def pad_image(self, img: np.ndarray, target_width: int = None, target_height: int = None) -> np.ndarray:
        """将图片填充到目标尺寸"""
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
            self.save_debug_image(padded_img, 'pad_320_after_padding')

        return padded_img

    def split_image(self, img, is_left_top=True):
        """分割图片为重叠区域和非重叠区域"""
        height, width = img.shape[:2]
        overlap_width = min(self.estimate_overlap_pixels, width // 2)
        non_overlap_width = width - overlap_width

        if is_left_top:
            non_overlap_region = img[:, :non_overlap_width]
            overlap_region = img[:, non_overlap_width:]
            if self.debug:
                self.save_debug_image(non_overlap_region, 'sp_120_left_top_non_overlap')
                self.save_debug_image(overlap_region, 'sp_140_left_top_overlap')
        else:
            overlap_region = img[:, :overlap_width]
            non_overlap_region = img[:, overlap_width:]
            if self.debug:
                self.save_debug_image(overlap_region, 'sp_220_right_bottom_overlap')
                self.save_debug_image(non_overlap_region, 'sp_240_right_bottom_non_overlap')

        return overlap_region, non_overlap_region

    def stitch_horizontal(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """水平拼接两张图片"""

        height, width = left_img.shape[:2]
        left_height, left_width = left_img.shape[:2]

        overlap_width = min(self.estimate_overlap_pixels, width // 2)
        non_overlap_width = width - overlap_width

        # 1. 分割重叠区域
        left_overlap, left_non_overlap = self.split_image(left_img, is_left_top=True)
        right_overlap, right_non_overlap = self.split_image(right_img, is_left_top=False)

        if self.debug:
            # 保存重叠区域的图像，用于调试
            self.save_debug_image(left_overlap, 'h_120_left_overlap_region')
            self.save_debug_image(right_overlap, 'h_140_right_overlap_region')

        # 2. 特征检测和匹配（只在重叠区域进行）
        match_result = self.detect_and_match_features(left_overlap, right_overlap)
        offset_x = match_result.offset_x  # -16
        offset_y = match_result.offset_y  # -11
        match_score = match_result.match_score

        # 匹配点就是右图最左上点的那个点
        # 右图0,0对应的是左图的 16,11
        # 所以重叠区域为
        self.real_overlap_width = overlap_width + offset_x  # 445，用模板匹配算的值也是445
        real_overlap_width = self.real_overlap_width

        # 计算右图相对于左图的y方向的偏移量
        y_offset_right2left = offset_y * (-1)

        # # 计算最终图像尺寸
        stitch_img_width = left_img.shape[1] + right_img.shape[1] - self.real_overlap_width
        stitch_img_height = max(left_img.shape[0], right_img.shape[0])

        if self.debug:
            with open(os.path.join(self.debug_dir, 'h_320_alignment_info.txt'), 'w') as f:
                f.write(f"match_result.offset_x: {match_result.offset_x}\n")
                f.write(f"match_result.offset_y: {match_result.offset_y}\n")
                f.write(f"match_score: {match_score}\n")
                f.write(f"real_overlap_width: {self.real_overlap_width}\n")
                f.write(f"y_offset_right2left: {y_offset_right2left}\n")
                f.write(f"stitch_img_width: {stitch_img_width}\n")
                f.write(f"stitch_img_height: {stitch_img_height}\n")

        if self.blend_type == 'half_importance':
            blend_stitch_img = self.blend_half_importance(left_img, right_img, stitch_img_width, stitch_img_height,
                                                          y_offset_right2left, real_overlap_width)
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
            blend_stitch_img = self.blend_half_importance_add_weight(left_img, right_img,
                                                                     stitch_img_width,
                                                                     stitch_img_height,
                                                                     y_offset_right2left,
                                                                     real_overlap_width,
                                                                     blend_ratio=self.blend_ratio)
        else:
            # 左边优先的拼接方式
            blend_stitch_img = None

        if self.debug:
            self.save_debug_image(blend_stitch_img, 'h_520_horizontal_stitch_img')

        return blend_stitch_img

    def stitch_vertical(self, top_img: np.ndarray, bottom_img: np.ndarray) -> np.ndarray:
        """垂直拼接两张图片"""
        try:
            # 将图片旋转后调用水平拼接
            top_rotated = cv2.rotate(top_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            bottom_rotated = cv2.rotate(bottom_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if self.debug:
                self.save_debug_image(top_rotated, 'v_120_top_rotated')
                self.save_debug_image(bottom_rotated, 'v_140_bottom_rotated')

            result_rotated = self.stitch_horizontal(top_rotated, bottom_rotated)

            # 将结果旋转回来
            result = cv2.rotate(result_rotated, cv2.ROTATE_90_CLOCKWISE)

            if self.debug:
                self.save_debug_image(result, 'v_520_final_result_vertical')

            return result

        except Exception as e:
            error_info = f"Vertical stitching failed: {str(e)}"
            if data_center_algo_inner_signals_obj is not None:
                print("警告", error_info)

            raise

    def stitch_main(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, float]:
        """主拼接方法"""
        try:
            # 根据拼接类型选择不同的拼接方式
            if self.stitch_type == 'horizontal':
                # 确保两张图片高度相同
                max_height = max(img1.shape[0], img2.shape[0])
                img1 = self.pad_image(img1, target_height=max_height)
                img2 = self.pad_image(img2, target_height=max_height)
                result = self.stitch_horizontal(img1, img2)
            else:  # vertical
                # 确保两张图片宽度相同
                max_width = max(img1.shape[1], img2.shape[1])
                img1 = self.pad_image(img1, target_width=max_width)
                img2 = self.pad_image(img2, target_width=max_width)
                result = self.stitch_vertical(img1, img2)
            self.best_score = self.match_score
            return result

        except Exception as e:
            error_info = f"Image stitching failed: {str(e)}"

            if data_center_algo_inner_signals_obj is not None:
                print("警告", error_info)
            raise


# 测试代码
if __name__ == '__main__':
    pass
    # 设置调试目录和重叠区域估计
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root_path = r"\_250115_Stitch_Image_TemplateMatch\test_images"
    root_path_obj = Path(root_path).absolute()

    stitch_type = "horizontal"
    debug_dir_str = str(root_path_obj / f'debug_{timestamp}_{stitch_type}')
    debug_dir_obj = Path(debug_dir_str).absolute()
    estimate_overlap_ratio = 0.45
    estimate_overlap_pixels = int(round(1024 * estimate_overlap_ratio))

    # 创建特征点拼接器实例
    stitcher = ImageStitcherKeyPoint(
        estimate_overlap_pixels=estimate_overlap_pixels,
        center_ratio=0.8,
        # stitch_type="horizontal",
        stitch_type="vertical",
        blend_type='half_importance_add_weight',
        debug=True,
        debug_dir=debug_dir_str,
        feature_detector='combine',  # 可选: 'akaze', 'sift', 'orb', 'brisk', 'combine'
        blend_ratio=0.5,
        combine_detectors=False
    )

    # 读取测试图片
    img_left_name = "20250123_162407_0001.jpg"
    img_right_name = "20250123_162409_0002.jpg"
    img_bottom_name = "20250123_162422_0007.jpg"

    img_left_path = str(root_path_obj / img_left_name)
    img_right_path = str(root_path_obj / img_right_name)
    img_bottom_path = str(root_path_obj / img_bottom_name)

    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)
    img_bottom = cv2.imread(img_bottom_path)

    if img_left is None or img_right is None:
        print("Error: Could not read one or both images")
        sys.exit(1)

    # 记录开始时间
    start_time = time.time()

    try:
        # 执行拼接
        result_img = stitcher.stitch_main(img_left, img_bottom)

        # 保存结果
        save_final_image_path = str(debug_dir_obj / 'result_img.jpg')
        cv2.imwrite(save_final_image_path, result_img)
        # 计算并打印处理时间
        end_time = time.time()
        print(f"拼接完成！")
        print(f"处理时间: {end_time - start_time:.2f} 秒")
        print(f"匹配得分: {stitcher.best_score:.4f}")
        print(f"结果已保存为: {save_final_image_path}")

    except Exception as e:
        print(f"拼接过程中出错: {str(e)}")
        sys.exit(1)
