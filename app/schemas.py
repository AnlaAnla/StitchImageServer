from enum import Enum


class StitchingMethod(str, Enum):
    """拼接方法枚举"""
    KEY_POINT = "key_point"
    TEMPLATE_MATCH = "template_match"


class KeypointFeatureDetector(str, Enum):
    """关键点检测器类型"""
    SIFT = "sift"
    ORB = "orb"
    BRISK = "brisk"
    COMBINE = "combine"


class KeypointBlendType(str, Enum):
    """关键点融合类型"""
    HALF_IMPORTANCE = "half_importance"
    RIGHT_FIRST = "right_first"
    HALF_IMPORTANCE_ADD_WEIGHT = "half_importance_add_weight"
    COMBINE = "combine"


class TemplateBlendType(str, Enum):
    """模板匹配融合类型"""
    HALF_IMPORTANCE_ADD_WEIGHT = "half_importance_add_weight"
    HALF_IMPORTANCE_GLOBAL_BRIGHTNESS = "half_importance_global_brightness"
    HALF_IMPORTANCE_PARTIAL_BRIGHTNESS = "half_importance_partial_brightness"
    BLEND_HALF_IMPORTANCE_PARTIAL_HV = "blend_half_importance_partial_HV"
    BLEND_HALF_IMPORTANCE_PARTIAL_SV = "blend_half_importance_partial_SV"
    BLEND_HALF_IMPORTANCE_PARTIAL_HSV = "blend_half_importance_partial_HSV"
    BLEND_HALF_IMPORTANCE_PARTIAL_BRIGHTNESS_ADD_WEIGHT = "blend_half_importance_partial_brightness_add_weight"
