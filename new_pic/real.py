import cv2, math, random
import numpy as np
from augraphy import *

# 读入你的图片
img = cv2.imread("form_out.jpg")

pipeline = AugraphyPipeline(
    ink_phase=[],
    paper_phase=[],
    post_phase=[
        # 轻微取景误差：微缩放、轻微平移、1°内旋转、四周裁掉1%
        Geometric(
            scale=(0.98, 1.02),
            translation=(0.0, 0.015),          # y 方向 1.5% 上移
            fliplr=0, flipud=0,
            crop=(0.01, 0.01, 0.99, 0.99),     # 保留中心 98%
            rotate_range=(-1, 1),
            padding=[0, 0, 0, 0],
            randomize=0,
            p=1.0,
        ),

        # 底边轻微投影阴影
        ShadowCast(
            shadow_side="bottom",
            shadow_vertices_range=(2, 3),      # 整数
            shadow_width_range=(0.45, 0.70),   # 按宽度比例
            shadow_height_range=(0.30, 0.55),  # 按高度比例
            shadow_color=(0, 0, 0),
            shadow_opacity_range=(0.22, 0.35),
            shadow_iterations_range=(1, 1),    # 整数
            shadow_blur_kernel_range=(101, 151),  # 必须为奇数区间，整数
            p=0.9,
        ),

        # 小而柔的反射高光
        ReflectedLight(
            reflected_light_smoothness=0.85,
            # 允许 0–1 比例，内部会按最小边转像素再取随机整数
            reflected_light_internal_radius_range=(0.02, 0.05),
            reflected_light_external_radius_range=(0.06, 0.16),
            reflected_light_minor_major_ratio_range=(0.78, 0.95),
            reflected_light_color=(255, 255, 255),
            reflected_light_internal_max_brightness_range=(0.88, 0.96),
            reflected_light_external_max_brightness_range=(0.70, 0.85),
            reflected_light_location="random",
            reflected_light_ellipse_angle_range=(330, 360),   # 整数角度
            reflected_light_gaussian_kernel_size_range=(5, 61),  # 整数，奇数会自动纠正
            p=0.65,
        ),

        # 线性灯带/环境光衰减
        LightingGradient(
            light_position=None,    # 随机位置
            direction=95,           # 整数角度
            max_brightness=255,
            min_brightness=0,
            mode="gaussian",
            transparency=0.45,
            numba_jit=1,
            p=0.8,
        ),

        # 细微噪声，避免纯白纯黑块
        SubtleNoise(subtle_range=6, p=0.6),

        # 轻度“滚轴条纹”，加强“扫描/复印器”感
        DirtyRollers(
            line_width_range=(6, 12),   # 必须整数范围
            scanline_type=0,
            p=0.3,
        ),

        # 轻压缩
        #Jpeg(quality_range=(78, 92), p=1.0),
    ],
    random_seed=42,
)

data = pipeline(img)
cv2.imwrite("form_out_realistic.jpg", data)
