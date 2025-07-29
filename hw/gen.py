# handwrite.py
# -*- coding: utf-8 -*-
"""
把文字以“拟手写”的实心+细描边方式贴到图片上。
- 先使用 Matplotlib/Agg 将 TextPath 按非零填充规则渲染为实心透明层（抗锯齿好、不空心）。
- 再用可变“压力”(宽度) + 轻微抖动的折线描边，增强手写质感。
- 提供 oversample 超采样，进一步提高清晰度，最后 LANCZOS 下采样。

依赖：
    pip install pillow matplotlib numpy
"""

from __future__ import annotations
import math
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageDraw

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas
from matplotlib import transforms
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath


# --------------------------- 基础工具 ---------------------------

def _quad(p0, p1, p2, t):
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

def _cubic(p0, p1, p2, p3, t):
    return (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t ** 2 * p2
        + t ** 3 * p3
    )

def _path_to_polylines(path: MplPath, steps: int = 24) -> List[np.ndarray]:
    """将 matplotlib Path 近似为若干折线段（多子路径）。"""
    verts, codes = path.vertices, path.codes
    segs: List[List[np.ndarray]] = []
    cur: List[np.ndarray] = []
    i = 0
    while i < len(verts):
        code = codes[i] if codes is not None else MplPath.LINETO
        if code == MplPath.MOVETO:
            if cur:
                segs.append(cur)
            cur = [verts[i].copy()]
        elif code == MplPath.LINETO:
            cur.append(verts[i].copy())
        elif code == MplPath.CURVE3:
            p0, p1, p2 = cur[-1], verts[i], verts[i + 1]
            cur += [_quad(p0, p1, p2, t) for t in np.linspace(0, 1, steps)[1:]]
            i += 1
        elif code == MplPath.CURVE4:
            p0, p1, p2, p3 = cur[-1], verts[i], verts[i + 1], verts[i + 2]
            cur += [_cubic(p0, p1, p2, p3, t) for t in np.linspace(0, 1, steps)[1:]]
            i += 2
        elif code == MplPath.CLOSEPOLY:
            if cur:
                cur.append(cur[0])
                segs.append(cur)
                cur = []
        i += 1
    if cur:
        segs.append(cur)
    return [np.vstack(s) for s in segs if len(s) >= 2]


# --------------------------- 抖动与压力 ---------------------------

def _wobble(pts: np.ndarray, amp: float, freq: float, seed: int):
    """给路径加入细微“手抖”。amp: 像素振幅；freq: 抖动频率。"""
    if amp <= 0:
        return pts
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, len(pts))
    phase_x = rng.uniform(0, 2 * np.pi)
    phase_y = rng.uniform(0, 2 * np.pi)
    dx = amp * np.sin(2 * np.pi * freq * t + phase_x)
    dy = amp * np.sin(2 * np.pi * (freq * 0.8) * t + phase_y)
    noise = rng.normal(scale=amp * 0.15, size=len(pts))
    out = pts.copy()
    out[:, 0] += dx + noise
    out[:, 1] += dy + noise
    return out

def _pressure(n: int, base: float, var: float, seed: int):
    """生成长度为 n 的“压力(半径)”序列。"""
    rng = np.random.default_rng(seed)
    shape = 0.6 + 0.4 * np.sin(np.pi * np.linspace(0, 1, n))  # 中间略粗，两端稍细
    rand = rng.normal(0, 0.15, n)
    return base * np.clip(shape * (1.0 + var * rand), 0.2, 5.0)


# --------------------------- 实心填充图层 ---------------------------

def render_filled_text_layer(
    size_wh: Tuple[int, int],
    text: str,
    font_path: str,
    font_size: int,
    position: Tuple[int, int],
    rotation_deg: float = 0.0,
    color: Tuple[int, int, int] = (0, 0, 0),
    opacity: int = 245,
    fillrule: str | None = None,  # None=默认(非零)，或 "evenodd"
) -> Image.Image:
    """
    使用 Matplotlib/Agg 将 TextPath 渲染为实心透明层 (RGBA)。
    - 坐标系：以 PIL 为准，(0,0) 在左上，y 向下增。
    - position 传入的是左上角像素坐标，内部会做基线与 y 翻转处理。
    """
    W, H = size_wh
    fp = FontProperties(fname=font_path, size=font_size)
    tp = TextPath((0, 0), text, prop=fp, usetex=False)

    fig = Figure(figsize=(W / 100, H / 100), dpi=100)
    Canvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # 与 PIL 对齐：y 轴向下

    xmin, ymin, xmax, ymax = tp.get_extents().bounds
    text_h = ymax - ymin
    x0, y0 = position

    trans = (
        transforms.Affine2D()
        .rotate_deg(rotation_deg)
        .scale(1, -1)  # y 翻转
        .translate(x0 - xmin, y0 + 0.8 * text_h - ymin)  # 基线下移一点更自然
    )

    rgba = (color[0] / 255, color[1] / 255, color[2] / 255, opacity / 255)
    patch = PathPatch(tp, facecolor=rgba, edgecolor="none", fill=True)
    if fillrule is not None:
        # Matplotlib 3.8+ 支持 set_fillrule
        try:
            patch.set_fillrule(fillrule)  # "nonzero" 或 "evenodd"
        except Exception:
            pass
    patch.set_transform(trans + ax.transData)
    ax.add_patch(patch)

    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(arr, mode="RGBA")


# --------------------------- 细描边（带抖动/压力） ---------------------------

def draw_stroke_on_image(
    bg: Image.Image,
    text: str,
    font_path: str,
    font_size: int,
    position: Tuple[int, int],
    *,
    color: Tuple[int, int, int] = (40, 40, 40),
    opacity: int = 240,
    stroke_base: float = 0.7,
    width_scale: float = 0.6,     # 线宽整体再缩放，降低糊度
    caps: bool = False,           # 是否绘制端点圆头，默认关闭减少“堆墨”
    wobble_amp: float = 0.35,
    wobble_freq: float = 8.5,
    rotation_deg: float = -1.2,
    curve_steps: int = 48,
    oversample: int = 3,
    seed: int = 42,
) -> Image.Image:
    """
    仅执行“线条描边”步骤（不做填充）。通常建议先调用 render_filled_text_layer() 贴一层填充，
    再调用本函数增强手写质感，避免空心。
    """
    W, H = bg.size
    scale = max(1, int(oversample))
    canvas = Image.new("RGBA", (W * scale, H * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")

    fp = FontProperties(fname=font_path, size=font_size)
    tp = TextPath((0, 0), text, prop=fp, usetex=False)
    polylines = _path_to_polylines(tp, steps=curve_steps)

    xmin, ymin, xmax, ymax = tp.get_extents().bounds
    text_h = ymax - ymin
    x0, y0 = position
    x0 *= scale
    y0 *= scale

    theta = math.radians(rotation_deg)
    rot_m = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])

    for seg in polylines:
        seg = (seg @ rot_m.T)
        seg[:, 1] = -seg[:, 1]  # y 翻转
        seg[:, 0] += x0 - xmin * scale
        seg[:, 1] += y0 + (text_h * 0.8) * scale - ymin * scale

        seg = _wobble(seg, wobble_amp * scale, wobble_freq, seed)
        radii = _pressure(len(seg), stroke_base * scale, var=0.35, seed=seed)

        for (x1, y1), (x2, y2), r1, r2 in zip(seg[:-1], seg[1:], radii[:-1], radii[1:]):
            w = max(1, int((r1 + r2) * 0.9 * width_scale))
            if caps:
                draw.ellipse([x1 - r1, y1 - r1, x1 + r1, y1 + r1],
                             fill=(color[0], color[1], color[2], opacity))
            draw.line([x1, y1, x2, y2],
                      fill=(color[0], color[1], color[2], opacity),
                      width=w)
        if caps:
            x2, y2, r2 = *seg[-1], radii[-1]
            draw.ellipse([x2 - r2, y2 - r2, x2 + r2, y2 + r2],
                         fill=(color[0], color[1], color[2], opacity))

    canvas = canvas.resize((W, H), Image.LANCZOS)
    out = bg.convert("RGBA")
    out.alpha_composite(canvas)
    return out.convert("RGB")


# --------------------------- 一体化：填充 + 细描边 ---------------------------

def draw_stroke_on_image_with_fill(
    bg: Image.Image,
    text: str,
    font_path: str,
    font_size: int,
    position: Tuple[int, int],
    *,
    # 填充层参数
    fill_color: Tuple[int, int, int] = (40, 40, 40),
    fill_opacity: int = 245,
    rotation_deg: float = -1.2,
    fillrule: str | None = None,   # None 或 "evenodd"
    # 细描边参数（轻微即可，避免变粗）
    line_color: Tuple[int, int, int] = (40, 40, 40),
    line_opacity: int = 235,
    stroke_base: float = 0.65,
    width_scale: float = 0.55,
    caps: bool = False,
    wobble_amp: float = 0.32,
    wobble_freq: float = 8.5,
    curve_steps: int = 48,
    oversample: int = 3,
    seed: int = 42,
) -> Image.Image:
    """推荐的“实心 + 细描边”组合渲染。"""
    # 1) 实心填充层
    filled = render_filled_text_layer(
        size_wh=bg.size,
        text=text,
        font_path=font_path,
        font_size=font_size,
        position=position,
        rotation_deg=rotation_deg,
        color=fill_color,
        opacity=fill_opacity,
        fillrule=fillrule,
    )
    out = bg.convert("RGBA")
    out.alpha_composite(filled)

    # 2) 细描边层（在 filled 基础上再叠加）
    out_rgb = draw_stroke_on_image(
        bg=out.convert("RGB"),
        text=text,
        font_path=font_path,
        font_size=font_size,
        position=position,
        color=line_color,
        opacity=line_opacity,
        stroke_base=stroke_base,
        width_scale=width_scale,
        caps=caps,
        wobble_amp=wobble_amp,
        wobble_freq=wobble_freq,
        rotation_deg=rotation_deg,
        curve_steps=curve_steps,
        oversample=oversample,
        seed=seed,
    )
    return out_rgb


# --------------------------- 便捷封装（读写文件） ---------------------------

def add_handwriting_with_fill(
    image_path: str,
    out_path: str,
    *,
    text: str,
    font_path: str,
    font_size: int = 52,
    position: Tuple[int, int] = (280, 175),
    rotation_deg: float = -1.2,
):
    """
    简化版：采用对 1203x1920 分辨率较友好的默认参数。
    你可以按需要改动 draw_stroke_on_image_with_fill 的参数。
    """
    bg = Image.open(image_path).convert("RGB")
    result = draw_stroke_on_image_with_fill(
        bg=bg,
        text=text,
        font_path=font_path,
        font_size=font_size,
        position=position,
        rotation_deg=rotation_deg,
        # 下面是一组“清晰、不发糊”的默认参数（深灰更像真实笔迹）
        fill_color=(40, 40, 40),
        fill_opacity=245,
        line_color=(40, 40, 40),
        line_opacity=235,
        stroke_base=0.65,     # 如果仍偏粗，降到 0.55；偏细升到 0.75
        width_scale=0.55,     # 控制最终线宽，越小越细
        caps=False,           # 端点不画圆头，避免堆墨
        wobble_amp=0.32,      # 抖动振幅
        wobble_freq=8.5,      # 抖动频率
        curve_steps=48,       # 曲线采样更密，边缘更平滑
        oversample=3,         # 超采样再回缩，提高清晰度
        seed=42,
    )
    result.save(out_path, quality=95)
    return result


# --------------------------- 示例 ---------------------------

if __name__ == "__main__":
    """
    示例：
    - 背景图尺寸：1203×1920
    - 文本：科大讯飞有限公司
    - 字体：手写体 TTF（需包含中文字形）
    - 位置：可根据你的表单具体区域微调
    """
    IMAGE = "form.jpg"
    OUT   = "form_with_handwriting.jpg"
    TEXT  = "科大讯飞有限公司"
    FONT  = "./ttf/ZhiMangXing-Regular.ttf"  # 请替换为可用的中文手写体 TTF

    # 你之前用的坐标 (280, 175)，如果位置不准，可微调 y 值，中文常需略大字号
    add_handwriting_with_fill(
        image_path=IMAGE,
        out_path=OUT,
        text=TEXT,
        font_path=FONT,
        font_size=52,
        position=(280, 175),
        rotation_deg=-1.2,
    )
    print("Saved:", OUT)
