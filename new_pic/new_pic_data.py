from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence, Tuple, List, Iterable
from PIL import Image, ImageDraw, ImageFont
import noise


class HandwrittenBoxFiller:
    """
    将标注中 style == 'handwritten' 的框涂白并写入文字。
    也可通过 style_filter 自定义需要处理的样式集合。
    """

    def __init__(
        self,
        font_path: str | Path,
        *,
        text_color: Tuple[int, int, int] = (0, 0, 0),
        padding_ratio: float = 0.08,
        align: str = "left",  # "left" | "center"
        min_font_size: int = 8,
        max_font_size: int = 300,
        style_filter: Iterable[str] | None = ("handwritten",),
    ) -> None:
        self.font_path = str(font_path)
        self.text_color = text_color
        self.padding_ratio = float(padding_ratio)
        self.align = align
        self.min_font_size = int(min_font_size)
        self.max_font_size = int(max_font_size)
        self.style_filter = (
            {s.lower() for s in style_filter} if style_filter is not None else None
        )

    # ---------- public API ----------

    def process(
        self,
        image_path: str | Path,
        anno_path: str | Path,
        out_path: str | Path,
    ) -> None:
        """处理一张图片并输出。"""
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        items = self._load_items(anno_path)

        for item in items:
            if not self._should_handle(item):
                continue

            text = str(item.get("transcription", "") or "")
            bbox = item.get("bbox")
            if not bbox or not text:
                print(f"Skipping item with missing bbox or text: {item}")
                continue

            x1, y1, x2, y2 = self._bbox_to_rect(bbox)
            if x2 <= x1 or y2 <= y1:
                continue

            # 添加细微噪声
            x1, y1 = noise.add_jitter_to_segment((x1, y1), sigma=0.5)
            x2, y2 = noise.add_jitter_to_segment((x2, y2), sigma=0.5)

            # 背景填白
            draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))

            # 计算内边距与可用区域
            pad = int((y2 - y1) * self.padding_ratio)
            ix1, iy1, ix2, iy2 = x1 + pad, y1 + pad, x2 - pad, y2 - pad
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            # 寻找字号
            font = self._fit_font_size(
                draw, text, self.font_path, ix2 - ix1, iy2 - iy1,
                self.min_font_size, self.max_font_size
            )

            # 放置坐标
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
            tw, th = r - l, b - t
            if self.align == "center":
                tx = ix1 + (ix2 - ix1 - tw) // 2
            else:
                tx = ix1
            ty = iy1 + (iy2 - iy1 - th) // 2

            draw.text((tx, ty), text, fill=self.text_color, font=font)
            print("drawn:", text, "at", (tx, ty), "with font size", font.size)

        img.save(out_path, quality=95)

    # ---------- helpers ----------

    @staticmethod
    def _bbox_to_rect(bbox: Sequence[Sequence[int]] | Sequence[int]) -> Tuple[int, int, int, int]:
        """
        支持：
        - [[x1,y1],[x2,y2]]
        - [x1,y1,x2,y2]
        - 四点 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        """
        # 二点
        if isinstance(bbox, (list, tuple)) and len(bbox) == 2 and all(len(p) == 2 for p in bbox):
            x1, y1 = bbox[0]
            x2, y2 = bbox[1]
            return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

        # 四点
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(
            isinstance(p, (list, tuple)) and len(p) == 2 for p in bbox
        ):
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            return min(xs), min(ys), max(xs), max(ys)

        # 扁平
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(
            isinstance(v, (int, float)) for v in bbox
        ):
            x1, y1, x2, y2 = map(int, bbox)
            return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

        raise ValueError(f"Unsupported bbox format: {bbox}")

    @staticmethod
    def _load_items(anno_path: str | Path) -> List[dict]:
        with open(anno_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
            return data["items"]
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError("Unsupported annotation JSON structure")

    def _should_handle(self, item: dict) -> bool:
        if self.style_filter is None:
            return True
        style = str(item.get("style", "")).lower()
        return style in self.style_filter

    @staticmethod
    def _fit_font_size(
        draw: ImageDraw.ImageDraw,
        text: str,
        font_path: str,
        max_w: int,
        max_h: int,
        min_size: int,
        max_size: int,
    ) -> ImageFont.FreeTypeFont:
        lo, hi = min_size, max_size
        best = ImageFont.truetype(font_path, size=max(lo, 1))
        while lo <= hi:
            mid = (lo + hi) // 2
            font = ImageFont.truetype(font_path, size=mid)
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
            w, h = r - l, b - t
            if w <= max_w and h <= max_h:
                best = font
                lo = mid + 1
            else:
                hi = mid - 1
        return best


# ---------------- 使用示例 ----------------
if __name__ == "__main__":
    filler = HandwrittenBoxFiller(
        font_path="./ttf/ZhiMangXing-Regular.ttf",
        align="left",
        style_filter=("handwritten",),  # 也可传 None 处理所有样式
    )
    filler.process(
        image_path="form.jpg",
        anno_path="form_new.json",
        out_path="form_out.jpg",
    )
    print("Saved: form_out.jpg")
