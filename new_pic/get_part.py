from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Any
from PIL import Image, ImageDraw
from openai import OpenAI
import os, io, json, re, base64
from data_phrase.call import client
import noise

Rect = Tuple[int, int, int, int]

class CheckboxClicker:
    """
    使用视觉大模型在指定区域内选择一个方框并打勾，
    支持从 JSON 中筛选 style == "click" 的项批量处理。

    依赖：Pillow、openai>=1.0
    端点：默认阿里云百炼 OpenAI 兼容接口 /chat/completions
    """

    def __init__(
        self,
        sys_prompt: str = (
            "你是表单勾选助手。你的任务是从图中多个候选方框里挑选一个最合适的方框。"
            "务必只输出严格 JSON：{\"bbox\":[x1,y1,x2,y2]}。"
            "坐标以当前裁剪图左上角为(0,0)，单位为像素。"
        ),
        user_prompt: str = (
            "在这张裁剪图中，找到并仅选择一个需要打勾的方框，返回其像素矩形 bbox。"
        ),
        tick_size: int = 12,
        tick_width: int = 2,
    ):
        self.llm_client = client(
            sys_prompt=sys_prompt,
            model="qwen-vl-max-latest"
        )
        self.user_prompt = user_prompt
        self.tick_size = tick_size
        self.tick_width = tick_width

    # -------------------- Public APIs -------------------- #
    def tick_from_json_click(
        self,
        image_path: str,
        json_path: str,
        out_image: str,
        *,
        style_filter: str = "click",
        margin_outer: int = 0,
        margin_inner: int = 0,
    ) -> List[Rect]:
        """
        从 JSON 中筛选 style == style_filter 的项逐个处理，输出打勾后的整图。
        返回每个被选中方框的整图坐标列表。
        """
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        items = self._load_items(json_path)

        results: List[Rect] = []
        for idx, it in enumerate(items):
            if it.get("style") != style_filter:
                continue
            bbox = it.get("bbox")
            if not bbox:
                continue

            region = self._clamp_rect(self._norm_bbox(bbox), W, H, margin_outer)
            try:
                gbox, img = self.tick_region(
                    img=img,
                    region_rect=region,
                    margin_inner=margin_inner,
                )
                results.append(gbox)
                print(f"[{idx}] clicked bbox: {gbox}")
            except Exception as e:
                print(f"[{idx}] failed: {e}")

        Path(out_image).parent.mkdir(parents=True, exist_ok=True)
        img.save(out_image, quality=95)
        print(f"saved => {out_image}")
        return results

    def tick_region(
        self,
        img: Image.Image,
        region_rect: Rect,           # 整图坐标
        *,
        margin_inner: int = 0,
    ) -> Tuple[Rect, Image.Image]:
        """
        对一块区域执行：裁剪 → 调模型拿 bbox → 在裁剪图画勾 → 贴回整图。
        返回 (global_bbox, modified_img)。
        """
        W, H = img.size
        gx1, gy1, gx2, gy2 = self._clamp_rect(region_rect, W, H, margin=0)
        crop = img.crop((gx1, gy1, gx2, gy2))
        cw, ch = crop.size

        # 调模型（以裁剪图为坐标系）
        rect_local = self._infer_bbox_on_crop(crop)

        # 约束 + 内扩
        bx1, by1, bx2, by2 = rect_local
        bx1 = max(0, min(cw - 1, bx1 - margin_inner))
        by1 = max(0, min(ch - 1, by1 - margin_inner))
        bx2 = max(0, min(cw,     bx2 + margin_inner))
        by2 = max(0, min(ch,     by2 + margin_inner))
        if bx2 <= bx1: bx2 = min(cw, bx1 + 1)
        if by2 <= by1: by2 = min(ch, by1 + 1)
        rect_local = (bx1, by1, bx2, by2)

        # 画对号
        draw = ImageDraw.Draw(crop)
        self._draw_tick(draw, rect_local, self.tick_size, self.tick_width)

        # 贴回
        img.paste(crop, (gx1, gy1))

        # 换算为整图
        global_bbox = (gx1 + bx1, gy1 + by1, gx1 + bx2, gy1 + by2)
        return global_bbox, img

    # -------------------- Core LLM call -------------------- #
    def _infer_bbox_on_crop(self, crop: Image.Image) -> Rect:
        """对裁剪图调用模型并解析 bbox（裁剪图坐标）。带简单重试。"""
        data_url = self._encode_img_to_data_url(crop, fmt="JPEG")
        rsp = self.llm_client.chat_completion(image=data_url, user_prompt=self.user_prompt)
        return self._parse_bbox_from_text(rsp)

    # -------------------- Helpers (static/instance) -------------------- #
    @staticmethod
    def _poly_to_rect(bbox4pts: Iterable[Iterable[int]]) -> Rect:
        xs = [int(p[0]) for p in bbox4pts]
        ys = [int(p[1]) for p in bbox4pts]
        return min(xs), min(ys), max(xs), max(ys)

    @classmethod
    def _norm_bbox(cls, bbox) -> Rect:
        if (
            isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(isinstance(v, (int, float)) for v in bbox)
        ):
            x1, y1, x2, y2 = map(int, bbox)
            return x1, y1, x2, y2
        return cls._poly_to_rect(bbox)

    @staticmethod
    def _clamp_rect(rect: Rect, w: int, h: int, margin: int = 0) -> Rect:
        x1, y1, x2, y2 = rect
        x1 = max(0, int(x1 - margin))
        y1 = max(0, int(y1 - margin))
        x2 = min(w, int(x2 + margin))
        y2 = min(h, int(y2 + margin))
        if x2 <= x1: x2 = min(w, x1 + 1)
        if y2 <= y1: y2 = min(h, y1 + 1)
        return x1, y1, x2, y2

    @staticmethod
    def _encode_img_to_data_url(img: Image.Image, fmt: str = "JPEG") -> str:
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _parse_bbox_from_text(text: str) -> Rect:
        # JSON 优先
        try:
            m = re.search(r"\{.*?\}", text, flags=re.S)
            if m:
                obj = json.loads(m.group(0))
                bx = obj.get("bbox")
                if isinstance(bx, list) and len(bx) == 4:
                    return tuple(int(v) for v in bx)  # type: ignore
        except Exception:
            pass
        # 退化：抓 4 个整数
        nums = list(map(int, re.findall(r"-?\d+", text)))
        if len(nums) >= 4:
            return nums[0], nums[1], nums[2], nums[3]
        raise ValueError(f"cannot parse bbox from model output: {text!r}")

    @staticmethod
    def _draw_tick(draw: ImageDraw.ImageDraw, rect: Rect, size: int, width: int) -> None:
        x1, y1, x2, y2 = rect
        x1, y1 = noise.add_jitter_to_segment((x1, y1), sigma=0.5)
        x2, y2 = noise.add_jitter_to_segment((x2, y2), sigma=0.5)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        draw.line((cx - size // 2, cy, cx - size // 8, cy + size // 2), width=width, fill=(0, 0, 0))
        draw.line((cx - size // 8, cy + size // 2, cx + size // 2, cy - size // 2), width=width, fill=(0, 0, 0))

    @staticmethod
    def _load_items(json_path: str) -> List[Dict[str, Any]]:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        if isinstance(data, list):
            return data
        raise ValueError("JSON 格式不支持：应为数组或含 items 字段的对象")


# -------------------- 使用示例 -------------------- #
if __name__ == "__main__":
    clicker = CheckboxClicker()

    # 1) 从 JSON 读取 style=="click" 的框并全部处理
    boxes = clicker.tick_from_json_click(
        image_path="form.jpg",
        json_path="form_new.json",
        out_image="form_click_all.jpg",
    )
    print("clicked boxes:", boxes)
