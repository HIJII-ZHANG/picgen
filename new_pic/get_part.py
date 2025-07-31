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

# ====== 新增 & 修改部分（放在类里；其余 helper 保持不变）====== #
    def _get_center(self, rect: Rect) -> Tuple[int, int]:
        x1, y1, x2, y2 = rect
        return (x1 + x2) // 2, (y1 + y2) // 2

    def _within_thresh(self, dx: int, dy: int, thresh: int = 3) -> bool:
        return abs(dx) <= thresh and abs(dy) <= thresh

    def _query_delta_on_crop(
        self,
        crop: Image.Image,
        cx: int,
        cy: int,
        *,
        thresh: int = 3,
    ) -> Tuple[int, int]:
        """
        让模型告诉我们 (当前勾中心) 与 (最近方框中心) 的 dx,dy。
        返回整数像素差值 (dx, dy)。
        """
        data_url = self._encode_img_to_data_url(crop, fmt="JPEG")
        sys_prompt = (
            f"当前图像尺寸 {crop.size[0]}x{crop.size[1]}。"
            f"已在 ({cx}, {cy}) 处打勾。请找离该点最近的方框中心，"
            f"并返回 JSON {{\"dx\": Δx, \"dy\": Δy}} 表示最近方框中心 - 勾中心。以左上角为原点"
            f"不要输出其他内容。"
        )

        accurate_client = client(
            sys_prompt=sys_prompt,
            model="qwen-vl-max-latest",
        )

        text = accurate_client.chat_completion(image=data_url)
        try:
            obj = json.loads(re.search(r"\{.*\}", text, flags=re.S).group(0))
            return int(obj.get("dx", 0)), int(obj.get("dy", 0))
        except Exception:
            # 解析失败就返回大差值，逼迫继续循环
            return thresh * 2, thresh * 2

# ------------------------- 重写 tick_region ------------------------- #
    def tick_region(
        self,
        img: Image.Image,
        region_rect: Rect,
        *,
        margin_inner: int = 0,
        max_iter: int = 10,          # 最多修正 7 次
        thresh_px: int = 1,        # 允许中心差阈值
    ) -> Tuple[Rect, Image.Image]:
        """
        裁剪 → 模型给 bbox → 询问 dx,dy → 调整 → 满足阈值后画勾 → 贴回整图
        """
        W, H = img.size
        gx1, gy1, gx2, gy2 = self._clamp_rect(region_rect, W, H)
        crop = img.crop((gx1, gy1, gx2, gy2))
        cw, ch = crop.size

        # 1) 首次让模型给 bbox
        rect_local = self._infer_bbox_on_crop(crop)

        for attempt in range(max_iter + 1):
            cx, cy = self._get_center(rect_local)

            dx, dy = self._query_delta_on_crop(crop, cx, cy, thresh=thresh_px)

            print(dx,dy)

            if self._within_thresh(dx, dy, thresh_px):
                break  # 已在框中心附近
            # 否则整体平移 bbox，并继续循环
            rect_local = (
                max(0, min(cw - 1, rect_local[0] + dx)),
                max(0, min(ch - 1, rect_local[1] + dy)),
                max(0, min(cw,     rect_local[2] + dx)),
                max(0, min(ch,     rect_local[3] + dy)),
            )

        # 2) 最终 bbox 内外扩 & 画勾
        bx1, by1, bx2, by2 = rect_local
        bx1 = max(0, min(cw - 1, bx1 - margin_inner))
        by1 = max(0, min(ch - 1, by1 - margin_inner))
        bx2 = max(0, min(cw,     bx2 + margin_inner))
        by2 = max(0, min(ch,     by2 + margin_inner))
        rect_local = (bx1, by1, bx2, by2)

        draw = ImageDraw.Draw(crop)
        self._draw_tick(draw, rect_local, self.tick_size, self.tick_width)

        # 3) 贴回整图
        img.paste(crop, (gx1, gy1))

        # 4) 返回换算后的整图坐标
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
