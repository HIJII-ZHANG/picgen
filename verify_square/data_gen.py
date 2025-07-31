import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

"""
Synthetic document-style images: table lines + rectangles (checkboxes) + random text
-----------------------------------------------------------------------------------
Outputs
    <out>/images/000001.jpg …
    <out>/train.jsonl                # one JSON per line: {file, boxes}

Each image contains:
    • near-white background with subtle paper noise
    • horizontal + vertical grey lines (like tables)
    • 1-N rectangles (annotated as checkboxes)
    • a few random text snippets in varying fonts/sizes

Example
    python verify_square/data_gen.py \
        --out data_doc  --num 300 \
        --img-size 512 384        \
        --boxes-per-img 2 5        \
        --box-size 30 120          \
        --line-gap 60 140          \
        --text-per-img 5 12        \
        --font "./ttf/SimSun.otf"
"""

def rand_color(lo=0, hi=255):
    return tuple(random.randint(lo, hi) for _ in range(3))

def paper_noise(img: Image.Image, sigma: float = 2.0) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def add_table_lines(img: Image.Image, gap_range):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    # horizontal
    y = random.randint(*gap_range)
    while y < h:
        gray = random.randint(170, 210)
        draw.line([(0, y), (w, y)], fill=(gray, gray, gray), width=1)
        y += random.randint(*gap_range)
    # vertical
    x = random.randint(*gap_range)
    while x < w:
        gray = random.randint(170, 210)
        draw.line([(x, 0), (x, h)], fill=(gray, gray, gray), width=1)
        x += random.randint(*gap_range)

def lorem_word():
    words = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua".split()
    return random.choice(words)

def add_random_text(img: Image.Image, font_paths, n_range, color=(0,0,0)):
    W, H = img.size
    draw = ImageDraw.Draw(img)
    n_min, n_max = n_range
    n = random.randint(n_min, n_max)
    for _ in range(n):
        txt = " ".join(lorem_word() for _ in range(random.randint(1, 4)))
        sz = random.randint(14, 28)
        try:
            font_path = random.choice(font_paths) if font_paths else None
            font = ImageFont.truetype(font_path, sz) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        left, top, right, bottom = draw.textbbox((0, 0), txt, font=font)
        tw, th = right - left, bottom - top
        x = random.randint(5, max(5, W - tw - 5))
        y = random.randint(5, max(5, H - th - 5))
        draw.text((x, y), txt, font=font, fill=color)

def gen_one(cfg):
    W, H = cfg.img_size
    min_boxes, max_boxes = cfg.boxes_per_img

    # auto‑clamp box size so that it fits the image (leave 16‑px padding)
    avail_w = max(1, W - 16)
    avail_h = max(1, H - 16)
    min_box = min(cfg.box_size[0], avail_w, avail_h)
    max_box = min(cfg.box_size[1], avail_w, avail_h)
    if min_box > max_box:
        raise ValueError(
            f"box-size {cfg.box_size} too large for img-size {cfg.img_size}. "
            "Decrease --box-size or increase --img-size."
        )

    img = Image.new("RGB", (W, H), rand_color(235, 255))
    add_table_lines(img, cfg.line_gap)
    img = paper_noise(img, sigma=1.2)

    draw = ImageDraw.Draw(img)
    boxes = []

    for _ in range(random.randint(min_boxes, max_boxes)):
        w = random.randint(min_box, max_box)
        h = random.randint(min_box, max_box)
        max_x1 = W - w - 8
        max_y1 = H - h - 8
        if max_x1 < 8 or max_y1 < 8:
            continue  # skip box if cannot fit (shouldn’t happen due to clamp)
        x1 = random.randint(8, max_x1)
        y1 = random.randint(8, max_y1)
        x2, y2 = x1 + w, y1 + h
        boxes.append([x1, y1, x2, y2])
        draw.rectangle([x1, y1, x2, y2], outline=rand_color(10, 120), width=2)

    add_random_text(img, cfg.fonts, cfg.text_per_img)

    if random.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))

    return img, boxes


def main():
    p = argparse.ArgumentParser("doc-style rectangle dataset generator with text")
    p.add_argument("--out", type=Path, default="data_doc")
    p.add_argument("--num", type=int, default=300)
    p.add_argument("--img-size", nargs=2, type=int, default=[1024, 768])
    p.add_argument("--boxes-per-img", nargs=2, type=int, default=[1, 4])
    p.add_argument("--box-size", nargs=2, type=int, default=[32, 120])
    p.add_argument("--line-gap", nargs=2, type=int, default=[60, 140])
    p.add_argument("--text-per-img", nargs=2, type=int, default=[4, 10])
    p.add_argument("--font", nargs="*", default=[], help="path(s) to .ttf fonts")
    p.add_argument("--seed", type=int, default=42)
    cfg = p.parse_args()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    cfg.fonts = cfg.font  # rename for convenience

    img_dir = cfg.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    anno_path = cfg.out / "dataset.jsonl"

    with anno_path.open("w", encoding="utf-8") as f:
        for i in range(1, cfg.num + 1):
            img, boxes = gen_one(cfg)
            fname = f"{i:06d}.jpg"
            img.save(img_dir / fname, quality=95)
            f.write(json.dumps({"file": fname, "boxes": boxes}, ensure_ascii=False) + "\n")
            if i % 50 == 0 or i == cfg.num:
                print(f"generated {i}/{cfg.num}")

    print("\nDone! Images ->", img_dir)
    print("Annotations ->", anno_path)


if __name__ == "__main__":
    main()
