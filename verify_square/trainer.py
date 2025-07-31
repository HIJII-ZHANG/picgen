# -*- coding: utf-8 -*-
"""Train a Faster R‑CNN (ResNet‑50‑FPN) detector for two classes
(background + square box).

Changes ➡️ 2025‑07‑31
--------------------
* **Import fix** for older torchvision versions (< 0.15):
  ``FastRCNNPredictor`` now imported from
  ``torchvision.models.detection.faster_rcnn`` instead of re‑exported top‑level
  module — avoids ``ImportError: cannot import name 'FastRCNNPredictor'``.
* Everything else identical: AMP, grad‑accum, tqdm, CLI.

Run example
-----------
```bash
python fasterrcnn_train.py \
       --train data_doc/train.jsonl --val data_doc/val.jsonl \
       --images data_doc/images --epochs 10 \
       --batch 2 --accum 2 --lr 1e-4
```
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
# ⚠️ 关键修正：FastRCNNPredictor 必须从子模块导入，
# 低版本 torchvision 不会在 detection/__init__.py 里 re‑export 它。
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset -------------------------------------------------------------------
# ---------------------------------------------------------------------------
class BoxDataset(torch.utils.data.Dataset):
    """Dataset that reads ``file``+``boxes`` from jsonl and returns tensor + target."""

    def __init__(self, anno_file: str | Path, img_dir: str | Path):
        self.img_dir = Path(img_dir)
        self.samples: List[Dict[str, Any]] = []
        with open(anno_file, encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

        self.tf = ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        img_path = self.img_dir / meta["file"]
        if not img_path.is_file():
            raise FileNotFoundError(img_path)
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)

        boxes = torch.tensor(meta["boxes"], dtype=torch.float32)
        target = {
            "boxes": boxes,
            "labels": torch.ones((boxes.shape[0],), dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))

# ---------------------------------------------------------------------------
# Utils ---------------------------------------------------------------------
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    #if torch.backends.mps.is_available():
    #    return torch.device("mps")
    return torch.device("cpu")


def mem_usage_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 2 ** 20
    return 0.0

# ---------------------------------------------------------------------------
# Train / Eval --------------------------------------------------------------
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, accum: int, epoch: int):
    model.train()
    pbar = tqdm(loader, desc=f"Train {epoch}")
    optimizer.zero_grad(set_to_none=True)

    for step, (imgs, targets) in enumerate(pbar, 1):
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values()) / accum

        scaler.scale(loss).backward()

        if step % accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        pbar.set_postfix({"loss": f"{loss.item()*accum:.3f}", "mem(MB)": f"{mem_usage_mb():.0f}"})


@torch.no_grad()
def eval_loss(model, loader, device, epoch: int):
    """Compute *training* losses on val-set.
    """
    was_training = model.training
    model.train()     # 必须！在 eval() 下不会给 loss_dict

    total, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Val   {epoch}")
    for imgs, targets in pbar:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)  # returns dict now
        batch_loss = sum(loss_dict.values()).item()
        total += batch_loss
        n += 1
        pbar.set_postfix({"loss": f"{batch_loss:.3f}"})

    if not was_training:
        model.eval()
    return total / max(n, 1)

# ---------------------------------------------------------------------------
# Main ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")

    device = get_device()
    logging.info("Using device: %s", device)

    train_ds = BoxDataset(args.train, args.images)
    val_ds = BoxDataset(args.val, args.images)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, scaler, device, args.accum, epoch)
        val_loss = eval_loss(model, val_loader, device, epoch)
        logging.info("Epoch %d | val_loss=%.4f | mem=%.0f MB", epoch, val_loss, mem_usage_mb())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    main(p.parse_args())
