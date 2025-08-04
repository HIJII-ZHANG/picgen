from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset -------------------------------------------------------------------
class BoxDataset(torch.utils.data.Dataset):
    """Reads a jsonl with {"file":..., "boxes":[[x1,y1,x2,y2], …]} per line."""

    def __init__(self, anno_file: str | Path, img_dir: str | Path):
        self.img_dir = Path(img_dir)
        with open(anno_file, encoding="utf-8") as f:
            self.samples: List[Dict[str, Any]] = [json.loads(l) for l in f]
        self.tf = ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        img_path = self.img_dir / meta["file"]
        if not img_path.is_file():
            raise FileNotFoundError(img_path)
        from PIL import Image
        img = self.tf(Image.open(img_path).convert("RGB"))
        boxes = torch.tensor(meta["boxes"], dtype=torch.float32)
        target = {
            "boxes": boxes,
            "labels": torch.ones((boxes.size(0),), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))

# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    #if torch.backends.mps.is_available():
    #    return torch.device("mps")
    return torch.device("cpu")


def mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 2 ** 20
    return 0.0

# ---------------------------------------------------------------------------
# Train / Eval --------------------------------------------------------------

def train_one_epoch(model, loader, optim, scaler, device, accum, epoch):
    model.train()
    pbar = tqdm(loader, desc=f"Train {epoch}")
    optim.zero_grad(set_to_none=True)

    for step, (imgs, targets) in enumerate(pbar, 1):
        imgs = [i.to(device) for i in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            loss = sum(model(imgs, targets).values()) / accum
        scaler.scale(loss).backward()
        if step % accum == 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
        pbar.set_postfix(loss=f"{loss.item()*accum:.3f}", mem=f"{mem_mb():.0f}MB")


@torch.no_grad()
def eval_loss(model, loader, device, epoch):
    was_train = model.training
    model.train()  # need train() to get loss_dict
    total, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Val   {epoch}")
    for imgs, targets in pbar:
        imgs = [i.to(device) for i in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        batch_loss = sum(model(imgs, targets).values()).item()
        total += batch_loss; n += 1
        pbar.set_postfix(loss=f"{batch_loss:.3f}")
    if not was_train:
        model.eval()
    return total / max(1, n)

# ---------------------------------------------------------------------------
# Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out_dir", default="models", help="where to save checkpoints")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    device = get_device(); logging.info("device=%s", device)

    train_ds = BoxDataset(args.train, args.images)
    val_ds   = BoxDataset(args.val, args.images)
    train_ld = DataLoader(
    train_ds,
    batch_size=args.batch,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True,
)
    val_ld = DataLoader(
    val_ds,
    batch_size=args.batch,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True,
)

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 2)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        train_one_epoch(model, train_ld, optim, scaler, device, args.accum, ep)
        vloss = eval_loss(model, val_ld, device, ep)
        logging.info("Epoch %d | val_loss=%.4f | mem=%.0fMB", ep, vloss, mem_mb())

        # ---- save best checkpoint ----
        ckpt = {
            "epoch": ep,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "val_loss": vloss,
        }
        torch.save(ckpt, out_dir / "last_boxdet.pth")
        if vloss < best:
            best = vloss
            torch.save(ckpt, out_dir / "best_boxdet.pth")
            logging.info("✔ saved best checkpoint (val_loss=%.4f)", best)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
