import json, torch
from PIL import Image
from torchvision import transforms as T
from pathlib import Path


class BoxDataset(torch.utils.data.Dataset):
    def __init__(self, anno_path, img_dir, transforms=None):
        self.items = [json.loads(l) for l in Path(anno_path).read_text().splitlines()]
        self.img_dir = Path(img_dir)
        self.tf = transforms or T.Compose([T.ToTensor()])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        meta = self.items[idx]
        img  = Image.open(self.img_dir / meta["file"]).convert("RGB")
        # boxes: (N,4) tensor, labels: (N,) all 1 (类别 id=1 表示“方框”)
        boxes = torch.tensor(meta["boxes"], dtype=torch.float32)
        labels = torch.ones(len(boxes), dtype=torch.long)
        target = {"boxes": boxes, "labels": labels}
        return self.tf(img), target

