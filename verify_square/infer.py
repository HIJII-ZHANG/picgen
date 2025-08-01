import torch
from pathlib import Path
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torchvision.transforms import ToTensor
from ultralytics import YOLO

class BoxDetector:
    def __init__(self, model_path: str | Path, score_thresh: float = 0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = fasterrcnn_resnet50_fpn(weights=None)  # 不要自带 COCO 头
        in_feat = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)["model"])
        self.model.to(self.device)
        self.model.eval()
        self.tf = ToTensor()
        self.score_thresh = score_thresh

    def predict(self, img: Image.Image):
        img_tensor = self.tf(img).to(self.device)
        with torch.no_grad():
            pred = self.model([img_tensor])[0]

        # ③ 可选：按阈值过滤低分框，直接返回 CPU 张量或 numpy
        keep = pred["scores"] >= self.score_thresh
        boxes  = pred["boxes"][keep].cpu()
        scores = pred["scores"][keep].cpu()

        return boxes, scores

class BoxDetector_detector:
    def __init__(self, model_path: Path):
        self.model = YOLO(model_path)

    def predict(self, img: Image.Image):
        results = self.model(img, conf=0.5)
        res     = results[0]
        boxes  = res.boxes.xyxy.cpu()
        scores = res.boxes.conf.cpu()
        return boxes, scores


if __name__ == "__main__":
    # Example usage
    detector = BoxDetector_detector(model_path=Path("models/detector-model.pt"))
    img = Image.open("form.jpg").convert("RGB")
      # Add batch dimension
    boxes, scores = detector.predict(img)
    print(boxes, scores)