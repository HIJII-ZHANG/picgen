from trainer import model, device
import torch
import torchvision.transforms as T
from PIL import Image

model.eval()
img = Image.open("test.jpg").convert("RGB")
tensor = T.ToTensor()(img).unsqueeze(0).to(device)
with torch.no_grad():
    pred = model(tensor)[0]        # dict: boxes, labels, scores
keep = pred["scores"] > 0.5
boxes = pred["boxes"][keep].cpu().numpy()

# 画框
from PIL import ImageDraw
draw = ImageDraw.Draw(img)
for x1,y1,x2,y2 in boxes:
    draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
img.save("test_out.jpg")
