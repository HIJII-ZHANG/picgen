import cv2
import json
from paddleocr import PaddleOCR
import multiprocessing as mp

def ocr_json_to_spans(ocr_result: dict,
                      score_thresh: float = 0.0,
                      use_bbox: bool = False) -> list[dict]:
    """
    将 PaddleOCR / MMOCR 之类的结果 JSON 转成:
        [
          {
            "id": "t0000",
            "text": "交通银行单位定期账户开户（变更、撤销）申请书",
            "polygon": [[93,0],[591,0],[591,24],[93,24]],
            "bbox":   {"x_min":93,"y_min":0,"x_max":591,"y_max":24}
          },
          ...
        ]

    Parameters
    ----------
    ocr_result : dict
        你的 OCR 结果字典（包含 rec_texts / rec_polys / rec_scores）。
    score_thresh : float, optional
        低于该置信度的条目将被过滤；默认不过滤。
    use_bbox : bool, optional
        True 时输出 bbox 字段，False 时输出 polygon 字段。
    """
    rec_texts  = ocr_result.get("rec_texts", [])
    rec_polys  = ocr_result.get("rec_polys") or ocr_result.get("dt_polys")
    rec_scores = ocr_result.get("rec_scores", [1.0] * len(rec_texts))

    if not (len(rec_texts) == len(rec_polys) == len(rec_scores)):
        raise ValueError("文本、坐标、置信度数量不一致")

    spans = []
    for i, (text, poly, score) in enumerate(zip(rec_texts, rec_polys, rec_scores)):
        if score < score_thresh:
            continue

        # polygon → axis‑aligned bounding box
        xs, ys = zip(*poly)
        bbox = {"x_min": int(min(xs)), "y_min": int(min(ys)),
                "x_max": int(max(xs)), "y_max": int(max(ys))}

        spans.append({
            "id": f"t{i:04d}",
            "text": text,
            **({"polygon": [[int(x), int(y)] for x, y in poly]} if not use_bbox else {"bbox": bbox})
        })

    return spans

def ocr_processer(image_path: str) -> list[dict]:
    img = cv2.imread(image_path)
    ocr = PaddleOCR(lang="ch", det_limit_side_len=2048, ocr_version='PP-OCRv4')  # 先识别
    result = ocr.predict(image_path)
    for res in result:
        res.print()
        res.save_to_img("output")
        res.save_to_json("output")
    #print(f"识别结果: {result}")
    #spans = ocr_json_to_spans(result, score_thresh=0.5, use_bbox=True)
    
    # 导出 PaddleOCR 标注格式
    #with open("./form_det2.json", "w", encoding="utf-8") as f:
     #   json.dump(spans, f, ensure_ascii=False, indent=2)

    

if __name__ == "__main__":
    ocr_processer("form2.jpg")  # 替换成你的图片路径