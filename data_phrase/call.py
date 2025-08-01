from openai import OpenAI
import base64
import os
import json
import re
import time
from typing import Optional

class client:

    def __init__(self, 
                 sys_prompt: str,  
                 temperature: float = 0.0, 
                 max_retries: int = 2,
                 retry_delay: float = 0.8,
                 model: str = "qwen-vl-max-latest"):
        self.sys_prompt = sys_prompt
        self.temperature = temperature
        self.model = model
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"), # 大模型api key 调用
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" # 大模型api 兼容url
        )

    def chat_completion(self, image: None | str = None, user_prompt: str = "") -> str:
        if self._is_valid_path(image):
            with open(image, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            messages=[
                        {"role": "system", "content": self.sys_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {"type": "image_url", 
                                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            ],
                        },
                    ]
        elif image is not None and self._is_base64_data_uri(image):
            # 处理 base64 数据 URI
            messages = [
                    {"role": "system", "content": self.sys_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", 
                             "image_url": {"url": f"{image}"}},
                        ],
                    },
                ]
        else:
            messages = [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise last_err
        
        raise RuntimeError("unreachable")
    
    def _is_valid_path(self, path: str | None) -> bool:
        """检查路径是否有效"""
        return path is not None and os.path.exists(path) and os.path.isfile(path)

    def _is_base64_data_uri(self, s: str) -> bool:
        """
        判断字符串 s 是否形如 'data:<mime>;base64,<base64 编码>'。

        :param s: 输入字符串
        :return: True / False
        """
        return re.match(r'^data:image/[^;]+;base64,(?P<b64>[A-Za-z0-9+/=]+)$', s) is not None

    def _is_valid_path(self, path: str | None) -> bool:
        """检查路径是否有效"""
        return path is not None and os.path.exists(path) and os.path.isfile(path)


def gen_json():

    client = OpenAI(
        api_key="sk-dd8bea4152e2439daf4eae33234cc929",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    with open("form.jpg", "rb") as f:
       b64 = base64.b64encode(f.read()).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你是一名严格的 OCR 标注员，任务是对输入的单据图片进行文本检测与识别，并输出格式化标注数据。要求：\
    1. **输出格式**:返回值必须是 **JSON 数组**，严禁任何额外字符（含换行前后空格），每个数组元素为一个对象，字段遵循给定 *JSON Schema*，常用字段示例：\
    {\
        \"bbox\": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],\
        \"transcription\": \"文本内容\",\
        \"style\": \"printed\" | \"handwritten\"\
    }\
    2. **bbox 规范**:四点四边形坐标，顺序固定：左上 → 右上 → 右下 → 左下（顺时针）。坐标必须为 **整数对**，单位为像素。\
    4. **文本内容**:无法识别写 \"###\"。\
    5. 特别注意:你接受的图片可能经过了处理和变换导致尺寸和比例发生变化，请确保输出的坐标是基于原始图片的，我会给你原始图片的尺寸和一组手动标注区域做参考：\
        尺寸:1203x1920\
            手动标注区域: [[272, 204], [934, 204], [934, 255], [272, 225]]\
                "},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                }
            ]
        }
    ]

# 3. 发请求
    resp = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=messages,
    )

# parse JSON content from the model response, stripping whitespace and code fences if present
    raw = resp.choices[0].message.content or ""
    raw = raw.strip()
# remove any Markdown code fences and optional language hint
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()
        if len(lines) > 2:
            raw = "\n".join(lines[1:-1]).strip()

# parse JSON and handle empty/invalid response
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
    # fallback to empty list if parsing fails
        data = []

    with open("form_det.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def change_ison():
    pass

if __name__ == "__main__":
    gen_json()
    print("JSON 文件已生成: form_det.json")