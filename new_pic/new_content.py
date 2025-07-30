from data_phrase.call import client
from pathlib import Path
import json
import ast
import re
import textwrap

class content_processer:
    def __init__(self, json_path: str | Path, pic_path: str | Path):
        self.json_path = json_path
        self.pic_path = pic_path

    def new_content_vl(self, num: int = 1):
        """
        生成新内容的处理管道，支持视觉大模型。
        """
        llm_client = client(
            sys_prompt=f"你是一个表单填写者，你需要根据已示例生成新的能够填写内容。\
                我会给你若干个词或是数字，用\"|\"隔开作为示例，还有一张图片，图片中填写了这些内容在某些区域。\
                你需要为每个词推断出这些都是属于什么类型的内容（最好根据表单中的提示），并生成{num}个合理的相似的填写内容。\
                务必只输出严格JSON:\
                {{ \"case1\": [\"word1\", \"word2\"...], #顺序为我给你的顺序\
                \"case2\": [\"word1\", \"word2\"...],...}}\
                不要生成除json字典之外的任何内容。\
                例如：如果给你的是一个电话号码，你需要生成{num}个新的电话号码；\
                有些词你需要注意关联性，例如证件类型:居民身份证和证件号码:340111197001011234，\
                你需要从图片中发现关联性并且修改需要使两者关联起来",
            model="qwen-vl-max-latest",
            temperature=0.7,
        )

        # 调用 LLM 生成新内容
        response = llm_client.chat_completion(user_prompt=self._join_handwritten_transcriptions(self.json_path), image=self.pic_path)
        self._append_to_optional(Path("form_new.json"), response)
        return response

    def _join_handwritten_transcriptions(self, json_path: str | Path = None) -> str:
        """返回|分隔的所有 handwritten transcription。"""
        if json_path is None:
            json_path = self.json_path
        with open(json_path, 'r', encoding='utf-8') as fp:
            items = json.load(fp)

        handwritten = [
            item['transcription']
            for item in items
            if item.get('style') == 'handwritten'
        ]

        return '|'.join(handwritten)


    def _parse_response(self, resp: str) -> dict[str, list[str]]:
        """
        将类似于
            ```json
            case6: [...],
            case7: [...]
            ```
        或不带 ``` 的字符串解析为 dict。
        允许裸键 / 单引号 / 末尾逗号 / 跨行。
        """
        # 1. 去掉 Markdown 代码围栏 ```xxx / ``` 及前后空行
        cleaned = re.sub(r'^\s*```.*?\n', '', resp, count=1, flags=re.DOTALL)  # 去开头 ```
        cleaned = re.sub(r'\n```[\s\n]*$', '', cleaned, count=1, flags=re.DOTALL)  # 去结尾 ```
        cleaned = textwrap.dedent(cleaned).strip()

        # 2. 若已包含最外层花括号就直接用，否则补上
        if not cleaned.startswith("{"):
            cleaned = "{" + cleaned + "}"

        # 3. literal_eval 安全解析
        try:
            return ast.literal_eval(cleaned)
        except SyntaxError as e:
            # 附带清晰诊断，方便你直接 print 查看 cleaned 内容调试
            raise SyntaxError(f"解析 response_str 失败，请检查格式:\n-----\n{cleaned}\n-----") from e


    def _append_to_optional(self, anno_path: Path, resp_str: str) -> None:
        cases = self._parse_response(resp_str)

        with anno_path.open(encoding="utf-8") as f:
            template_annos: list[dict] = json.load(f)

        handwritten_total = sum(
            1 for item in template_annos if item.get("style") == "handwritten"
        )
        if handwritten_total == 0:
            raise ValueError("模板中没有 handwritten 项，无法追加数据")
        
        annos = json.loads(json.dumps(template_annos))  # 深拷贝
        for case_name, words in cases.items():
            if len(words) < handwritten_total:
                raise ValueError(
                    f"case '{case_name}' 提供 {len(words)} 个元素，少于 handwritten 项 {handwritten_total} 个"
                )

            word_iter = iter(words)

            for item in annos:
                if item.get("style") == "handwritten":
            # 若没有 optional 字段，先放空列表
                    if "optional" not in item:
                        item["optional"] = []
                    elif isinstance(item["optional"], str):
                # 把旧字符串包成列表，保留原内容
                        item["optional"] = [item["optional"]]
                    elif not isinstance(item["optional"], list):
                        raise TypeError(
                            f"optional 字段类型非法，期望 list/str，实际为 {type(item['optional'])}"
                        )
                    item["optional"].append(next(word_iter))  # 取第一个元素

        out_path = anno_path.with_suffix(".json")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(annos, f, ensure_ascii=False, indent=2)

    def optional_to_handwritten(self, anno_path: Path = None) -> None:
        """
        将 optional 字段转换为 handwritten 字段。
        """
        if anno_path is None:
            anno_path = self.json_path

        with anno_path.open(encoding="utf-8") as f:
            annos: list[dict] = json.load(f)

        for item in annos:
            if "optional" in item:
                new_word = item["optional"]
                if not isinstance(new_word, list):
                    break
                item.update(transcription=new_word.pop())

        with anno_path.open("w", encoding="utf-8") as f:
            json.dump(annos, f, ensure_ascii=False, indent=2)

    def flush_optional(self, anno_path: Path = None) -> None:
        """
        清空 optional 字段。
        """
        if anno_path is None:
            anno_path = self.json_path

        with anno_path.open(encoding="utf-8") as f:
            annos: list[dict] = json.load(f)

        for item in annos:
            if "optional" in item:
                item["optional"] = []

        with anno_path.open("w", encoding="utf-8") as f:
            json.dump(annos, f, ensure_ascii=False, indent=2)

    def optional_is_empty(self, anno_path: Path = None) -> bool:
        """
        检查 optional 字段是否为空。
        """
        if anno_path is None:
            anno_path = self.json_path

        with anno_path.open(encoding="utf-8") as f:
            annos: list[dict] = json.load(f)

        for item in annos:
            if "optional" in item and item["optional"]:
                return False
        return True


def new_content(num: int = 1):
    """
    生成新内容的处理管道。
    """
    llm_client = client(
        sys_prompt=f"你是一个表单填写者，但你没有原始表单只有别人填入的文字。\
            你需要根据已有的填写内容生成新的能够填写内容。\
            我会给你若干个词或是数字，用\"|\"隔开，\
            你需要推断出这些都是属于什么类型的内容，并生成{num}个合理的相似的填写内容。\
            务必只输出严格JSON:\
            - \"value\": 表示你生成的词或者数字的字符串\
            例如：如果给你的是一个电话号码，你需要生成{num}个新的电话号码；\
            有些词你可能不能更换，这需要你自行判断，这是只需要返回{num}个原来的词",
        model="deepseek-r1",
        temperature=0.7,
    )

    # 调用 LLM 生成新内容
    response = llm_client.chat_completion(user_prompt="居民身份证, 港澳台居民证件")
    return response







if __name__ == "__main__":
    # 测试生成新内容
    test_response = new_content(num = 3)
    print(test_response)