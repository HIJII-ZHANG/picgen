from new_pic.new_content import content_processer
from pathlib import Path

if __name__ == "__main__":
    # 测试生成新内容
    test_response = content_processer(json_path = Path("form_new.json"), pic_path = Path("form_out.jpg"))
    test_response.new_content_vl(num=1)
    test_response.optional_to_handwritten()
    test_response.flush_optional()
    #print(new_content.join_handwritten_transcriptions("form_new.json"))