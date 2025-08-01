from new_pic import get_part
from new_pic import new_pic_data
from new_pic import new_content

save_path = "form_out.jpg"
pic_path = "form.jpg"
json_path = "form_new.json"

def main():
    
    click_path = "form_clicked.jpg"
    clicker = get_part.CheckboxClicker()
    clicker.tick_from_json_click(pic_path, json_path, click_path)
    filler = new_pic_data.HandwrittenBoxFiller(
        font_path="./ttf/SimSun.otf",
        align="left",
    )
    filler.process(
        image_path=click_path,
        anno_path=json_path,
        out_path=save_path,
    )

    #content_processor = new_content.content_processor(json_path=json_path, save_path)
    #content_processor.new_content_vl(num=1)
    #content_processor.optional_to_handwritten()
    #content_processor.flush_optional()
    
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
