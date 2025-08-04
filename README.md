## 简介

这是一个票据类ocr模型训练数据生成器，通过原始票据生成新的票据，在生成之前需要指定一定范围的文字区域和每个需要多选一的勾选框的大区域。现在这些数据在form.json中。

## 实现路径

1. 裁剪以标注好的含有勾选框的区域，交给一已训练好的方块识别大模型识别

2. 为识别区域画勾，重新将裁剪区域粘贴回图片中（get_part.py）

3. 为标注的文字区域补充文字，为字体提供选择（new_pic/new_pic_data.py）

4. 生成新的预选文字（new_pic/new_content.py）

5. （可选）真实化图片（new_pic/real.py）

## 识别方框的模型

采取 3 种方案，

1. 自行训练：

需运行：verify_square/data_gen.py
```bash
        uv run verify_square/data_gen.py \
        --out data_doc  --num 300 \
        --img-size 512 384        \
        --boxes-per-img 2 5        \
        --box-size 30 120          \
        --line-gap 60 140          \
        --text-per-img 5 12        \
        --font "./ttf/SimSun.otf
```
再运行：verify_square/data_gen.py
```bash
        uv run verify_square/make_split.py
    
```
再运行：verify_square/trainer.py
```bash
uv run verify_square/trainer.py --train data_doc/train.jsonl --val data_doc/val.jsonl \
       --images data_doc/images --epochs 10 \
       --batch 2 --accum 2 --lr 1e-4
```
至此训练阶段已完成，如果需要加强模型识别能力可以加大生成训练数据数量和修改一些训练参数。

2. 开源模型文件

在 models/detector-model.pt

3. 大模型识别

使用 vl 模型，准确率很低。

在代码中为上述三种方式依次尝试，第 3 种方式可以保证有输出。

## 生成新预选文字

采用vl大模型，特别注意需要将表单和所有json文字内容一并传入，否则生成效果会显著降低。

## json格式

```json
 {
    "bbox": [ //框选区四点坐标，顺序不重要
      [
        550,
        131
      ],
      [
        550,
        142
      ],
      [
        620,
        131
      ],
      [
        620,
        142
      ]
    ],
    "transcription": "2030.1.1", //当前文字内容，若style=click 可为空或删除
    "style": "handwritten", //handwritten:文字内容, click:打勾内容
    "optional": [ //待替换文字内容，为加速可做cache设计，若style=click 可为空或删除
      "2035.1.1",
      "2040.1.1"
    ]
 }
```