# make_split.py
import json, random, argparse, pathlib

def main(src, train_path, val_path, ratio):
    data = [json.loads(l) for l in open(src, encoding="utf-8")]
    random.shuffle(data)                # 洗牌确保随机
    k = int(len(data) * ratio)          # 划分点
    val, train = data[:k], data[k:]

    def dump(lst, path):
        with open(path, "w", encoding="utf-8") as f:
            for x in lst:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    dump(train, train_path)
    dump(val,   val_path)
    print(f"Done. Train={len(train)}, Val={len(val)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src",   default="data_doc/dataset.jsonl")
    p.add_argument("--train", default="data_doc/train.jsonl")
    p.add_argument("--val",   default="data_doc/val.jsonl")
    p.add_argument("--ratio", type=float, default=0.1, help="val 占比")
    args = p.parse_args()
    main(args.src, args.train, args.val, args.ratio)
