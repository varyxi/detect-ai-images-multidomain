import random
from tqdm import tqdm
from pathlib import Path
import yaml
import os

import numpy as np
import hashlib
from datasets import load_dataset
from tqdm import tqdm
import csv


DATASET_NAME = "NasrinImp/Defactify4_Train"
LABEL_MAP = {
    "coco_image"      : 0,
    "sd3_image"       : 1,
    "sd21_image"      : 2,
    "sdxl_image"      : 3,
    "dalle_image"     : 4,
    "midjourney_image": 5,
}

def main():
    with open("../../configs/config.yaml") as f:
        config = yaml.safe_load(f)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)

    # hash-based split: caption -> split
    def split_name(caption):
        h = int(hashlib.sha1(caption.encode()).hexdigest(), 16) % 100
        return "train" if h < 80 else "val" if h < 90 else "test"

    # out dirs
    root_csv = Path("../../data/defactify")
    root_img = Path("../../data/defactify/images")
    root_csv.mkdir(exist_ok=True)
    root_img.mkdir(exist_ok=True)

    writers = {}
    files = {}
    for split in ["train", "val", "test"]:
        (root_img / split).mkdir(parents=True, exist_ok=True)

        f = open(root_csv / f"{split}.csv", "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["split", "label", "image_path", "prompt"])
        writers[split] = writer
        files[split] = f

    stream_ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    idx_counter = {"train": 0, "val": 0, "test": 0}

    print("Streaming and saving...")
    for row in tqdm(stream_ds, unit="bundle"):
        sp = split_name(row["caption"])
        caption = row["caption"]
        for col, lab in LABEL_MAP.items():
            img = row[col]
            if img is None:
                continue
            idx = idx_counter[sp]
            idx_counter[sp] += 1

            out_path = root_img / sp / f"{idx:07d}.jpg"
            img.convert("RGB").save(out_path, quality=90)
            writers[sp].writerow([sp, lab, f"images/{sp}/{idx:07d}.jpg", caption])

    for f in files.values():
        f.close()

    print("All done")

if __name__ == "__main__":
    main()
