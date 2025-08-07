import os
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_PATH = "/workspace/HDD/machine_images/data/ArtiFact"
SAVE_PATH = "../../data/artifact"
PIC_NUM = 300_000
COLLECTION_NAMES = ["celebahq", "coco", "ffhq", "imagenet", "landscape", "lsun", 
                    "lama", "stable_diffusion", "latent_diffusion", "pro_gan"]
LABEL_MAP = {
    "celebahq":         0,
    "coco":             0,
    "ffhq":             0,
    "imagenet":         0,
    "landscape":        0,
    "lsun":             0,
    "lama":             1,
    "stable_diffusion": 2,
    "latent_diffusion": 3,
    "pro_gan":          4
}

def prepare_artifact_split(
    data_path: str = BASE_PATH,
    collection_names: Dict[str, List[str]] = COLLECTION_NAMES,
    pic_num: int = 6000,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def collect_data(folders: List[str]) -> pd.DataFrame:
        data = []
        for folder in folders:
            folder_path = os.path.join(data_path, folder)
            metadata_path = os.path.join(folder_path, "metadata.csv")
            meta = pd.read_csv(metadata_path)
            if meta.shape[0] < pic_num // len(folders):
                print(f"{folder} has only {meta.shape[0]} images (less than {pic_num // len(folders)})")
            meta = meta.sample(min(pic_num // len(folders), meta.shape[0]), random_state=seed)

            for _, row in meta.iterrows():
                full_path = os.path.join(folder_path, row["image_path"])
                data.append({
                    "image_path": full_path,
                    "label": LABEL_MAP[folder],
                    "source": folder
                })
        return pd.DataFrame(data)

    real_folders = [folder for folder in COLLECTION_NAMES if LABEL_MAP[folder] == 0]
    gen_folders = [folder for folder in COLLECTION_NAMES if LABEL_MAP[folder] != 0]
    real_df = collect_data(real_folders)
    gen_df = collect_data(gen_folders)

    min_len = min(len(real_df), len(gen_df))
    real_df = real_df.sample(min_len, random_state=seed).reset_index(drop=True)
    gen_df = gen_df.sample(min_len, random_state=seed).reset_index(drop=True)

    full_df = pd.concat([real_df, gen_df], ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_val_df, test_df = train_test_split(full_df, test_size=test_ratio, random_state=seed, stratify=full_df["label"])
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio / (1 - test_ratio), random_state=seed, stratify=train_val_df["label"])

    Path(SAVE_PATH).mkdir(exist_ok=True)
    train_df.to_csv(os.path.join(SAVE_PATH, "train.csv"))
    val_df.to_csv(os.path.join(SAVE_PATH, "val.csv"))
    test_df.to_csv(os.path.join(SAVE_PATH, "test.csv"))
    return train_df, val_df, test_df

if __name__ == "__main__":
    prepare_artifact_split()