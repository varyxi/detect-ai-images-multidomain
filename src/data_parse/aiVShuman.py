import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_PATH = "/workspace/HDD/machine_images/REPO/data/aiVShuman/4"
SAVE_PATH = "../../data/aiVShuman"

def main(val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    meta = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
    full_df = pd.DataFrame({"image_path": [], "label": []})
    for _, row in meta.iterrows():
        full_df.loc[len(full_df)] = [os.path.join(BASE_PATH, row.file_name), row.label]

    train_val_df, test_df = train_test_split(full_df, test_size=test_ratio, random_state=seed, stratify=full_df["label"])
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio / (1 - test_ratio), random_state=seed,
                                        stratify=train_val_df["label"])
    Path(SAVE_PATH).mkdir(exist_ok=True)
    train_df.to_csv(os.path.join(SAVE_PATH, "train.csv"))
    val_df.to_csv(os.path.join(SAVE_PATH, "val.csv"))
    test_df.to_csv(os.path.join(SAVE_PATH, "test.csv"))

if __name__ == "__main__":
    main()