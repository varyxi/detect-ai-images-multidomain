import os
import random
from typing import List, Tuple

import albumentations as A
import numpy as np
from PIL import Image
import pandas as pd
from transformers import CLIPProcessor
from torch.utils.data import Dataset
import torch


def load_meta(root, num_classes=6) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(f"{root}/train.csv")
    val_df = pd.read_csv(f"{root}/val.csv")
    test_df = pd.read_csv(f"{root}/test.csv")
    if num_classes == 2:
        train_df["label"] = train_df["label"].apply(lambda x: 0 if x == 0 else 1)
        val_df["label"] = val_df["label"].apply(lambda x: 0 if x == 0 else 1)
        test_df["label"] = test_df["label"].apply(lambda x: 0 if x == 0 else 1)
    return train_df, val_df, test_df

def rotate_180(image: np.ndarray, **kwargs) -> np.ndarray:
    return np.rot90(image, k=2)


class Augmenter:
    """Albumentations‑based augmenter for PIL.Image objects"""
    def __init__(self, base_size: Tuple[int, int] = (224, 224)) -> None:
        self.base_h, self.base_w = base_size
        self._pool: List[A.BasicTransform] = self._create_pool()
        self.transform: A.Compose = self._get_standard_pipeline()

    def _create_pool(self) -> List[A.BasicTransform]:
        return [
            A.RandomResizedCrop(size=(self.base_h, self.base_w), scale=(0.85, 0.95), p=1.0),
            A.RandomRotate90(p=1.0),
            A.Rotate(limit=30, p=1.0),
            A.Lambda(image=rotate_180, p=1.0),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.ToGray(p=1.0),
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=1.0),
            A.RandomBrightnessContrast(p=1.0),
        ]

    def _get_standard_pipeline(self) -> A.Compose:
        return A.Compose([
            A.RandomResizedCrop(size=(self.base_h, self.base_w), scale=(0.85, 0.95), p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.Lambda(image=rotate_180, p=0.2),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ToGray(p=0.2),
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.2),
            A.RandomBrightnessContrast(p=0.3),
        ])

    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        aug_np = self.transform(image=img_np)["image"]
        return Image.fromarray(aug_np)

    def random_transform(self, image: Image.Image, n_aug: int | None = None, *, seed: int = 42) -> Image.Image:
        """Apply a random subset of augmentations

        Parameters
        image : PIL.Image
            Input image.
        n_aug : int | None, optional
            Exact number of transforms to sample. If None (default) – pick a random value in [1, 3]
        seed : int | None, optional
            Seed for reproducibility of the sampling
        """
        if seed is not None:
            random.seed(seed)

        if n_aug is None:
            n_aug = random.randint(1, 3)
        n_aug = max(1, min(n_aug, 3))

        chosen = random.sample(self._pool, k=n_aug)
        aug = A.Compose(chosen)
        aug_np = aug(image=np.array(image))["image"]
        return Image.fromarray(aug_np)


class MGDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, processor: CLIPProcessor, base_path: str, num_classes: int,
                 transform=None, seed: int = 42, make_augs: bool = False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.processor = processor
        self.transform = transform
        self.make_augs = make_augs
        self.base_path = base_path
        self.num_classes = num_classes
        self.seed = seed
        if make_augs:
            self.augmenter = Augmenter()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        if row['image_path'][0] == "/":
            image = Image.open(os.path.join(row['image_path'])).convert("RGB")
        else:
            image = Image.open(os.path.join(self.base_path, row['image_path'])).convert("RGB")
        if self.make_augs and random.randint(0, 1):
            image = self.augmenter.random_transform(image, self.seed)
        if self.transform:
            image = self.transform(image)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        if self.num_classes == 2:
            label = torch.tensor(int(bool(int(row['label']))), dtype=torch.long)
        else:
            label = torch.tensor(row['label'], dtype=torch.long)
        return pixel_values, label
