import os
import yaml
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

from model import CLIPClassifier, CLIPFeatureExtractor
from dataset import MGDataset, load_meta
from utils import setup_logger
from evaluate_common import evaluate_one


SAVE_PATH = "../results/common"
MODEL_NAME = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
SEED = 42


def main():
    seed = SEED
    setup_logger()
    with open("../configs/config.yaml") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = None
    extractor = CLIPFeatureExtractor(model_name=MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(config["model"]["extractor_name"])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip()
    ])
    eval_transform = transforms.Resize((224, 224))
    for model_name in sorted(os.listdir("../models")):
        logging.info(f"Start evaluate {model_name}")
        clip_model = CLIPModel.from_pretrained(MODEL_NAME)
        splitname = model_name.split("_")
        num_classes = int(splitname[1])
        unfrozen = int(splitname[2][-5])
        model = CLIPClassifier(clip_model, num_classes)
        model.load_state_dict(torch.load(os.path.join("../models", model_name), map_location=device))
        model = model.eval().to(device)
            
        logging.info("Evaluating...")
        for test_data in tqdm(sorted(os.listdir("../data"))):
            _, _, test_df = load_meta(os.path.join("../data", test_data), num_classes)
            save_path = os.path.join(SAVE_PATH, f"model_{model_name[:-4]}", f"data_{test_data}")
            if os.path.exists(save_path):
                continue
            test_ds = MGDataset(test_df, processor, os.path.join("../data", test_data), num_classes,
                         transform=train_transform, make_augs=True, seed=seed)
            oodetector, pca, centroids, inv_covs, threshold = None, None, None, None, None
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=True, num_workers=0)
            evaluate_one(model, extractor, oodetector, pca, centroids, inv_covs,
                         test_loader, save_path, device="cuda", num_classes=num_classes, threshold=threshold)

if __name__ == "__main__":
    main()
