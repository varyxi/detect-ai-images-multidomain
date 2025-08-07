import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
from typing import Tuple
import numpy as np
import yaml
import logging

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from model import CLIPClassifier
from dataset import MGDataset, load_meta
from utils import setup_logger


def get_class_weights(df: pd.DataFrame) -> torch.Tensor:
    counts = df["label"].value_counts().sort_index()
    total = counts.sum()
    weights = total / (len(counts) * counts)
    # Increase "real" weight (label == 0) to minimize false positives
    weights[0] *= 1.5
    return torch.tensor(weights.values, dtype=torch.float32)


def train(model, train_loader, val_loader, save_path, device="cuda", lr=1e-5, epochs=10, class_weights=None):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for pixel_values, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - train"):
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            logits = model(pixel_values)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pixel_values, labels in val_loader:
                pixel_values, labels = pixel_values.to(device), labels.to(device)
                logits = model(pixel_values)
                loss = criterion(logits, labels)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_model = model
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            logging.info(f"Saved best model (val_loss={best_val_loss:.4f})")

    return best_model


def evaluate(model, test_loader, save_csv_path, device="cuda", num_classes=2):
    model.eval().to(device)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for pixel_values, labels in test_loader:
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            logits = model(pixel_values)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    report = classification_report(all_labels, all_preds, labels=list(range(num_classes)), digits=3)

    logging.info(f"\nTest Accuracy: {acc:.4f}")
    logging.info("Confusion Matrix:\n", cm)
    logging.info("Classification Report:\n", report)

    df = pd.DataFrame({
        "label": all_labels,
        "predicted_label": all_preds,
    })
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
    df.to_csv(save_csv_path, index=False)
    logging.info(f"Test results are saved")


def main():
    with open("../configs/config.yaml") as f:
        config = yaml.safe_load(f)
    num_classes = config["model"]["num_classes"]
    unfreeze_extractor = config["training"]["unfreeze_extractor"]
    dataset_name = config["data"]["dataset_name"]
    seed = config["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() and config["training"].get("use_gpu", 0) else "cpu")
    log_path = f"../results/{dataset_name}_{num_classes}_unfreeze{str(unfreeze_extractor)}.log"
    setup_logger(log_path)
    logging.info(f"Logs will be saved by path {log_path}")

    processor = CLIPProcessor.from_pretrained(config["model"]["extractor_name"])
    clip_model = CLIPModel.from_pretrained(config["model"]["extractor_name"])

    logging.info("Start loading data...")
    train_df, val_df, test_df = load_meta(config["data"]["data_path"], num_classes)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip()
    ])
    eval_transform = transforms.Resize((224, 224))

    train_ds = MGDataset(train_df, processor, config["data"]["data_path"], num_classes,
                         transform=train_transform, make_augs=True, seed=seed)
    val_ds = MGDataset(val_df, processor, config["data"]["data_path"], num_classes,
                       transform=eval_transform, make_augs=False, seed=seed)
    test_ds = MGDataset(test_df, processor, config["data"]["data_path"], num_classes,
                        transform=eval_transform, make_augs=False, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, num_workers=0)
    logging.info("Finish loading data")

    for name, param in clip_model.named_parameters():
        if unfreeze_extractor and "visual.transformer" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model = CLIPClassifier(clip_model, num_classes)
    class_weights = get_class_weights(train_df).to(device)

    save_model_path = f"../models/{dataset_name}_{num_classes}_unfreeze{str(unfreeze_extractor)}.pth"
    save_results_path = f"../results/{dataset_name}_{num_classes}_unfreeze{str(unfreeze_extractor)}.csv"
    logging.info(f"model will be saved by path {save_model_path}")
    logging.info(f"test results will be saved by path {save_results_path}")

    best_model = train(model, train_loader, val_loader, save_model_path, class_weights=class_weights, device=device,
                       lr=float(config["training"]["lr"]), epochs=config["training"]["epochs"])
    evaluate(best_model, test_loader, save_results_path, device, config["model"]["num_classes"])

if __name__ == "__main__":
    main()
