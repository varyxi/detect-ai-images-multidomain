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

from model import CLIPClassifier, CLIPFeatureExtractor
from dataset import MGDataset, load_meta
from utils import setup_logger
from detect_ood import OODetector

SAVE_PATH = "../results/common"
MODEL_NAME = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
SEED = 42

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_one(model, extractor, detector, pca, centroids, inv_covs,
                 test_loader, save_path, device="cuda", num_classes=2, threshold=None):
    model.eval().to(device)

    all_preds, all_labels = [], []
    all_scores, all_ood_dists = [], []
    all_final_preds = []

    with torch.no_grad():
        for pixel_values, labels in test_loader:
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            logits = model(pixel_values)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(pixel_values.shape[0]):
                emb = extractor.get_image_embedding(pixel_values[i].unsqueeze(0)).squeeze(0)
                emb_np = emb.cpu().numpy()
                # is_ood, odd_distance = detector.detect(emb_np, pca, centroids, inv_covs, threshold)

                # final_pred = 0 if is_ood else int(preds[i].cpu().item())

                all_preds.append(int(preds[i].cpu().item()))
                all_labels.append(int(labels[i].cpu().item()))
                # all_final_preds.append(final_pred)
                all_scores.append(round(float(probs[i][int(preds[i].cpu().item())].cpu().item()), 4))
                # all_ood_dists.append(round(odd_distance, 4))

    os.makedirs(save_path, exist_ok=True)

    acc_clf = accuracy_score(all_labels, all_preds)
    cm_clf = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    report_clf = classification_report(all_labels, all_preds, labels=list(range(num_classes)), digits=3)

    # acc_final = accuracy_score(all_labels, all_final_preds)
    # cm_final = confusion_matrix(all_labels, all_final_preds, labels=list(range(num_classes)))
    # report_final = classification_report(all_labels, all_final_preds, labels=list(range(num_classes)), digits=3)

    with open(os.path.join(save_path, "metrics_report.md"), "w") as f:
        f.write("# Classification Metrics (Raw Classifier)\n\n")
        f.write(pd.DataFrame(cm_clf).to_markdown() + "\n\n")
        # f.write("# Classification Metrics (Final Prediction with OOD)\n\n")
        # f.write(pd.DataFrame(cm_final).to_markdown() + "\n\n")
        # f.write("\n" + report_final + "\n```\n")

    df = pd.DataFrame({
        "label": all_labels,
        "predicted_raw": all_preds,
        # "predicted_final": all_final_preds,
        "scores": all_scores,
        # "ood_dists": all_oods
    })
    df.to_csv(os.path.join(save_path, "predictions.csv"), index=False)

    def plot_confusion_matrix(cm, title, filename):
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Pred_{i}' for i in range(num_classes)],
                    yticklabels=[f'True_{i}' for i in range(num_classes)])
        plt.title(title)
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, filename))
        plt.close()

    plot_confusion_matrix(cm_clf, "Confusion Matrix - Raw Classifier", "confusion_matrix_raw.png")
    #plot_confusion_matrix(cm_final, "Confusion Matrix - Final Prediction (OOD filtered)", "confusion_matrix_final.png")

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
        # if splitname[0] != train_data:
        #     train_data = splitname[0]
        #     train_df, _, _ = load_meta(os.path.join("../data", train_data), num_classes)
        #     train_ds = MGDataset(train_df.sample(10_000), processor, os.path.join("../data", train_data), num_classes,
        #                  transform=train_transform, make_augs=True, seed=42)
        #     train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
        #     #train_embeddings = np.array([])
        #     train_embeddings = []
        #     with torch.no_grad():
        #         for pixel_values, labels in train_loader:
        #             pixel_values, labels = pixel_values.to(device), labels.to(device)
        #             with torch.no_grad():
        #                 embedding = extractor.get_image_embedding(pixel_values).squeeze(0)
        #                 embedding_np = embedding.cpu().squeeze().numpy()
        #                 #print(embedding)
        #                 #np.append(train_embeddings, embedding_np)
        #                 train_embeddings.append(embedding_np)

        #     oodetector = OODetector()
        #     logging.info("Train ood detector...")
        #     pca, gmm, centroids, inv_covs, threshold = oodetector.fit(train_embeddings)
            
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