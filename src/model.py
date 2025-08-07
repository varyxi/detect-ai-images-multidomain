from typing import Tuple

import torch
import pandas as pd
from torch import nn
from transformers import CLIPModel

class CLIPFeatureExtractor:
    def __init__(self, model_name, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).eval().to(self.device)

    @torch.no_grad()
    def get_image_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device)
        return self.model.get_image_features(pixel_values=pixel_values)


class MLPClassifier(nn.Module):
    def __init__(self, num_classes=2, input_dim=512, hidden_dim=256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=2, hidden_dim=256):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, pixel_values):
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
        return self.classifier(image_features)