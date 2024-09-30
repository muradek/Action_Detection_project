import sys
import os
import random
import zipfile
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms


class Dinov2Tune(nn.Module):
    def __init__(self, backbone_size):
        super(Dinov2Tune, self).__init__()

        REPO_PATH = "/home/muradek/project/DINO_dir/dinov2" # Specify a local path to the repository (or use installed package instead)
        sys.path.append(REPO_PATH)
        
        possible_sizes = ["small", "base", "large", "giant"]
        if not(backbone_size in possible_sizes):
            print("backbone size is invalid")

        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[backbone_size]
        backbone_name = f"dinov2_{backbone_arch}"
        
        backbone_embeddings = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }

        embedding_dim = backbone_embeddings[backbone_size]
        out_dim = 11 # number of classes for detection

        self.backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)

        self.labels_head = nn.Sequential(nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Linear(256, out_dim), nn.Softmax(dim=1))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, frame):
        features = self.backbone_model(frame)
        labels_prob = self.labels_head(features)
        return labels_prob

def main():
    return 0
    
if __name__ == "__main__":
    main()