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


# This function creates a DINO model with the specified backbone size
def create_dino_model(backbone_size):
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

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    
    return backbone_model


# This model contains the *original* DINOv2 model with a FC layer on top of it
# used for finetuning the model
class RawDINOv2(nn.Module):
    def __init__(self, backbone_size):
        super(RawDINOv2, self).__init__()
        self.backbone_model = create_dino_model(backbone_size)
        
        backbone_embeddings = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }

        embedding_dim = backbone_embeddings[backbone_size]
        out_dim = 11 # number of classes for detection

        self.labels_head = nn.Sequential(nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Linear(256, out_dim), nn.Softmax(dim=1))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, frame):
        features = self.backbone_model(frame)
        labels_prob = self.labels_head(features)
        return labels_prob

# This model contains the *fine-tuned* DINOv2 model
# used to extract embeddings from the dataset
class finetunedDINOv2(nn.Module):
    def __init__(self, backbone_size, state_dict_path):
        super(finetunedDINOv2, self).__init__()
        self.backbone_model = create_dino_model(backbone_size)

        state_dict = torch.load(state_dict_path)    
        self.load_state_dict(state_dict, strict=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, frame):
        embedding = self.backbone_model(frame)
        return embedding

def main():
    model = finetunedDINOv2("base", "/home/muradek/project/Action_Detection_project/tuned_models/finetuned_09-30_20:28.pth")
    
    frame_path = "/home/muradek/project/Action_Detection_project/data/small_set_sampled_2024-09-28_19:03:47/CT54_22_06_08_tuft_control_trial0001.mp40.jpg"
    transform = transforms.Compose([
        transforms.Resize((392, 798)),
        transforms.ToTensor()])

    transformed_frame = transform(Image.open(frame_path))
    final_img = torch.unsqueeze(transformed_frame, dim=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_img = final_img.to(device)

    with torch.inference_mode():
        embedding = model(transformed_frame)
    print(embedding.shape)
    return 0
    
if __name__ == "__main__":
    main()