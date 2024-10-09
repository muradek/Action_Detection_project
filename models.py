import sys
import os
import random
import zipfile
from copy import deepcopy
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

from datasets import FramesDataset


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
    backbone_size = "base"
    state_dict_path = "/home/muradek/project/Action_Detection_project/tuned_models/finetuned_base_18000frames_20epochs.pth"
    model = finetunedDINOv2(backbone_size, state_dict_path) 

    transform = transforms.Compose([
    transforms.Resize((392, 798)),   # Resize image as it needs to be a mulitple of 14
    transforms.ToTensor()])

    src_dir = "/home/muradek/project/Action_Detection_project/data/small_set_sampled_2024-10-01_00:49:24"
    sample_frequency = 100
    dataset = FramesDataset(src_dir, sample_frequency=sample_frequency, transform=transform)
    print(f"dataset has {dataset.__len__()} frames")
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)  

    current_time = datetime.now().strftime("%m-%d_%H:%M")
    root_dir = f"/home/muradek/project/Action_Detection_project/embeddings/{current_time}"
    os.makedirs(root_dir)

    # copy the labels csv file to the root directory
    shutil.copy(dataset.labels_csv_path, root_dir)

    # save the embeddings in the root directory
    all_embeddings = []
    frame_paths = []
    with torch.no_grad():
        for frame_batch, _, frame_path_batch in dataloader:
            frame_batch = frame_batch.cuda()
            embeddings_batch = model(frame_batch)
            print(f"embedding batch shape is {embeddings_batch.shape}")
            all_embeddings.append(embeddings_batch)
            print("all_embeddings shape is ", len(all_embeddings))
            frame_paths.extend(frame_path_batch)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"all_embeddings after cat shape is {all_embeddings.shape}")
    split_embeddings = torch.split(all_embeddings, 24)

    # create a list of the video names
    video_names = []
    for frame_path in frame_paths:
        video_name = frame_path.split(".mp4")[0]
        video_name = video_name.split("/")[-1]
        if video_name not in video_names:
            video_names.append(video_name)

    print("video names are ", video_names)
    frames_per_video = int(dataset.__len__() / len(video_names))
    split_embeddings = torch.split(all_embeddings, frames_per_video)
    for video_name, embedding in zip(video_names, split_embeddings):
        video_pt = f"{root_dir}/{video_name}.pt"
        torch.save(embedding, video_pt)
        print("final embedding shape is ", embedding.shape)
        
    return 0
    
if __name__ == "__main__":
    main()