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


# This function creates a DINO model with the specified backbone size
def create_dino_model(backbone_size):
    # REPO_PATH = "/home/muradek/project/DINO_dir/dinov2" # Specify a local path to the repository (or use installed package instead)
    # sys.path.append(REPO_PATH)
    
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

        self.labels_head = nn.Sequential(nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Linear(256, out_dim))
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

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size=256, num_layers=2, sequence_length=15, num_classes=11):
        super(LSTM, self).__init__()
        print("hidden size: ", hidden_size)
        print("num layers: ", num_layers)
        print("sequence length: ", sequence_length)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, num_classes)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, x):
        # num layers * num directions(2=bidirectional), batch, hidden size
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        mid_idx = self.sequence_length//2
        out = self.fc(out[:, mid_idx, :]) # out.shape=torch.Size([24, 11])
        # for training return "probabilities" (loss uses softmax)
        if self.training:
            return out
        # for testing return final predictions (argmax)
        one_hots = torch.argmax(out, dim=1) # one_hots.shape=torch.Size([24])
        return one_hots


class TwoStreamModel(nn.Module):
    def __init__(self, backbone_size, lstm_hidden_size=256, lstm_num_layers=2, sequence_length=15, num_classes=11):
        super(TwoStreamModel, self).__init__()

        # DINO Stream (Visual Features)
        self.dino = finetunedDINOv2(backbone_size, state_dict_path="path_to_finetunedDINO")
        dino_embedding_dim = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }[backbone_size]

        # LSTM Stream (Temporal Features)
        self.lstm = LSTM(embedding_dim=dino_embedding_dim, hidden_size=lstm_hidden_size,
                         num_layers=lstm_num_layers, sequence_length=sequence_length, num_classes=num_classes)

        # Fusion Layer (Concatenation + FC)
        self.fc_fusion = nn.Sequential(
            nn.Linear(2 * num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, frames, sequences):

        # DINO Forward Pass (Visual Features)
        visual_features = self.dino(frames)  # (batch_size, dino_embedding_dim)
        visual_out = torch.softmax(visual_features, dim=-1)  # (batch_size, num_classes)

        # LSTM Forward Pass (Temporal Features)
        temporal_out = self.lstm(sequences)  # (batch_size, num_classes)

        # Concatenation + Final Prediction
        combined_features = torch.cat((visual_out, temporal_out), dim=1)  # (batch_size, 2*num_classes)
        final_out = self.fc_fusion(combined_features)  # (batch_size, num_classes)

        return final_out


def main():
    return 0
    
if __name__ == "__main__":
    main()