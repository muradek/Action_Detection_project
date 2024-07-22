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
    def __init__(self, backbone_model, out_dim):
        super(Dinov2Tune, self).__init__()
        self.backbone_model = deepcopy(backbone_model)
        self.labels_head = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, out_dim))
        # is this enough layers? need to make sure that the head is big enough so that the last layers of dino will output features
        # and do not change to "detection layers"

    def forward(self, frame):
        features = self.backbone_model(frame)
        # print("features size is: ", features.size())
        labels_prob = self.labels_head(features)
        # print("labels size is: ", labels_prob.size())
        # with torch.no_grad(): # this is a non differentiable operation
            # one_hot_vector = F.one_hot(torch.argmax(labels_prob), num_classes=labels_prob.size(0)).float()
        max_index = torch.argmax(labels_prob)
        output = torch.zeros_like(labels_prob)
        output[0, max_index] = 1

        # print("output size is: ", output.size())
        return output

def create_model():
    REPO_PATH = "/home/muradek/project/DINO_dir/dinov2" # Specify a local path to the repository (or use installed package instead)
    sys.path.append(REPO_PATH)
    
    BACKBONE_SIZE = "large" # in ("small", "base", "large" or "giant")

    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    # backbone_model.cuda()

    out_dim = 11 # number of classes for detection
    model = Dinov2Tune(backbone_model, out_dim) # maybe pass reference of the model, or model args and consruct it in the __init__?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device) 
    
    return model

def main():
    REPO_PATH = "/home/muradek/project/DINO_dir/dinov2" # Specify a local path to the repository (or use installed package instead)
    sys.path.append(REPO_PATH)
    
    BACKBONE_SIZE = "large" # in ("small", "base", "large" or "giant")

    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    # backbone_model.cuda()

    out_dim = 11 # number of classes for detection
    model = Dinov2Tune(backbone_model, out_dim) # maybe pass reference of the model, or model args and consruct it in the __init__?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # load frame for inference
    frame_path = "/home/muradek/project/DINO_dir/dino/frames/frame-0012.jpg"
    frame = Image.open(frame_path) 

    # Todo: this should be changed to a transform module!!!
    new_height = (frame.height // 14) * 14  # Round down to the nearest multiple of 14
    new_width = (frame.width // 14) * 14
    resized_image = frame.resize((new_width, new_height))
    transform = transforms.Compose([transforms.ToTensor(),]) # Converts the image to a tensor
    # Apply the transform to the image
    image_tensor = transform(resized_image)
    final_img = torch.unsqueeze(image_tensor, dim=0)
    final_img = final_img.to(device)

    #inference
    # print("img device is: ", final_img.device)
    # print("model device is: ", model.device)
    # print("model device is: ", model.backbone_model.device)
    # print("model device is: ", model.labels_head.device)

    with torch.inference_mode():
        output = model(final_img) 
    print(output)

if __name__ == "__main__":
    main()