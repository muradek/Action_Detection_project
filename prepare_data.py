import cv2
from PIL import Image
import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def sample_video(video_path, labels_csv_path, sample_frequency=100):
    # prepare frames:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")

    sampled_frames = []
    frame_count = 0
    ret, frame = cap.read() # read the first frame
    while ret:
        if frame_count % sample_frequency == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert frame from BGR to RGB (OpenCV uses BGR by default) is it needed in gray pic?
            pil_image = Image.fromarray(frame_rgb) # Convert to PIL Image
            sampled_frames.append(pil_image)
        frame_count += 1
        ret, frame = cap.read()
    cap.release()

    # prepare labels:
    df = pd.read_csv(labels_csv_path)
    df = df.drop(df.columns[0], axis=1) # drop the first column (labels index)
    sampled_labels = df.iloc[::sample_frequency, :].values

    if len(sampled_frames) != len(sampled_labels):
        raise ValueError("Mismatch between number of sampled frames and sampled labels")

    return sampled_frames, sampled_labels

def sample_all_data(dir_path, sample_frequency):
    videos_paths = [f for f in os.listdir(dir_path) if f.endswith('.mp4')]
    labels_paths = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    all_frames = []
    all_labels = []

    for video, labels in zip(videos_paths, labels_paths):
        abs_video_path = os.path.join(dir_path, video)
        abs_labels_path = os.path.join(dir_path, labels)
        sampled_frames, sampled_labels = sample_video(abs_video_path, abs_labels_path, sample_frequency)
        all_frames.extend(sampled_frames)
        all_labels.extend(sampled_labels)
    
    if len(all_frames) != len(all_labels):
        raise ValueError("Mismatch between number of frames and labels")

    return all_frames, all_labels

class SampledDataset(Dataset):
    def __init__(self, dir_path, sample_frequency, transform=None):
        self.data_path = dir_path
        self.sample_frequency = sample_frequency 
        self.frames, self.labels = sample_all_data(dir_path, sample_frequency)
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]

        if self.transform:
            frame = self.transform(frame)
            # frame = torch.unsqueeze(frame, dim=0) # maybe should be out of if?

        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float32)
        # label = label.unsqueeze(0) #adding a dimention to fit torch.Size([1, 11]) maybe not needed
        
        return frame, label


def main():
    # define the transform method:
    transform = transforms.Compose([
    transforms.Resize((392, 798)),   # Resize image as it needs to be a mulitple of 14
    transforms.ToTensor()])

    # Create dataset and dataloader
    dir_path = "/home/muradek/project/DINO_dir/small_set"
    dataset = SampledDataset(dir_path, 100, transform)
    dataloader = DataLoader(dataset, batch_size=24, shuffle=True, num_workers=2)

if __name__ == "__main__":
    main()
    # output_path = 'frame_1.jpg'
    # cv2.imwrite(output_path, frame)