import cv2
from PIL import Image
import pandas as pd
import numpy as np
import os
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def sample_video(video_path, labels_csv_path, dst_dir, sample_frequency=100):
    video_name = video_path.rsplit('/', 1)[-1] #get the name of the video without path prefix

    # prepare frames:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")

    frames_paths = []
    frame_count = 0
    ret, frame = cap.read() # read the first frame
    while ret:
        if frame_count % sample_frequency == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert frame from BGR to RGB (OpenCV uses BGR by default) is it needed in gray pic?
            pil_image = Image.fromarray(frame_rgb) # Convert to PIL Image
            new_frame_path = os.path.join(dst_dir, video_name + str(frame_count) + ".jpg")
            
            pil_image.save(new_frame_path)
            frames_paths.append(new_frame_path)
        frame_count += 1
        ret, frame = cap.read()
    cap.release()

    # prepare labels:
    df = pd.read_csv(labels_csv_path)
    df = df.drop(df.columns[0], axis=1) # drop the first column (labels index)
    sampled_labels = df.iloc[::sample_frequency, :].values

    if len(frames_paths) != len(sampled_labels):
        raise ValueError("Mismatch between number of sampled frames and sampled labels")

    return frames_paths, sampled_labels

def sample_all_data(src_dir, dst_dir, sample_frequency):
    videos_paths = [f for f in os.listdir(src_dir) if f.endswith('.mp4')]
    labels_paths = [f for f in os.listdir(src_dir) if f.endswith('.csv')]
    all_frames = []
    all_labels = []

    for video, labels in zip(videos_paths, labels_paths):
        abs_video_path = os.path.join(src_dir, video)
        abs_labels_path = os.path.join(src_dir, labels)
        frames_paths, sampled_labels = sample_video(abs_video_path, abs_labels_path, dst_dir, sample_frequency)
        all_frames.extend(frames_paths)
        all_labels.extend(sampled_labels)
    
    if len(all_frames) != len(all_labels):
        raise ValueError("Mismatch between number of frames and labels")

    frames_csv_path = os.path.join(dst_dir, "frames.csv")
    with open(frames_csv_path, mode='w', newline='') as frames_csv:
        labels_csv_path = os.path.join(dst_dir, "labels.csv")
        with open(labels_csv_path, mode='w', newline='') as labels_csv:
            frames_writer = csv.writer(frames_csv)
            labels_writer = csv.writer(labels_csv)

            for frame, label in zip(all_frames, all_labels):
                frames_writer.writerow(frame)
                labels_writer.writerow(label)
    return frames_csv_path, labels_csv_path

class SampledDataset(Dataset):
    def __init__(self, frames_paths, labels_path, transform=None):

        self.frames_list = []
        self.labels_list = []
        with open(frames_paths, mode='r') as frames_file:
            frames_reader = csv.reader(frames_file)
            with open(labels_path, mode='r') as labels_file:
                labels_reader = csv.reader(labels_file)
                for frame, label in zip(frames_reader, labels_reader):
                    frame_str = "".join(frame)
                    label_int_list = [eval(i) for i in label]
                    self.frames_list.append(frame_str)
                    self.labels_list.append(label_int_list)
        self.transform = transform

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        frame_path = self.frames_list[idx]
        frame = Image.open(frame_path)

        if self.transform:
            frame = self.transform(frame)
            # frame = torch.unsqueeze(frame, dim=0) # maybe should be out of if?

        label = self.labels_list[idx]
        label = torch.tensor(label, dtype=torch.float32)
        # label = label.unsqueeze(0) #adding a dimention to fit torch.Size([1, 11]) maybe not needed
        
        return frame, label


def main():
    # # define the transform method:
    # transform = transforms.Compose([
    # transforms.Resize((392, 798)),   # Resize image as it needs to be a mulitple of 14
    # transforms.ToTensor()])

    # # Create dataset and dataloader
    # dir_path = "/home/muradek/project/DINO_dir/small_set"
    # dataset = SampledDataset(dir_path, 100, transform)
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    src_dir = "/home/muradek/project/Action_Detection_project/small_set"
    dst_dir = "/home/muradek/project/Action_Detection_project/new_format"

    frames_csv, labels_csv = sample_all_data(src_dir, dst_dir, sample_frequency = 100)
    dataset = SampledDataset(frames_csv, labels_csv)

if __name__ == "__main__":
    main()
    # output_path = 'frame_1.jpg'
    # cv2.imwrite(output_path, frame)