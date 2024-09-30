import cv2
from PIL import Image
import pandas as pd
import numpy as np
import os
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datetime import datetime


def sample_one_video(video_path, labels_csv_path, dst_dir, sample_frequency):
    video_name = video_path.rsplit('/', 1)[-1] # get the name of the video without path prefix

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

def sample_all_videos(src_dir, sample_frequency):
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dst_dir = src_dir + "_sampled_" + current_time 
    os.makedirs(dst_dir)

    videos_paths = [f for f in os.listdir(src_dir) if f.endswith('.mp4')]
    labels_paths = [f for f in os.listdir(src_dir) if f.endswith('.csv')]
    all_frames = []
    all_labels = []

    for video, labels in zip(videos_paths, labels_paths):
        abs_video_path = os.path.join(src_dir, video)
        abs_labels_path = os.path.join(src_dir, labels)
        frames_paths, sampled_labels = sample_one_video(abs_video_path, abs_labels_path, dst_dir, sample_frequency)
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
    def __init__(self, src_dir, sample_frequency=100, transform=None):
        self.frames_list = []
        self.labels_list = []
        self.load_data_from_dir(src_dir, sample_frequency)
        self.transform = transform

    def load_data_from_dir(self, src_dir, sample_frequency):
        # determine type of src_dir
        src_is_video = False
        src_is_jpg = False
        for filename in os.listdir(src_dir):
            if filename.endswith('.mp4'):
                src_is_video = True
                break
            elif filename.endswith('.jpg'):
                src_is_jpg = True
                break

        if src_is_video:
            frames_paths, labels_path = sample_all_videos(src_dir, sample_frequency)
        elif src_is_jpg:
            frames_paths = os.path.join(src_dir, "frames.csv")
            labels_path = os.path.join(src_dir, "labels.csv")
        else:
            print("ERROR: src_dir format")
            exit(0)

        with open(frames_paths, mode='r') as frames_file:
            frames_reader = csv.reader(frames_file)
            with open(labels_path, mode='r') as labels_file:
                labels_reader = csv.reader(labels_file)
                for frame, label in zip(frames_reader, labels_reader):
                    frame_str = "".join(frame)
                    label_int_list = [eval(i) for i in label]
                    self.frames_list.append(frame_str)
                    self.labels_list.append(label_int_list)

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        frame_path = self.frames_list[idx]
        frame = Image.open(frame_path)

        if self.transform:
            frame = self.transform(frame) # maybe transform all frames in the first load
            
        label = self.labels_list[idx]
        label = torch.tensor(label, dtype=torch.float32)
        
        return frame, label


def main():
    return 0

if __name__ == "__main__":
    main()