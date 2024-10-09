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
    new_dir = dst_dir + "/" + video_name
    os.makedirs(new_dir)
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
            new_frame_path = os.path.join(new_dir, str(frame_count) + ".jpg")
            
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

class FramesDataset(Dataset):
    def __init__(self, src_dir, sample_frequency=100, transform=None):
        self.frames_list = []
        self.labels_list = []
        self.labels_csv_path = None
        self.load_data_from_dir(src_dir, sample_frequency)
        self.transform = transform

    def load_data_from_dir(self, src_dir, sample_frequency):
        # determine type of src_dir
        src_is_video = False
        for filename in os.listdir(src_dir):
            if os.path.isfile(filename) and filename.endswith('.mp4'):
                src_is_video = True
                break

        if src_is_video:
            frames_paths, labels_path = sample_all_videos(src_dir, sample_frequency)
        else:
            frames_paths = os.path.join(src_dir, "frames.csv")
            labels_path = os.path.join(src_dir, "labels.csv")

        self.labels_csv_path = labels_path

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
        
        return frame, label, frame_path

class EmbeddingsDataset(Dataset):
    def __init__(self, src_dir, sequence_length, transform=None):
        self.src_dir = src_dir
        self.sequence_length = sequence_length
        self.one_side_context_frames = (sequence_length - 1) // 2
        self.labels_list = []
        self.video_names = []
        self.video_length = 0
        self.load_data_from_dir(src_dir)

    def load_data_from_dir(self, src_dir):
        # load labels
        labels_path = os.path.join(src_dir, "labels.csv")
        with open(labels_path, mode='r') as labels_file:
            labels_reader = csv.reader(labels_file)
            for label in labels_reader:
                self.labels_list.append(label)

        # load video names
        for filename in os.listdir(src_dir):
            if os.path.isfile(filename) and filename.endswith('.pt'):
                self.video_names.append(filename)

        # load video length
        first_video_path = os.path.join(src_dir, self.video_names[0])
        first_video = torch.load(first_video_path)
        self.video_length = len(first_video)

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):
        curr_video_idx = idx // self.video_length
        curr_video_path = os.path.join(self.src_dir, self.video_names[curr_video_idx])

        # load embeddings
        curr_frame_idx = idx % self.video_length
        lower_bound = max(0, curr_frame_idx - self.one_side_context_frames)
        upper_bound = min(self.video_length, curr_frame_idx + self.one_side_context_frames)
        embeddings = torch.load(curr_video_path)[lower_bound:upper_bound]

        # padding if needed
        prefix_padding_size = max(0, self.one_side_context_frames - curr_frame_idx)
        suffix_padding_size = max(0, self.one_side_context_frames - (self.video_length - curr_frame_idx))
        prefix_padding = torch.zeros(prefix_padding_size, embeddings.size()[1])
        suffix_padding = torch.zeros(suffix_padding_size, embeddings.size()[1])
        prefix_padding = prefix_padding.cuda()
        suffix_padding = suffix_padding.cuda()
        embeddings = torch.cat((prefix_padding, embeddings, suffix_padding), 0)
        
        label = self.labels_list[idx]
        print("labels type: ", type(label))
        print("labels shape: ", label.shape)
        print("labels: ", label)
        # label = torch.tensor(label, dtype=torch.float32)
        
        return embedding, label

def main():
    return 0

if __name__ == "__main__":
    main()