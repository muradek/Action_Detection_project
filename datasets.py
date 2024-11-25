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
import shutil

from models import finetunedDINOv2


def sample_one_video(video_path, labels_csv_path, dst_dir, crop_range):
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
        if frame_count >= crop_range[0] and frame_count < crop_range[1]:
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
    sampled_labels = df.iloc[crop_range[0]:crop_range[1]].values

    if len(frames_paths) != len(sampled_labels):
        print("len frames: ", len(frames_paths))
        print("len labels: ", len(sampled_labels))
        raise ValueError("Mismatch between number of sampled frames and sampled labels")

    return frames_paths, sampled_labels

def sample_all_videos(src_dir, crop_range):
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dst_dir = src_dir + "_sampled_" + current_time 
    os.makedirs(dst_dir)

    videos_paths = [f for f in os.listdir(src_dir) if f.endswith('.mp4')]
    trials_names = [f.removesuffix(".mp4") for f in videos_paths]

    all_frames = []
    all_labels = []
    for trial_name in trials_names:
        video = trial_name + ".mp4"
        labels = trial_name + ".csv"
    # for video, labels in zip(videos_paths, labels_paths):
        abs_video_path = os.path.join(src_dir, video)
        abs_labels_path = os.path.join(src_dir, labels)
        frames_paths, sampled_labels = sample_one_video(abs_video_path, abs_labels_path, dst_dir, crop_range)
        all_frames.extend(frames_paths)
        all_labels.extend(sampled_labels)
    
    if len(all_frames) != len(all_labels):
        print("len all frames: ", len(all_frames))
        print("len all labels: ", len(all_labels))
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
    def __init__(self, src_dir, crop_range=[700,1700], transform=None):
        self.frames_list = []
        self.labels_list = []
        self.labels_csv_path = None
        self.frames_csv_path = None
        self.crop_range = crop_range
        self.num_frames_in_video = crop_range[1] - crop_range[0]
        self.load_data_from_dir(src_dir, crop_range)
        self.transform = transform

    def load_data_from_dir(self, src_dir, crop_range):
        # determine type of src_dir
        src_is_video = False
        for filename in os.listdir(src_dir):
            if (not os.path.isdir(os.path.join(src_dir, filename))) and filename.endswith('.mp4'):
                src_is_video = True
                break

        if src_is_video:
            frames_paths, labels_path = sample_all_videos(src_dir, crop_range)
        else:
            frames_paths = os.path.join(src_dir, "frames.csv")
            labels_path = os.path.join(src_dir, "labels.csv")

        self.labels_csv_path = labels_path
        self.frames_csv_path = frames_paths

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
    def __init__(self, src_dir, sequence_length, crop_range):
        current_time = datetime.now().strftime("%m-%d_%H:%M")
        print("starting dataset", current_time)

        self.src_dir = src_dir
        self.sequence_length = sequence_length
        self.one_side_window_len = (sequence_length - 1) // 2
        self.crop_range = crop_range
        self.video_length = crop_range[1] - crop_range[0]
        self.labels_list = []
        self.frames_list = []
        self.labels_csv_path = None
        self.frames_csv_path = None
        self.load_data_from_dir(src_dir)

        current_time = datetime.now().strftime("%m-%d_%H:%M")
        print("Finished dataset", current_time)

    def load_data_from_dir(self, src_dir):
        labels_path = os.path.join(src_dir, "labels.csv")
        frames_paths = os.path.join(src_dir, "frames.csv")
        self.labels_csv_path = labels_path
        self.frames_csv_path = frames_paths

        # load labels and embeddings
        with open(frames_paths, mode='r') as frames_file:
            frames_reader = csv.reader(frames_file)
            with open(labels_path, mode='r') as labels_file:
                labels_reader = csv.reader(labels_file)
                for frame, label in zip(frames_reader, labels_reader):
                    frame_str = "".join(frame)
                    label_int_list = [eval(i) for i in label] # maybe not needed
                    self.frames_list.append(frame_str)
                    self.labels_list.append(label_int_list)

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):
        # get embedding path and index
        frame_path = self.frames_list[idx]
        split_path = frame_path.split("/")
        frame_initial_idx = int(split_path[-1].split(".")[0])
        embedding_idx = frame_initial_idx - self.crop_range[0]
        embedding_file_name = split_path[-2].replace("mp4", "pt")
        embedding_path = os.path.join(self.src_dir, embedding_file_name)

        # load embeddings
        lower_bound = max(0, embedding_idx - self.one_side_window_len)
        upper_bound = min(self.video_length, embedding_idx + self.one_side_window_len + 1)
        embeddings = torch.load(embedding_path)[lower_bound:upper_bound]

        # padding if needed
        prefix_padding_size = max(0, self.one_side_window_len - embedding_idx)
        suffix_padding_size = max(0, self.one_side_window_len - (self.video_length - embedding_idx) + 1)
        prefix_padding = torch.zeros(prefix_padding_size, embeddings.size()[1])
        suffix_padding = torch.zeros(suffix_padding_size, embeddings.size()[1])
        embeddings = torch.cat((prefix_padding, embeddings, suffix_padding), 0)

        label = self.labels_list[idx]
        tensor_label = torch.tensor(label, dtype=torch.float32)
        return embeddings, tensor_label, embedding_file_name

# create embeddings from a video/frames and save them in a local directory
def create_embeddings(backbone_size, state_dict_path, src_dir, dst_dir, crop_range):
    model = finetunedDINOv2(backbone_size, state_dict_path) 
    print("model loaded")
    transform = transforms.Compose([
    transforms.Resize((392, 798)),   # Resize image as it needs to be a mulitple of 14
    transforms.ToTensor()])
    # print("memory after model: ", torch.cuda.memory_allocated() / (1024**3))

    dataset = FramesDataset(src_dir, crop_range=crop_range, transform=transform)
    # print("memory after dataset: ", torch.cuda.memory_allocated() / (1024**3))

    print(f"datast loaded with {dataset.__len__()} frames")
    dataloader = DataLoader(dataset, batch_size=24, shuffle=False, num_workers=2)
    # print("memory after dataloader: ", torch.cuda.memory_allocated() / (1024**3)) 
    print("dataloader loaded")
    current_time = datetime.now().strftime("%m-%d_%H:%M")
    src_dir_name = src_dir.split("/")[-1]
    root_dir = f"{dst_dir}/{current_time}_{src_dir_name}_{dataset.__len__()}embeddings"
    os.makedirs(root_dir)

    # copy the csv files to the root directory
    shutil.copy(dataset.labels_csv_path, root_dir)
    shutil.copy(dataset.frames_csv_path, root_dir)

    # save the embeddings in the root directory
    all_embeddings = []
    frame_paths = []
    with torch.no_grad():
        # i = 1
        for frame_batch, _, frame_path_batch in dataloader:
            frame_batch = frame_batch.cuda()
            # print("after frame to cuda: ", torch.cuda.memory_allocated() / (1024**3))
            embeddings_batch = model(frame_batch).cpu()
            # print("after model(frame): ", torch.cuda.memory_allocated() / (1024**3))
            all_embeddings.append(embeddings_batch)
            frame_paths.extend(frame_path_batch)
            # torch.cuda.empty_cache()
            # print(f"batch {i} done")
            # i += 1
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # create a list of the video names
    video_names = []
    for frame_path in frame_paths:
        video_name = frame_path.split(".mp4")[0]
        video_name = video_name.split("/")[-1]
        if video_name not in video_names:
            video_names.append(video_name)

    frames_per_video = int(dataset.__len__() / len(video_names))
    split_embeddings = torch.split(all_embeddings, frames_per_video)
    for video_name, embedding in zip(video_names, split_embeddings):
        video_pt = f"{root_dir}/{video_name}.pt"
        torch.save(embedding, video_pt)
        
    return root_dir

def main():
    # save local embeddings
    # backbone_size = "base"
    # state_dict_path = "/home/muradek/project/Action_Detection_project/tuned_models/finetuned_base_18000frames_20epochs.pth"
    # src_dir = "/home/muradek/project/Action_Detection_project/data/small_set_sampled_2024-10-01_00:49:24"
    # dst_dir = "/home/muradek/project/Action_Detection_project/embeddings"
    # root_dir = create_embeddings(backbone_size, state_dict_path, src_dir, dst_dir, crop_range=[700,1700)
    # print(f"root directory is {root_dir}")

    # load embeddings to dataset
    root_dir = "/home/muradek/project/Action_Detection_project/embeddings/10-09_14:37"
    embeddings_dataset = EmbeddingsDataset(root_dir, sequence_length=21)

    return 0

if __name__ == "__main__":
    main()