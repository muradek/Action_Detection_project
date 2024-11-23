import torch
import os
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def print_hists(src_dir, hist_dir):
    csv_list = [f for f in os.listdir(src_dir) if f.endswith('.csv')]
    total_videos = len(csv_list)

    first_data = pd.read_csv(os.path.join(src_dir, csv_list[0]))
    num_frames = first_data.shape[0]
    print("num_frames: ", num_frames)
    csv_cols = first_data.columns

    total_labels = 0 # to make sure summing is correct
    count_per_label = {}

    for i in range(1, len(csv_cols)):
        col_name = csv_cols[i]
        print("col_name: ", col_name)
        curr_hist = np.zeros(num_frames)
        for csv in csv_list:
            data = pd.read_csv(os.path.join(src_dir, csv))
            curr_hist += data.iloc[:, i].values

        curr_label_count = sum(curr_hist)
        count_per_label[col_name] = curr_label_count
        total_labels += curr_label_count

        curr_hist = (curr_hist / total_videos)*100
        indices = np.arange(len(curr_hist))
        
        plt.plot(indices, curr_hist)
        plt.title(f"Histogram of {col_name}")
        plt.xlabel("Frame")
        plt.xlim(left=0, right=num_frames)
        plt.ylabel("Repitetions(%)")
        plt.ylim(0, 100)
        hist_path = os.path.join(hist_dir, f"{col_name}_histogram.png")
        plt.savefig(hist_path)
        plt.clf()
        
        print(f"total {col_name}: {curr_label_count}")
        print(f"average {col_name}: {(curr_label_count/total_videos)/num_frames*100}%")
        print("")

    colors = plt.cm.tab20(np.linspace(0, 1, len(count_per_label)))

    plt.pie(count_per_label.values(), labels=count_per_label.keys(), colors=colors, autopct='%1.1f%%')
    plt.title("Labels Distribution Before Cropping")
    hist_path = os.path.join(hist_dir, "PIE.png")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.clf()

    print("total videos: ", total_videos)
    print("total labels: ", total_labels)
    print("all labels avg: ", total_labels/total_videos)


def copy_files(src_dir, video_list, dst_dir):
    for video in video_list:
        video_path = os.path.join(src_dir, video)
        csv_path = os.path.join(src_dir, video.replace('.mp4', '.csv'))
        if os.path.exists(video_path):
            shutil.copy(video_path, dst_dir)
        if os.path.exists(csv_path):
            shutil.copy(csv_path, dst_dir)

 
def split_data(src_dir, train_dir, test_dir, train_size, test_size):
    video_files = [f for f in os.listdir(src_dir) if f.endswith('.mp4')]
    random.shuffle(video_files)

    # Allocate videos for training and testing
    train_videos = video_files[:train_size]
    test_videos = video_files[train_size : train_size + test_size]

    copy_files(src_dir, train_videos, train_dir)
    copy_files(src_dir, test_videos, test_dir)

    print(f"Allocated {len(train_videos)} videos to {train_dir}")
    print(f"Allocated {len(test_videos)} videos to {test_dir}")
    print("Done!")

if __name__ == "__main__":
    src_dir = "/home/muradek/project/Action_Detection_project/data/full_dataset"
    hist_dir = "/home/muradek/project/Action_Detection_project/hist_dir/before_cropping"
    print_hists(src_dir, hist_dir)