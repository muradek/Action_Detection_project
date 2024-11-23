import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hist_dir = "/home/muradek/project/Action_Detection_project/hist_dir/before_cropping"
src_dir = "/home/muradek/project/Action_Detection_project/data/full_dataset"
csv_list = [f for f in os.listdir(src_dir) if f.endswith('.csv')]
total_videos = len(csv_list)

first_data = pd.read_csv(os.path.join(src_dir, csv_list[0]))
csv_cols = first_data.columns

total_labels = 0 # to make sure summing is correct
count_per_label = {}

for i in range(1, len(csv_cols)):
    col_name = csv_cols[i]
    print("col_name: ", col_name)

    curr_hist = np.zeros(2400)
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
    plt.xlim(left=0, right=2400)
    plt.ylabel("Repitetions(%)")
    plt.ylim(0, 100)
    hist_path = os.path.join(hist_dir, f"{col_name}_histogram.png")
    plt.savefig(hist_path)
    plt.clf()
    
    print(f"total {col_name}: {curr_label_count}")
    print(f"average {col_name}: {(curr_label_count/total_videos)/2400*100}%")
    print("")

colors = plt.cm.tab20(np.linspace(0, 1, len(count_per_label)))

plt.pie(count_per_label.values(), labels=count_per_label.keys(), colors=colors, autopct='%1.1f%%')
plt.title("Labels Distribution Before Cropping")
hist_path = os.path.join(hist_dir, "labels_distribution.png")
plt.tight_layout()
plt.savefig(hist_path)
plt.clf()

print("total videos: ", total_videos)
print("total labels: ", total_labels)
print("all labels avg: ", total_labels/total_videos)
    