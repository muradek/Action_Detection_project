import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from model import create_model
from prepare_data import SampledDataset

import os


def train_model(model, criterion, optimizer, dataloader, num_epochs):
    for epoch in range(num_epochs):  # Number of epochs
        # print(torch.cuda.memory_summary())
        losses = []
        for frame, label in dataloader:
            frame, label = frame.cuda(), label.cuda()
            frame.requires_grad = True # is this the right place to put this?
            label.requires_grad = True

            optimizer.zero_grad()
            output = model(frame)

            loss = criterion(output, label)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        avg_loss = sum(losses)/len(losses)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

def main():
    # print("entered main: ")
    # print(torch.cuda.memory_summary())
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # # Define the transform method:
    transform = transforms.Compose([
    transforms.Resize((392, 798)),   # Resize image as it needs to be a mulitple of 14
    transforms.ToTensor()])

    # prepare_data.py
    src_dir = "/home/muradek/project/Action_Detection_project/small_set"
    sample_frequency = 100
    dataset = SampledDataset(src_dir, sample_frequency, transform)

    # prepare_data2.py
    # frames_paths = "/home/muradek/project/Action_Detection_project/new_format/frames.csv"
    # labels_path = "/home/muradek/project/Action_Detection_project/new_format/labels.csv"
    # dataset = SampledDataset(frames_paths, labels_path, transform=transform)

    print("total frames number is: ", dataset.__len__())
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    # Instantiate the model, loss function, and optimizer
    model_size = "base" # in ["small", "base", "large", "giant"]
    model = create_model(model_size)

    # param_size = 0
    # for param in model.parameters():
    #     param_size += param.nelement() * param.element_size()
    # buffer_size = 0
    # for buffer in model.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()

    # size_all_mb = (param_size + buffer_size) / 1024**2
    # print('model size: {:.3f}MB'.format(size_all_mb))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("starting training")
    train_model(model, criterion, optimizer, dataloader, num_epochs = 5)

if __name__ == "__main__":
    main()