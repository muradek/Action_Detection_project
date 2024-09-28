import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import configparser

from model import Dinov2Tune
from prepare_data2 import SampledDataset

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
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.8f}")

def main():
    config = configparser.ConfigParser()
    config.read('argsconfig.ini')

    config_type = 'DEFAULT' # in ['DEFAULT',...]
    src_dir = config[config_type]['src_dir']
    sample_frequency = config[config_type].getint('sample_frequency')
    backbone_size = config[config_type]['backbone_size']
    batch_size = config[config_type].getint('batch_size')
    lr = config[config_type].getfloat('lr')
    num_epochs = config[config_type].getint('num_epochs')

    for key, value in config[config_type].items():
        print(f'{key} = {value}')
    
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # # Define the transform method:
    transform = transforms.Compose([
    transforms.Resize((392, 798)),   # Resize image as it needs to be a mulitple of 14
    transforms.ToTensor()])

    # prepare_data2.py
    dataset = SampledDataset(src_dir, sample_frequency=sample_frequency, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Instantiate the model, loss function, and optimizer
    model = Dinov2Tune(backbone_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_model(model, criterion, optimizer, dataloader, num_epochs = num_epochs)
    

if __name__ == "__main__":
    main()

"""
code snippets
1. Memory management
print(torch.cuda.memory_summary())

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))


2. pipelines (prepare_data.py)
src_dir = "/home/muradek/project/Action_Detection_project/small_set"
sample_frequency = 100
dataset = SampledDataset(src_dir, sample_frequency, transform)

"""