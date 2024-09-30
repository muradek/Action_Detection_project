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

    for config_name in config.sections():
        print(f"config name: {config_name}")
        src_dir = config[config_name]['src_dir']
        sample_frequency = config[config_name].getint('sample_frequency')
        backbone_size = config[config_name]['backbone_size']
        batch_size = config[config_name].getint('batch_size')
        lr = config[config_name].getfloat('lr')
        num_epochs = config[config_name].getint('num_epochs')

        for key, value in config[config_name].items():
            print(f'{key} = {value}')
        
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

        # # Define the transform method:
        transform = transforms.Compose([
        transforms.Resize((392, 798)),   # Resize image as it needs to be a mulitple of 14
        transforms.ToTensor()])

        # prepare_data2.py
        dataset = SampledDataset(src_dir, sample_frequency=sample_frequency, transform=transform)
        print(f"dataset has {dataset.__len__()} frames")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        # Instantiate the model, loss function, and optimizer
        model = Dinov2Tune(backbone_size)
        model = torch.compile(model)
        print("compiled the model!")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_model(model, criterion, optimizer, dataloader, num_epochs = num_epochs)
        print("finished training. saving the model:")
        backbone_path = "/home/muradek/project/Action_Detection_project/finetuned_backbone_2.pth"
        torch.save(model.backbone_model.state_dict(), backbone_path) 
        # need to make sure:
        # 1. this is only the DINOv2 without the FC layers
        # 2. the backbone was updated in backpropagation and its not the original backbone 
        print("model saved!")

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

3. Freezing DINO
for param in model.backbone_model.parameters():
    param.requires_grad = False


for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

"""