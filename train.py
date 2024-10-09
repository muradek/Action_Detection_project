import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import configparser
from datetime import datetime
import os
import torch.multiprocessing as mp

from models import *
from datasets import *


def train_model(model, criterion, optimizer, dataloader, num_epochs):
    for epoch in range(num_epochs):  
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

def train_dino_model(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

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

        transform = transforms.Compose([
        transforms.Resize((392, 798)),
        transforms.ToTensor()])

        dataset = FramesDataset(src_dir, sample_frequency=sample_frequency, transform=transform)
        total_frames = dataset.__len__()
        print(f"dataset has {total_frames} frames")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        # Instantiate the model, loss function, and optimizer
        model = RawDINOv2(backbone_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_model(model, criterion, optimizer, dataloader, num_epochs = num_epochs)
        print("finished training")

        current_time = datetime.now().strftime("%m-%d_%H:%M")
        model_name = f"finetuned_{backbone_size}_{total_frames}frames_{num_epochs}epochs_{current_time}"
        backbone_path = f"/home/muradek/project/Action_Detection_project/tuned_models/{model_name}.pth" 
        torch.save(model.state_dict(), backbone_path) 
        print(f"{model_name} saved!")
        return model

def train_lstm_model(backbone_size="base"):
    backbone_embeddings = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }

    embedding_dim = backbone_embeddings[backbone_size]
    model = LSTM(embedding_dim=embedding_dim, hidden_size=512, num_layers=3, sequence_length=21, num_classes=11)
    
    src_dir = "/home/muradek/project/Action_Detection_project/embeddings/10-09_14:37"
    sequence_length = 21
    dataset = EmbeddingsDataset(src_dir, sequence_length)
    lr = 0.001
    num_epochs = 20
    dataloader = DataLoader(dataset, batch_size=24, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # mp.set_start_method('spawn', force=True)
    train_model(model, criterion, optimizer, dataloader, num_epochs = num_epochs)


def main():
    # model = train_dino_model("argsconfig.ini")
    model = train_lstm_model()
    print("finished training")

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
dataset = FramesDataset(src_dir, sample_frequency, transform)

3. Freezing DINO
for param in model.backbone_model.parameters():
    param.requires_grad = False


for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

"""