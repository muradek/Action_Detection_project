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


def train_model(model, criterion, optimizer, dataloader, num_epochs, epsilon):
    current_time = datetime.now().strftime("%m-%d_%H:%M")
    print("starting training", current_time)
    prev_loss = 0
    for epoch in range(num_epochs):  
        losses = []
        for frame, label in dataloader:
            frame, label = frame.cuda(), label.cuda()
            frame.requires_grad = True
            label.requires_grad = True

            optimizer.zero_grad()
            output = model(frame)

            loss = criterion(output, label)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        avg_loss = sum(losses)/len(losses)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.8f}")
        if abs(prev_loss - avg_loss) < epsilon:
            print(f"Converged at epoch {epoch + 1}")
            break
        prev_loss = avg_loss

    current_time = datetime.now().strftime("%m-%d_%H:%M")
    print("finished training", current_time)

def train_dino_model(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    for config_name in config.sections():
        print(f"config name: {config_name}")
        src_dir = config[config_name]['src_dir']
        crop_point = config[config_name].getint('crop_point')
        backbone_size = config[config_name]['backbone_size']
        batch_size = config[config_name].getint('batch_size')
        lr = config[config_name].getfloat('lr')
        num_epochs = config[config_name].getint('num_epochs')
        epsilon = config[config_name].getfloat('epsilon')

        for key, value in config[config_name].items():
            print(f'{key} = {value}')
        
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

        transform = transforms.Compose([
        transforms.Resize((392, 798)),
        transforms.ToTensor()])

        dataset = FramesDataset(src_dir, crop_point=crop_point, transform=transform)
        total_frames = dataset.__len__()
        print(f"dataset has {total_frames} frames")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        # Instantiate the model, loss function, and optimizer
        model = RawDINOv2(backbone_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_model(model, criterion, optimizer, dataloader, num_epochs=num_epochs, epsilon=epsilon)
        print("finished training")

        current_time = datetime.now().strftime("%m-%d_%H:%M")
        model_name = f"finetuned_{backbone_size}_{total_frames}frames_{num_epochs}epochs_{current_time}"
        backbone_path = f"/home/muradek/project/Action_Detection_project/tuned_DINO_models/{model_name}.pth" 
        torch.save(model.state_dict(), backbone_path) 
        print(f"{model_name} saved!")
        return model

def train_lstm_model(backbone_size, src_dir, sequence_length, epsilon):
    print("training LSTM model")
    backbone_embeddings = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }

    embedding_dim = backbone_embeddings[backbone_size]
    model = LSTM(embedding_dim=embedding_dim, hidden_size=512, num_layers=3, sequence_length=sequence_length, num_classes=11)
    
    dataset = EmbeddingsDataset(src_dir, sequence_length)
    lr = 0.00001
    print("lr: ", lr)
    num_epochs = 10
    dataloader = DataLoader(dataset, batch_size=24, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # mp.set_start_method('spawn', force=True)
    train_model(model, criterion, optimizer, dataloader, num_epochs=num_epochs, epsilon=epsilon)

    print("finished training")

    current_time = datetime.now().strftime("%m-%d_%H:%M")
    total_embeddings = dataset.__len__()
    model_name = f"LSTM_{backbone_size}_{total_embeddings}embeddings_{num_epochs}epochs_{current_time}"
    backbone_path = f"/home/muradek/project/Action_Detection_project/tuned_LSTM_models/{model_name}.pth" 
    torch.save(model.state_dict(), backbone_path) 
    print(f"{model_name} saved!")
    return model


def main():
    # model = train_dino_model("argsconfig.ini")
    # model = train_lstm_model()
    print("currently not training any models")

if __name__ == "__main__":
    main()