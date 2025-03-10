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

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def train_model(model, criterion, optimizer, dataloader, num_epochs, epsilon):
    model.train()
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"cuda:{local_rank}")
    
    if dist.get_rank() == 0:
        current_time = datetime.now().strftime("%m-%d_%H:%M")
        print("starting training", current_time)

    # prev_loss = 0
    for epoch in range(num_epochs):
        start_time = datetime.now() 
        losses = []
        for frame, label, _ in dataloader:
            frame, label = frame.to(device, non_blocking=True), label.to(device, non_blocking=True)
            frame.requires_grad = True
            label.requires_grad = True

            optimizer.zero_grad()
            output = model(frame)

            loss = criterion(output, label)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        avg_loss = torch.tensor(sum(losses)/len(losses), device=local_rank) # loss per single GPU
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM) # Aggregate loss across all processes
        avg_loss /= dist.get_world_size() # Normalize by number of GPUs
        if dist.get_rank()==0:
            end_time = datetime.now()
            time_passed = end_time - start_time
            print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.8f}")
            print(f"Epoch [{epoch + 1}/{num_epochs}] took {time_passed}")

        # if abs(prev_loss - avg_loss) < epsilon:
        #     print(f"Converged at epoch {epoch + 1}")
        #     break
        # prev_loss = avg_loss

def train_dino_model(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    for config_name in config.sections():
        print(f"config name: {config_name}")
        src_dir = config[config_name]['src_dir']
        crop_range_str = config[config_name]['crop_range']
        crop_range = [int(i) for i in crop_range_str.split(",")]
        backbone_size = config[config_name]['backbone_size']
        batch_size = config[config_name].getint('batch_size')
        lr = config[config_name].getfloat('lr')
        num_epochs = config[config_name].getint('num_epochs')
        epsilon = config[config_name].getfloat('epsilon')

        if dist.get_rank() == 0:
            for key, value in config[config_name].items():
                print(f'{key} = {value}')

        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

        transform = transforms.Compose([
        transforms.Resize((392, 798)),
        transforms.ToTensor()])

        dataset = FramesDataset(src_dir, crop_range=crop_range, transform=transform)
        if dist.get_rank() == 0:
            total_frames = dataset.__len__()
            print(f"dataset has {total_frames} frames")

        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False, num_workers=2)

        # Instantiate the model, loss function, and optimizer
        model = RawDINOv2(backbone_size).cuda()  # Move model to GPU
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_model(model, criterion, optimizer, dataloader, num_epochs=num_epochs, epsilon=epsilon)
        if dist.get_rank() == 0:
            current_time = datetime.now().strftime("%m-%d_%H:%M")
            print("finished training", current_time)
            model_name = f"finetuned_{backbone_size}_{total_frames}frames_{num_epochs}epochs_{current_time}"
            backbone_path = f"/home/muradek/project/Action_Detection_project/tuned_DINO_models/{model_name}.pth" 
            torch.save(model.state_dict(), backbone_path) 
            print(f"{model_name} saved!")
        return model

def train_lstm_model(backbone_size, src_dir, sequence_length, crop_range, epsilon):
    print("training LSTM model")
    backbone_embeddings = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }

    embedding_dim = backbone_embeddings[backbone_size]
    model = LSTM(embedding_dim=embedding_dim, hidden_size=256, num_layers=3, sequence_length=sequence_length, num_classes=11)
    
    dataset = EmbeddingsDataset(src_dir, sequence_length, crop_range)
    lr = 0.00001
    print("lr: ", lr)
    num_epochs = 4
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
    setup()
    train_dino_model("argsconfig.ini")
    cleanup()
    # model = train_lstm_model()
    # print("currently not training any models")

if __name__ == "__main__":
    main()