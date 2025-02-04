import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import configparser
from datetime import datetime
import os
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models import *
from datasets import *

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique idetifier of each process
        world_size: Total number of process
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def train_model(model, criterion, optimizer, dataloader, num_epochs, epsilon, rank, world_size):
    model.train()
    current_time = datetime.now().strftime("%m-%d_%H:%M")
    print("starting training", current_time)
    prev_loss = 0
    for epoch in range(num_epochs):
        dataloader.sampler.set_epoch(epoch)  # Shuffle data for each epoch in DDP  
        losses = []
        for frame, label, _ in dataloader:
            frame, label = frame.cuda(), label.cuda()
            frame.requires_grad = True
            label.requires_grad = True

            optimizer.zero_grad()
            output = model(frame)

            loss = criterion(output, label)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Synchronize and calculate average loss across processes
        loss_tensor = torch.tensor(sum(losses) / len(losses), device=rank)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size
        if rank == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.8f}")
        if abs(prev_loss - avg_loss) < epsilon:
            print(f"Converged at epoch {epoch + 1}")
            break
        prev_loss = avg_loss

    current_time = datetime.now().strftime("%m-%d_%H:%M")
    print("finished training", current_time)

def train_dino_model(config_file, rank, world_size):
    ddp_setup(rank, world_size)

    config = configparser.ConfigParser()
    config.read(config_file)

    for config_name in config.sections():
        print(f"config name: {config_name}")
        src_dir = config[config_name]['src_dir']
        crop_range_str = config[config_name]['crop_range']
        crop_range = [int(i) for i in crop_range_str.split(",")]
        backbone_size = config[config_name]['backbone_size']
        batch_size = config[config_name].getint('batch_size') // world_size
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

        dataset = FramesDataset(src_dir, crop_range=crop_range, transform=transform)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        total_frames = dataset.__len__()
        print(f"dataset has {total_frames} frames")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # Instantiate the model, loss function, and optimizer
        model = RawDINOv2(backbone_size)
        model = DDP(model, device_ids=[rank])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_model(model, criterion, optimizer, dataloader, num_epochs, epsilon, rank, world_size)
        print("finished training")

        if rank == 0:
            current_time = datetime.now().strftime("%m-%d_%H:%M")
            model_name = f"finetuned_{backbone_size}_{total_frames}frames_{num_epochs}epochs_{current_time}"
            backbone_path = f"/home/shahafb/Action_Detection_project/Tuned_DINO_models/{model_name}.pth" 
            torch.save(model.module.state_dict(), backbone_path) 
            print(f"{model_name} saved!")
        
    destroy_process_group()

def train_lstm_model(backbone_size, src_dir, sequence_length, crop_range, epsilon, rank, world_size):
    print("training LSTM model")
    backbone_embeddings = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }
    embedding_dim = backbone_embeddings[backbone_size]
    model = LSTM(embedding_dim=embedding_dim, hidden_size=256, num_layers=3, sequence_length=sequence_length, num_classes=11)
    model = DDP(model, device_ids=[gpu_id])
    dataset = EmbeddingsDataset(src_dir, sequence_length, crop_range)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    lr = 0.00001
    print("lr: ", lr)
    num_epochs = 4
    dataloader = DataLoader(dataset, batch_size=(24 // world_size), shuffle=False, sampler=sampler, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # mp.set_start_method('spawn', force=True)
    train_model(model, criterion, optimizer, dataloader, num_epochs, epsilon, rank, world_size)

    print("finished training")

    if rank == 0:
        current_time = datetime.now().strftime("%m-%d_%H:%M")
        total_embeddings = dataset.__len__()
        model_name = f"LSTM_{backbone_size}_{total_embeddings}embeddings_{num_epochs}epochs_{current_time}"
        backbone_path = f"/home/shahafb/Action_Detection_project/Tuned_LSTM_models/{model_name}.pth" 
        torch.save(model.module.state_dict(), backbone_path) 
        print(f"{model_name} saved!")

    destroy_process_group()

def main():
    #model = train_dino_model("argsconfig.ini")
    #model = train_lstm_model()
    world_size = torch.cuda.device_count()
    mp.spawn(train_dino_model_ddp, args=( "argsconfig.ini", world_size), nprocs=world_size, join=True)
        
    backbone_size = "small"  # Example, replace with your desired size
    src_dir = "/home/shahafb/Action_Detection_project/data/small_dataset"  # Replace with your source directory
    sequence_length = 10  # Example sequence length
    crop_range = [0, 2399]  # Example crop range
    epsilon = 1e-4

    mp.spawn(
        train_lstm_model_ddp,
        args=(world_size, backbone_size, src_dir, sequence_length, crop_range, epsilon),
        nprocs=world_size,
        join=True,
    )
    print("currently not training any models")

if __name__ == "__main__":
    main()