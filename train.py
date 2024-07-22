import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from fine_tune import create_model, Dinov2Tune
from prepare_data import SampledDataset


def train_model(model, criterion, optimizer, dataloader, num_epochs):
    for epoch in range(num_epochs):  # Number of epochs
        for frame, label in dataloader:
            frame, label = frame.cuda(), label.cuda()
            optimizer.zero_grad()
            output = model(frame)

            frame.requires_grad = True # is this the right place to put this?
            label.requires_grad = True

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}")

def main():
    # Define the transform method:
    transform = transforms.Compose([
    transforms.Resize((392, 798)),   # Resize image as it needs to be a mulitple of 14
    transforms.ToTensor()])

    # Create dataset and dataloader
    dir_path = "/home/muradek/project/DINO_dir/small_set"
    sample_frequency = 20
    dataset = SampledDataset(dir_path, sample_frequency, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # Instantiate the model, loss function, and optimizer
    model = create_model()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, criterion, optimizer, dataloader, num_epochs = 10)
    print("finished :)")


if __name__ == "__main__":
    main()