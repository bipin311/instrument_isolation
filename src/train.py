import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from data.musdb_dataset import MUSDBDataset
from model.unet import UNet

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for mixture, vocals in tqdm(train_loader, desc="Training"):
        mixture, vocals = mixture.to(device), vocals.to(device)

        optimizer.zero_grad()
        output = model(mixture)
        loss = criterion(output, vocals)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * mixture.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def main():
    # Hyperparameters
    batch_size = 4 # Reduced batch size for memory constraints
    epochs = 10 # A small number of epochs for a quick test
    lr = 1e-4
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Dataset and Dataloader
    train_dataset = MUSDBDataset(root_dir='musdb18/train', duration=2, samples_per_track=16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model, Criterion, Optimizer
    model = UNet(n_channels=1, n_classes=1).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        train_loss = train(model, device, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/unet_vocals.pth')
    print("Model saved to checkpoints/unet_vocals.pth")

if __name__ == '__main__':
    main()
