import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import glob
import stempeg
import librosa
import numpy as np
import random

from model.unet import UNet

class MUSDBBassDataset(Dataset):
    """
    A PyTorch Dataset to load mixture and bass stems from the MUSDB18 dataset.
    """
    def __init__(self, root_dir, duration=2, samples_per_track=16):
        self.root_dir = root_dir
        self.duration = duration
        self.samples_per_track = samples_per_track
        self.track_paths = glob.glob(os.path.join(self.root_dir, '*.stem.mp4'))
        if not self.track_paths:
            raise FileNotFoundError(f"No .stem.mp4 files found in {self.root_dir}")
        print(f"Found {len(self.track_paths)} tracks for bass training.")

    def __len__(self):
        return len(self.track_paths) * self.samples_per_track

    def __getitem__(self, idx):
        track_idx = idx // self.samples_per_track
        track_path = self.track_paths[track_idx]

        # Load mixture (stem 0) and bass (stem 2)
        stems, rate = stempeg.read_stems(track_path, stem_id=[0, 2])
        mixture_stereo = stems[0]
        bass_stereo = stems[1]

        # Convert to mono
        mixture_mono = np.mean(mixture_stereo, axis=1)
        bass_mono = np.mean(bass_stereo, axis=1)

        # Take a random 2-second crop
        segment_samples = int(self.duration * rate)
        if len(mixture_mono) > segment_samples:
            start_sample = random.randint(0, len(mixture_mono) - segment_samples)
            end_sample = start_sample + segment_samples
            mixture_mono = mixture_mono[start_sample:end_sample]
            bass_mono = bass_mono[start_sample:end_sample]

        # Compute STFT
        mixture_stft = librosa.stft(mixture_mono, n_fft=2048, hop_length=512)
        bass_stft = librosa.stft(bass_mono, n_fft=2048, hop_length=512)

        # Get magnitudes and convert to tensor
        mixture_mag = torch.from_numpy(np.abs(mixture_stft)).float().unsqueeze(0)
        bass_mag = torch.from_numpy(np.abs(bass_stft)).float().unsqueeze(0)

        return mixture_mag, bass_mag

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for mixture, bass in tqdm(train_loader, desc="Training Bass Model"):
        mixture, bass = mixture.to(device), bass.to(device)

        optimizer.zero_grad()
        output = model(mixture)
        loss = criterion(output, bass)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * mixture.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def main():
    # Hyperparameters (same as vocals)
    batch_size = 4
    epochs = 10
    lr = 1e-4
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Dataset and Dataloader
    train_dataset = MUSDBBassDataset(root_dir='musdb18/train', duration=2, samples_per_track=16)
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
    torch.save(model.state_dict(), 'checkpoints/unet_bass.pth')
    print("Bass separation model saved to checkpoints/unet_bass.pth")

if __name__ == '__main__':
    main()