import torch
import numpy as np
import os
import glob
from model.unet import UNet
from inference import separate_vocals

def main(song_index=2):  # Default to song #2
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = UNet(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load('checkpoints/unet_vocals.pth', map_location=device))
    print("Model loaded successfully")
    model.eval()
    
    # Find test songs
    test_dir = 'musdb18/test'
    test_files = glob.glob(f"{test_dir}/*.stem.mp4")
    
    if not test_files:
        print(f"No test files found in {test_dir}")
        return
    
    # List all available songs
    print("\nAvailable songs:")
    for i, file_path in enumerate(test_files):
        song_name = os.path.basename(file_path).replace('.stem.mp4', '')
        print(f"[{i+1}] {song_name}")
    
    # Make sure song_index is valid
    if song_index < 1 or song_index > len(test_files):
        print(f"Invalid song index {song_index}. Using song #1 instead.")
        song_index = 1
    
    # Select the song by index
    selected_file = test_files[song_index - 1]
    song_name = os.path.basename(selected_file).replace('.stem.mp4', '')
    
    # Create output directory
    output_dir = 'selected_song_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the selected song
    print(f"\nProcessing {song_name}...")
    
    separate_vocals(model, device, selected_file, output_dir=output_dir)
    
    # Get output paths
    output_filename = os.path.basename(selected_file).replace('.stem.mp4', '_vocals.wav')
    output_path = os.path.join(output_dir, output_filename)
    mixed_path = os.path.join(output_dir, f"mixed_{output_filename}")
    
    print("\nProcessing complete!")
    print(f"Separated vocals saved to: {output_path}")
    print(f"Mixed comparison saved to: {mixed_path}")

if __name__ == '__main__':
    import sys
    
    # Get song index from command line argument if provided
    song_index = 2  # Default to song #2
    if len(sys.argv) > 1:
        try:
            song_index = int(sys.argv[1])
        except ValueError:
            print("Invalid song index. Using default (song #2).")
    
    main(song_index)