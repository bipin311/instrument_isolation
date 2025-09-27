import torch
import numpy as np
import librosa
import soundfile as sf
import os
import stempeg
from tqdm import tqdm

from model.unet import UNet

def separate_bass(model, device, audio_path, output_dir='output_bass'):
    """
    Separates bass from a music file using the trained U-Net model.
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio file properly
        print(f"Loading audio from {audio_path}")
        
        # MUSDB18 has 5 stems:
        # 0 = mix, 1 = drums, 2 = bass, 3 = other, 4 = vocals
        try:
            # Load the full mixture stem correctly
            stems, rate = stempeg.read_stems(audio_path)
            print(f"Loaded {len(stems)} stems, shapes: {[s.shape for s in stems]}")
            
            # Get the mixture (first stem)
            mixture = stems[0]
            print(f"Using mixture stem: shape {mixture.shape}, rate {rate}")
            
            # If we have actual stems, we can also extract the real bass for comparison
            has_real_bass = False
            if len(stems) > 2:
                real_bass = stems[2]  # The bass stem (3rd stem, ID 2)
                print(f"Also extracted real bass: shape {real_bass.shape}")
                has_real_bass = True
            
        except Exception as e:
            print(f"Error loading stems with stempeg: {e}")
            print("Trying alternative loading method...")
            
            # Fallback to librosa if stempeg fails
            mixture, rate = librosa.load(audio_path, sr=44100, mono=False)
            mixture = mixture.T # Transpose to match (samples, channels)
            has_real_bass = False
            print(f"Loaded with alternative method (librosa): shape {mixture.shape}, rate {rate}")
        
        # Convert stereo to mono
        if mixture.ndim > 1 and mixture.shape[1] > 1:
            mixture_mono = np.mean(mixture, axis=1)
            print(f"Converted to mono: shape {mixture_mono.shape}")
        else:
            mixture_mono = mixture.flatten()
            print(f"Audio already mono or flattened: shape {mixture_mono.shape}")
        
        # Set parameters - match what was used in training
        duration = 2  # 2 seconds
        samples_per_chunk = int(duration * rate)
        hop_length = 512
        n_fft = 2048
        
        # Process in 2-second chunks
        total_samples = len(mixture_mono)
        num_chunks = total_samples // samples_per_chunk
        print(f"Total audio length: {total_samples} samples ({total_samples/rate:.2f} seconds)")
        print(f"Processing {num_chunks} chunks of audio for bass separation...")
        
        # Initialize buffer for the full output
        bass_output = np.zeros(total_samples, dtype=np.float32)
        
        # Process each chunk
        for i in tqdm(range(num_chunks)):
            # Extract chunk
            start_sample = i * samples_per_chunk
            end_sample = start_sample + samples_per_chunk
            
            audio_chunk = mixture_mono[start_sample:end_sample]
            chunk_length = len(audio_chunk)
            
            if chunk_length < hop_length * 2:
                continue
                
            # Compute STFT
            stft_chunk = librosa.stft(audio_chunk, n_fft=n_fft, hop_length=hop_length)
            mag_chunk = np.abs(stft_chunk)
            phase_chunk = np.angle(stft_chunk)
            
            # Convert to tensor and send to device
            mag_tensor = torch.from_numpy(mag_chunk).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Get model prediction (mask)
            with torch.no_grad():
                pred_mask = model(mag_tensor)
            
            # Apply mask to the original magnitude
            pred_mask_np = pred_mask.squeeze().cpu().numpy()
            bass_mag = mag_chunk * pred_mask_np
            
            # Reconstruct audio with original phase
            bass_stft = bass_mag * np.exp(1j * phase_chunk)
            bass_audio = librosa.istft(bass_stft, hop_length=hop_length, length=chunk_length)
            
            # Add to output buffer
            bass_output[start_sample:end_sample] = bass_audio
        
        # Save the result
        output_filename = os.path.basename(audio_path).replace('.stem.mp4', '_bass.wav')
        output_path = os.path.join(output_dir, output_filename)
        
        # Normalize audio to a reasonable level
        if np.abs(bass_output).max() > 1e-10:
            bass_output = bass_output / np.abs(bass_output).max() * 0.9
            
        print(f"Final output shape: {bass_output.shape} ({len(bass_output)/rate:.2f} seconds)")
        
        # Save as 32-bit float
        sf.write(output_path, bass_output, rate, subtype='FLOAT')
        print(f"Separated bass saved to {output_path}")
        
        # Create a mixed version (original + isolated bass) to verify results
        mixed_output = np.zeros((total_samples, 2), dtype=np.float32)
        mixed_output[:, 0] = mixture_mono
        mixed_output[:, 1] = bass_output
        
        mixed_path = os.path.join(output_dir, f"mixed_{output_filename}")
        sf.write(mixed_path, mixed_output, rate, subtype='FLOAT')
        print(f"Mixed comparison saved to {mixed_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load the BASS model
    model = UNet(n_channels=1, n_classes=1).to(device)
    try:
        model.load_state_dict(torch.load('checkpoints/unet_bass.pth', map_location=device))
        print("Bass model loaded successfully from checkpoints/unet_bass.pth")
        model.eval()
    except Exception as e:
        print(f"Error loading bass model: {e}")
        return
    
    # Test file
    test_file = 'musdb18/test/Al James - Schoolboy Facination.stem.mp4'
    if not os.path.exists(test_file):
        print(f"Error: Test file not found at {test_file}")
        return
    
    # Separate bass
    separate_bass(model, device, test_file)

if __name__ == '__main__':
    main()