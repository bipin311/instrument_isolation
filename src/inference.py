import torch
import numpy as np
import librosa
import soundfile as sf
import os
import stempeg
from tqdm import tqdm

from model.unet import UNet

def separate_vocals(model, device, audio_path, output_dir='output'):
    """
    Separates vocals from a music file using the trained U-Net model.
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio file properly
        print(f"Loading audio from {audio_path}")
        
        # MUSDB18 has 5 stems:
        # 0 = mix, 1 = drums, 2 = bass, 3 = other, 4 = vocals
        # We want to use stem 0 (the mixture) as input
        try:
            # Load the full mixture stem correctly
            stems, rate = stempeg.read_stems(audio_path)
            print(f"Loaded {len(stems)} stems, shapes: {[s.shape for s in stems]}")
            
            # Get the mixture (first stem)
            mixture = stems[0]  # This should be stereo, shape (samples, 2)
            print(f"Using mixture stem: shape {mixture.shape}, rate {rate}")
            
            # If we have actual stems, we can also extract the real vocals for comparison
            has_real_vocals = False
            if len(stems) > 4:
                real_vocals = stems[4]  # The vocals stem (5th stem)
                print(f"Also extracted real vocals: shape {real_vocals.shape}")
                has_real_vocals = True
            
        except Exception as e:
            print(f"Error loading stems with stempeg: {e}")
            print("Trying alternative loading method...")
            
            import subprocess
            from pydub import AudioSegment
            
            # Create a temporary wav file
            temp_wav = "temp_mixture.wav"
            subprocess.run(["ffmpeg", "-i", audio_path, "-y", temp_wav], 
                          stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            
            # Load with pydub
            audio = AudioSegment.from_wav(temp_wav)
            rate = audio.frame_rate
            mixture = np.array(audio.get_array_of_samples()).reshape(-1, audio.channels)
            mixture = mixture.astype(np.float32) / 32768.0  # Convert to float
            
            # Clean up
            os.remove(temp_wav)
            has_real_vocals = False
            
            print(f"Loaded with alternative method: shape {mixture.shape}, rate {rate}")
        
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
        print(f"Processing {num_chunks} chunks of audio...")
        
        # Initialize buffer for the full output
        vocals_output = np.zeros(total_samples, dtype=np.float32)
        
        # Process each chunk
        for i in tqdm(range(num_chunks)):
            # Extract chunk
            start_sample = i * samples_per_chunk
            end_sample = start_sample + samples_per_chunk
            
            # Ensure we don't go out of bounds
            if end_sample > total_samples:
                end_sample = total_samples
                
            audio_chunk = mixture_mono[start_sample:end_sample]
            chunk_length = len(audio_chunk)
            
            # Skip processing if chunk is too short
            if chunk_length < hop_length * 2:
                continue
                
            # Compute STFT
            stft_chunk = librosa.stft(audio_chunk, n_fft=n_fft, hop_length=hop_length)
            mag_chunk = np.abs(stft_chunk)
            phase_chunk = np.angle(stft_chunk)
            
            # Debug info for first chunk only
            if i == 0:
                print(f"STFT shape: {stft_chunk.shape}")
                print(f"Magnitude shape: {mag_chunk.shape}")
            
            # Convert to tensor and send to device
            mag_tensor = torch.from_numpy(mag_chunk).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Get model prediction (mask)
            with torch.no_grad():
                pred_mask = model(mag_tensor)
            
            # Debug info for first chunk only
            if i == 0:
                print(f"Model output shape: {pred_mask.shape}")
                print(f"Model output min: {pred_mask.min().item()}, max: {pred_mask.max().item()}")
            
            # Apply mask to the original magnitude
            pred_mask_np = pred_mask.squeeze().cpu().numpy()
            vocals_mag = mag_chunk * pred_mask_np
            
            # Reconstruct audio with original phase
            vocals_stft = vocals_mag * np.exp(1j * phase_chunk)
            vocals_audio = librosa.istft(vocals_stft, hop_length=hop_length, length=chunk_length)
            
            # Add to output buffer
            vocals_output[start_sample:end_sample] = vocals_audio
        
        # Save the result
        output_filename = os.path.basename(audio_path).replace('.stem.mp4', '_vocals.wav')
        output_path = os.path.join(output_dir, output_filename)
        
        # Normalize audio to a reasonable level
        if np.abs(vocals_output).max() > 1e-10:
            vocals_output = vocals_output / np.abs(vocals_output).max() * 0.9
            
        print(f"Final output shape: {vocals_output.shape} ({len(vocals_output)/rate:.2f} seconds)")
        print(f"Final output min: {vocals_output.min()}, max: {vocals_output.max()}")
        
        # Save as 32-bit float
        sf.write(output_path, vocals_output, rate, subtype='FLOAT')
        print(f"Vocals saved to {output_path}")
        
        # Create a mixed version (original + isolated vocals) to verify results
        mixed_output = np.zeros((total_samples, 2), dtype=np.float32)  # Stereo
        mixed_output[:, 0] = mixture_mono  # Left channel: original
        mixed_output[:, 1] = vocals_output  # Right channel: isolated vocals
        
        mixed_path = os.path.join(output_dir, f"mixed_{output_filename}")
        sf.write(mixed_path, mixed_output, rate, subtype='FLOAT')
        print(f"Mixed comparison saved to {mixed_path}")
        
        # If we have the real vocals, create a comparison with them too
        if has_real_vocals:
            real_vocals_mono = np.mean(real_vocals, axis=1) if real_vocals.ndim > 1 else real_vocals
            real_mixed_output = np.zeros((total_samples, 3), dtype=np.float32)  # 3 channels
            real_mixed_output[:, 0] = mixture_mono  # Channel 1: original
            real_mixed_output[:, 1] = vocals_output  # Channel 2: predicted vocals
            real_mixed_output[:, 2] = real_vocals_mono  # Channel 3: real vocals
            
            real_compare_path = os.path.join(output_dir, f"compare_{output_filename}")
            sf.write(real_compare_path, real_mixed_output[:, :2], rate, subtype='FLOAT')  # Save just the first 2 channels
            print(f"Comparison with real vocals saved to {real_compare_path}")
        
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
    
    # Load model
    model = UNet(n_channels=1, n_classes=1).to(device)
    try:
        model.load_state_dict(torch.load('checkpoints/unet_vocals.pth', map_location=device))
        print("Model loaded successfully")
        
        # Set model to evaluation mode
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test file
    test_file = 'musdb18/test/Al James - Schoolboy Facination.stem.mp4'
    if not os.path.exists(test_file):
        print(f"Error: Test file not found at {test_file}")
        # Try to find any .stem.mp4 file in the test directory
        test_dir = 'musdb18/test'
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                if file.endswith('.stem.mp4'):
                    test_file = os.path.join(test_dir, file)
                    print(f"Using alternative test file: {test_file}")
                    break
    
    # Separate vocals
    separate_vocals(model, device, test_file)

if __name__ == '__main__':
    main()