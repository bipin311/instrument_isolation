import os
import argparse
import subprocess
import sys
import torch

def check_demucs_installed():
    """Checks if the demucs library is installed."""
    try:
        import demucs
        return True
    except ImportError:
        print("`demucs` library not found.")
        print("Please install it by running: pip install -U demucs")
        # For MPS (Apple Silicon) support, you might need a specific PyTorch version.
        print("Ensure you have a compatible PyTorch version installed.")
        return False

def separate_with_demucs(audio_path, output_dir='output_demucs'):
    """
    Separates an audio file into its sources using the pre-trained Demucs model.
    This function calls the demucs command-line interface, which is the most
    stable and recommended way to use the tool.
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting source separation for: {audio_path}")
    print(f"Output will be saved in: {output_dir}")

    # The command to run demucs.
    # We use sys.executable to ensure we use the same Python environment.
    # -o specifies the output directory.
    # -n htdemucs selects the default hybrid transformer model, which is excellent.
    # The last argument is the input file path.
    command = [
        sys.executable,
        "-m", "demucs.separate",
    ]
    
    # For 4-stem separation (vocals, bass, drums, other), use this command instead:
    # command = [
    #     sys.executable,
    #     "-m", "demucs.separate",
    #     "-o", str(output_dir),
    #     # "-n", "htdemucs_ft", # Fine-tuned model, often better
    #     str(audio_path)
    # ]

    # Add device flag if not using CPU
    if torch.cuda.is_available():
        print("CUDA device found, using GPU.")
    elif torch.backends.mps.is_available():
        print("MPS device found, using Apple Silicon GPU.")
        command.extend(["-d", "mps"]) # Add --mps flag for Apple Silicon
    else:
        print("No GPU found, using CPU. This might be slow.")

    command.extend([
        "-o", str(output_dir),
        str(audio_path)  # This is the required 'tracks' argument
    ])

    try:
        # Run the demucs separation command
        print(f"\nRunning command: {' '.join(command)}\n")
        # Using subprocess.run to execute the command
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        print("--- Demucs Output ---")
        print(result.stdout)
        print("---------------------")

        print("\nSeparation successful!")
        # Demucs creates a subdirectory inside the output folder, e.g., output_demucs/htdemucs/
        print(f"Separated files are located in a subfolder within '{output_dir}'.")

    except FileNotFoundError:
        print("Error: 'python' command not found.")
        print("Please ensure Python is in your system's PATH.")
    except subprocess.CalledProcessError as e:
        print("\n--- ERROR ---")
        print("Demucs process failed.")
        print("Error output:")
        print(e.stderr)
        print("-------------")

def main():
    # First, check if demucs is installed
    if not check_demucs_installed():
        return

    parser = argparse.ArgumentParser(description="Separate audio sources using the Demucs library.")
    parser.add_argument(
        "audio_file", 
        type=str, 
        help="Path to the audio file you want to separate (e.g., 'musdb18/test/song.stem.mp4')."
    )
    args = parser.parse_args()

    separate_with_demucs(args.audio_file)

if __name__ == '__main__':
    main()