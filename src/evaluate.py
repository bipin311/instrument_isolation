import torch
import numpy as np
import librosa
import soundfile as sf
import os
import stempeg
from tqdm import tqdm
import glob
import time
import argparse
import csv
from datetime import datetime
from typing import Optional

from model.unet import UNet
from inference import separate_vocals

def evaluate_model(
    model,
    device,
    test_dir: str = 'musdb18/test',
    output_dir: str = 'evaluation_results',
    log_dir: Optional[str] = 'logs',
    model_tag: str = 'unet_vocals',
    limit: Optional[int] = None,
    save_comparison: bool = True,
    skip_existing: bool = True,
):
    """Evaluate vocals separation across MUSDB18 test set and log results.

    Parameters:
    - model, device: Loaded model and compute device.
    - test_dir: Directory containing `.stem.mp4` test files.
    - output_dir: Where separated vocals and comparisons are written.
    - log_dir: Directory to write CSV/JSON logs (set None to disable logging).
    - model_tag: Identifier for the model (used in logs).
    - limit: Optional cap on number of files (process all if None or <=0).
    - save_comparison: Save side-by-side predicted vs. ground-truth vocals WAV.
    - skip_existing: If True, skip separation when output file already exists.
    """
    os.makedirs(output_dir, exist_ok=True)

    test_files = sorted(glob.glob(os.path.join(test_dir, '*.stem.mp4')))
    if not test_files:
        print(f"No test files found in {test_dir}")
        return []

    if limit and limit > 0:
        test_files = test_files[:limit]

    print(f"Found {len(test_files)} test files in {test_dir}")

    # Prepare logging
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    csv_path = None
    json_path = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, f'eval_vocals_{model_tag}_{timestamp}.csv')
        json_path = os.path.join(log_dir, f'eval_vocals_summary_{model_tag}_{timestamp}.json')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp','model_tag','dataset_split','file','duration_seconds','rms','peak',
                'processing_time_seconds','sdr_db','output_path','comparison_path'
            ])

    results = []

    for i, test_file in enumerate(test_files):
        print(f"\n[{i+1}/{len(test_files)}] Processing {os.path.basename(test_file)}")

        output_filename = os.path.basename(test_file).replace('.stem.mp4', '_vocals.wav')
        output_path = os.path.join(output_dir, output_filename)

        # Skip if already exists and requested
        if skip_existing and os.path.exists(output_path):
            print(f"Skipping separation (exists): {output_path}")
            processing_time = 0.0
        else:
            start_time = time.time()
            separate_vocals(model, device, test_file, output_dir=output_dir)
            processing_time = time.time() - start_time

        # Evaluate separated output
        try:
            separated_vocals, sr = librosa.load(output_path, sr=None)
            duration = float(len(separated_vocals) / sr) if sr else None
            rms = float(np.sqrt(np.mean(separated_vocals**2))) if len(separated_vocals) else None
            peak = float(np.max(np.abs(separated_vocals))) if len(separated_vocals) else None

            sdr = None
            comparison_path = None
            if save_comparison:
                try:
                    stems, _ = stempeg.read_stems(test_file)
                    if len(stems) > 4:
                        real_vocals = stems[4]
                        real_vocals_mono = np.mean(real_vocals, axis=1) if real_vocals.ndim > 1 else real_vocals
                        min_len = min(len(separated_vocals), len(real_vocals_mono))
                        if min_len > 0:
                            sep_cut = separated_vocals[:min_len]
                            ref_cut = real_vocals_mono[:min_len]
                            sdr = float(calculate_sdr(ref_cut, sep_cut))
                            comparison = np.zeros((min_len, 2), dtype=np.float32)
                            comparison[:, 0] = ref_cut
                            comparison[:, 1] = sep_cut
                            comparison_path = os.path.join(output_dir, f"comparison_{output_filename}")
                            sf.write(comparison_path, comparison, sr, subtype='FLOAT')
                except Exception as e:
                    print(f"Couldn't calculate metrics/comparison: {e}")

            result = {
                'timestamp': timestamp,
                'model_tag': model_tag,
                'dataset_split': 'musdb18/test',
                'file': os.path.basename(test_file),
                'duration_seconds': duration,
                'rms': rms,
                'peak': peak,
                'processing_time_seconds': processing_time,
                'sdr_db': sdr,
                'output_path': output_path,
                'comparison_path': comparison_path,
            }

            results.append(result)

            # Append to CSV
            if csv_path:
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        result['timestamp'], result['model_tag'], result['dataset_split'], result['file'],
                        result['duration_seconds'], result['rms'], result['peak'],
                        result['processing_time_seconds'], result['sdr_db'],
                        result['output_path'], result['comparison_path']
                    ])

            # Console summary per track
            dur_str = f"{duration:.2f}s" if duration is not None else "n/a"
            rms_str = f"{rms:.4f}" if rms is not None else "n/a"
            peak_str = f"{peak:.4f}" if peak is not None else "n/a"
            print(f"Duration: {dur_str}, RMS: {rms_str}, Peak: {peak_str}")
            if sdr is not None:
                print(f"SDR: {sdr:.2f} dB")
            print(f"Processing time: {processing_time:.2f}s")

        except Exception as e:
            print(f"Error evaluating output for {output_filename}: {e}")

    # Aggregate summary
    print("\n=== Evaluation Summary ===")
    print(f"Processed {len(results)} songs")
    summary = {
        'timestamp': timestamp,
        'model_tag': model_tag,
        'dataset_split': 'musdb18/test',
        'count': len(results),
        'avg_duration_seconds': float(np.mean([r['duration_seconds'] for r in results if r['duration_seconds'] is not None])) if results else None,
        'avg_processing_time_seconds': float(np.mean([r['processing_time_seconds'] for r in results if r['processing_time_seconds'] is not None])) if results else None,
        'avg_rms': float(np.mean([r['rms'] for r in results if r['rms'] is not None])) if results else None,
        'avg_sdr_db': float(np.mean([r['sdr_db'] for r in results if r['sdr_db'] is not None])) if results else None,
        'csv_path': csv_path,
    }

    if summary['avg_duration_seconds'] is not None:
        print(f"Average duration: {summary['avg_duration_seconds']:.2f} s")
    if summary['avg_processing_time_seconds'] is not None:
        print(f"Average processing time: {summary['avg_processing_time_seconds']:.2f} s")
    if summary['avg_rms'] is not None:
        print(f"Average RMS: {summary['avg_rms']:.4f}")
    if summary['avg_sdr_db'] is not None:
        print(f"Average SDR: {summary['avg_sdr_db']:.2f} dB")

    # Write JSON summary
    if json_path:
        try:
            import json as _json
            with open(json_path, 'w') as f:
                _json.dump({'summary': summary, 'results': results}, f, indent=2)
            print(f"Logs written: {csv_path}")
            print(f"Summary written: {json_path}")
        except Exception as e:
            print(f"Failed to write summary JSON: {e}")

    return results


def calculate_sdr(reference, estimation):
    """
    Calculate Signal-to-Distortion Ratio in dB.
    A simple implementation for mono signals.
    
    Higher values are better.
    """
    # Small epsilon to avoid division by zero
    eps = np.finfo(np.float64).eps
    
    # Calculate signal energy
    signal_energy = np.sum(reference**2)
    
    # Calculate noise energy
    noise = estimation - reference
    noise_energy = np.sum(noise**2)
    
    # Calculate SDR
    sdr = 10 * np.log10(signal_energy / (noise_energy + eps))
    
    return sdr


def main():
    parser = argparse.ArgumentParser(description="Evaluate vocals model over MUSDB18 test set and log results.")
    parser.add_argument('--test-dir', type=str, default='musdb18/test', help='Directory containing MUSDB18 test .stem.mp4 files')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Directory to write separated outputs and comparisons')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory to write CSV/JSON logs (set to empty to disable)')
    parser.add_argument('--model-tag', type=str, default='unet_vocals', help='Identifier for the evaluated model (used in logs)')
    parser.add_argument('--limit', type=int, default=0, help='Process only first N files (0 = all)')
    parser.add_argument('--no-save-comparison', action='store_true', help='Do not save comparison WAV with ground-truth vocals')
    parser.add_argument('--no-skip-existing', action='store_true', help='Re-run separation even if output exists')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/unet_vocals.pth', help='Path to model checkpoint')
    args = parser.parse_args()

    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    model = UNet(n_channels=1, n_classes=1).to(device)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Model loaded from {args.checkpoint}")
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    log_dir = None if not args.log_dir or args.log_dir.strip() == '' else args.log_dir
    limit = None if args.limit is None or args.limit <= 0 else args.limit

    evaluate_model(
        model,
        device,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        log_dir=log_dir,
        model_tag=args.model_tag,
        limit=limit,
        save_comparison=(not args.no_save_comparison),
        skip_existing=(not args.no_skip_existing),
    )


if __name__ == '__main__':
    main()