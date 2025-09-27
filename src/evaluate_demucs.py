import os
import glob
import time
import argparse
import csv
from datetime import datetime
from typing import Optional

import numpy as np
import librosa
import soundfile as sf
import stempeg

from demucs_inference import check_demucs_installed, separate_with_demucs


def calculate_sdr(reference, estimation):
    eps = np.finfo(np.float64).eps
    signal_energy = np.sum(reference**2)
    noise = estimation - reference
    noise_energy = np.sum(noise**2)
    sdr = 10 * np.log10(signal_energy / (noise_energy + eps))
    return sdr


def find_demucs_output(base_out: str, input_path: str, model_name_hint: Optional[str] = None) -> Optional[str]:
    """Locate Demucs output folder for a given input based on typical structure.

    Demucs usually writes to: <base_out>/<model_name>/<track_name>/[stems].wav
    """
    track_basename = os.path.splitext(os.path.basename(input_path))[0]
    # Some demucs variants include dots in model dir; search one level deep
    candidates = glob.glob(os.path.join(base_out, '*', track_basename))
    if model_name_hint:
        candidates = [c for c in candidates if os.path.basename(os.path.dirname(c)) == model_name_hint] or candidates
    return candidates[0] if candidates else None


def evaluate_demucs(
    test_dir: str = 'musdb18/test',
    demucs_out_dir: str = 'output_demucs',
    evaluation_dir: str = 'evaluation_demucs',
    log_dir: Optional[str] = 'logs',
    model_tag: str = 'demucs',
    target: str = 'vocals',  # one of ['vocals','bass','drums','other']
    limit: Optional[int] = None,
    overwrite: bool = False,
):
    if not check_demucs_installed():
        print("Demucs not installed. Install with: pip install -U demucs")
        return []

    os.makedirs(evaluation_dir, exist_ok=True)
    test_files = sorted(glob.glob(os.path.join(test_dir, '*.stem.mp4')))
    if not test_files:
        print(f"No test files found in {test_dir}")
        return []
    if limit and limit > 0:
        test_files = test_files[:limit]
    print(f"Found {len(test_files)} test files in {test_dir}")

    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    csv_path = None
    json_path = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, f'eval_{target}_{model_tag}_{timestamp}.csv')
        json_path = os.path.join(log_dir, f'eval_{target}_summary_{model_tag}_{timestamp}.json')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp','model_tag','dataset_split','file','duration_seconds','rms','peak',
                'processing_time_seconds','sdr_db','output_path'
            ])

    # Map target -> demucs stem filename
    stem_name = {
        'vocals': 'vocals.wav',
        'bass': 'bass.wav',
        'drums': 'drums.wav',
        'other': 'other.wav',
    }[target]

    results = []
    for i, test_file in enumerate(test_files):
        print(f"\n[{i+1}/{len(test_files)}] Processing {os.path.basename(test_file)}")

        # Run Demucs once per input; it writes all stems
        start_time = time.time()
        separate_with_demucs(test_file, output_dir=demucs_out_dir)
        processing_time = time.time() - start_time

        # Locate Demucs outputs
        out_folder = find_demucs_output(demucs_out_dir, test_file)
        if not out_folder:
            print(f"Could not find Demucs output folder for {test_file}")
            continue
        target_path = os.path.join(out_folder, stem_name)
        if not os.path.exists(target_path):
            print(f"Missing Demucs {target} stem at {target_path}")
            continue

        # Copy/normalize to evaluation folder (optional: re-save)
        out_name = os.path.basename(test_file).replace('.stem.mp4', f'_{target}.wav')
        eval_out_path = os.path.join(evaluation_dir, out_name)

        # Load, compute metrics, and save normalized copy for consistency
        try:
            audio, sr = librosa.load(target_path, sr=None, mono=True)
            duration = float(len(audio) / sr) if sr else None
            rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else None
            peak = float(np.max(np.abs(audio))) if len(audio) else None

            # Normalize peak to 0.9, similar to U-Net outputs
            if peak and peak > 1e-10:
                audio_norm = audio / peak * 0.9
            else:
                audio_norm = audio

            if overwrite or not os.path.exists(eval_out_path):
                sf.write(eval_out_path, audio_norm, sr, subtype='FLOAT')

            # Compute SDR vs ground-truth stem
            sdr = None
            try:
                stems, _ = stempeg.read_stems(test_file)
                # MUSDB stem ids: 0 mix, 1 drums, 2 bass, 3 other, 4 vocals
                idx = {'drums':1, 'bass':2, 'other':3, 'vocals':4}[target]
                ref = stems[idx]
                ref_mono = np.mean(ref, axis=1) if ref.ndim > 1 else ref
                min_len = min(len(audio_norm), len(ref_mono))
                if min_len > 0:
                    sdr = float(calculate_sdr(ref_mono[:min_len], audio_norm[:min_len]))
            except Exception as e:
                print(f"Couldn't compute SDR: {e}")

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
                'output_path': eval_out_path,
            }
            results.append(result)

            # Log CSV row
            if csv_path:
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        result['timestamp'], result['model_tag'], result['dataset_split'], result['file'],
                        result['duration_seconds'], result['rms'], result['peak'],
                        result['processing_time_seconds'], result['sdr_db'], result['output_path']
                    ])

            # Print short summary
            dur_str = f"{duration:.2f}s" if duration is not None else "n/a"
            rms_str = f"{rms:.4f}" if rms is not None else "n/a"
            peak_str = f"{peak:.4f}" if peak is not None else "n/a"
            print(f"Duration: {dur_str}, RMS: {rms_str}, Peak: {peak_str}")
            if sdr is not None:
                print(f"SDR: {sdr:.2f} dB")
            print(f"Processing time: {processing_time:.2f}s")

        except Exception as e:
            print(f"Error processing Demucs output for {test_file}: {e}")

    # Aggregate summary
    print("\n=== Demucs Evaluation Summary ===")
    print(f"Processed {len(results)} songs")
    summary = {
        'timestamp': timestamp,
        'model_tag': model_tag,
        'dataset_split': 'musdb18/test',
        'target': target,
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

    if log_dir:
        try:
            import json as _json
            with open(json_path, 'w') as f:
                _json.dump({'summary': summary, 'results': results}, f, indent=2)
            print(f"Logs written: {csv_path}")
            print(f"Summary written: {json_path}")
        except Exception as e:
            print(f"Failed to write summary JSON: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Demucs separation on MUSDB18 test set with logging.")
    parser.add_argument('--test-dir', type=str, default='musdb18/test', help='Directory containing MUSDB18 test .stem.mp4 files')
    parser.add_argument('--demucs-out-dir', type=str, default='output_demucs', help='Directory where Demucs writes its outputs')
    parser.add_argument('--evaluation-dir', type=str, default='evaluation_demucs', help='Directory to write normalized outputs for evaluation')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory to write CSV/JSON logs (set empty to disable)')
    parser.add_argument('--model-tag', type=str, default='demucs', help='Identifier for this evaluation run')
    parser.add_argument('--target', type=str, default='vocals', choices=['vocals','bass','drums','other'], help='Which stem to evaluate')
    parser.add_argument('--limit', type=int, default=0, help='Process only first N files (0 = all)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite normalized copies in evaluation dir')
    args = parser.parse_args()

    log_dir = None if not args.log_dir or args.log_dir.strip() == '' else args.log_dir
    limit = None if args.limit is None or args.limit <= 0 else args.limit

    evaluate_demucs(
        test_dir=args.test_dir,
        demucs_out_dir=args.demucs_out_dir,
        evaluation_dir=args.evaluation_dir,
        log_dir=log_dir,
        model_tag=args.model_tag,
        target=args.target,
        limit=limit,
        overwrite=args.overwrite,
    )


if __name__ == '__main__':
    main()
