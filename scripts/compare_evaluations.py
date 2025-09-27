import os
import re
import glob
import argparse
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None


def find_latest_csv(logs_dir: str, target: str, pattern: str) -> Optional[str]:
    # Files are named like eval_<target>_<model_tag>_<timestamp>.csv
    glob_pat = os.path.join(logs_dir, f"eval_{target}_*_.csv")  # fallback if unusual
    all_csvs = glob.glob(os.path.join(logs_dir, f"eval_{target}_*.csv"))
    # Filter by substring pattern in model_tag portion
    filtered = []
    for p in all_csvs:
        base = os.path.basename(p)
        # eval_target_modeltag_timestamp.csv => split parts
        m = re.match(rf"eval_{re.escape(target)}_(.+)_(\d{{8}}T\d{{6}}Z)\.csv$", base)
        if not m:
            continue
        model_tag = m.group(1)
        ts = m.group(2)
        if pattern.lower() in model_tag.lower():
            filtered.append((p, ts))
    if not filtered:
        return None
    # Pick latest by timestamp string
    filtered.sort(key=lambda x: x[1])
    return filtered[-1][0]


def load_csv(path: str):
    if pd is None:
        raise RuntimeError("pandas is required. Install with: pip install pandas")
    df = pd.read_csv(path)
    # Ensure expected columns exist; fill if missing
    for col in ['sdr_db','processing_time_seconds','file']:
        if col not in df.columns:
            df[col] = np.nan
    return df


def compare_target(logs_dir: str, target: str, unet_pattern: str, demucs_pattern: str, out_dir: str) -> Optional[dict]:
    os.makedirs(out_dir, exist_ok=True)
    unet_csv = find_latest_csv(logs_dir, target, unet_pattern)
    demucs_csv = find_latest_csv(logs_dir, target, demucs_pattern)
    if not unet_csv or not demucs_csv:
        print(f"Missing CSV(s) for target={target}. unet_csv={unet_csv}, demucs_csv={demucs_csv}")
        return None

    d_unet = load_csv(unet_csv)
    d_dem = load_csv(demucs_csv)
    # Align on filename
    key = 'file'
    cols = ['sdr_db','processing_time_seconds','duration_seconds']
    d_unet = d_unet[[key] + [c for c in cols if c in d_unet.columns]].rename(columns={
        'sdr_db': 'sdr_unet', 'processing_time_seconds': 'time_unet', 'duration_seconds': 'dur_unet'
    })
    d_dem = d_dem[[key] + [c for c in cols if c in d_dem.columns]].rename(columns={
        'sdr_db': 'sdr_demucs', 'processing_time_seconds': 'time_demucs', 'duration_seconds': 'dur_demucs'
    })

    df = d_unet.merge(d_dem, on=key, how='inner')
    if df.empty:
        print(f"No overlapping files for target={target}")
        return None

    # Compute differences: positive means Demucs > U-Net
    df['sdr_diff'] = df['sdr_demucs'] - df['sdr_unet']
    df['time_diff'] = df['time_demucs'] - df['time_unet']

    # Summaries
    summary = {
        'target': target,
        'count': int(len(df)),
        'avg_sdr_unet': float(df['sdr_unet'].mean()),
        'avg_sdr_demucs': float(df['sdr_demucs'].mean()),
        'avg_sdr_diff': float(df['sdr_diff'].mean()),
        'avg_time_unet': float(df['time_unet'].mean()),
        'avg_time_demucs': float(df['time_demucs'].mean()),
        'avg_time_diff': float(df['time_diff'].mean()),
        'unet_csv': unet_csv,
        'demucs_csv': demucs_csv,
    }

    # Save merged table
    merged_csv = os.path.join(out_dir, f"compare_{target}.csv")
    df.to_csv(merged_csv, index=False)

    # Plotting
    if plt is None:
        print("matplotlib is required for plotting. Install with: pip install matplotlib")
        return {'summary': summary, 'merged_csv': merged_csv}

    # 1) Average SDR bar chart
    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(['U-Net','Demucs'], [summary['avg_sdr_unet'], summary['avg_sdr_demucs']], color=['#4e79a7','#f28e2b'])
    ax.set_title(f"Average SDR (target={target})")
    ax.set_ylabel("SDR (dB)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"avg_sdr_{target}.png"), dpi=150)
    plt.close(fig)

    # 2) SDR difference histogram (Demucs - U-Net)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(df['sdr_diff'].dropna(), bins=15, color='#59a14f', alpha=0.8)
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set_title(f"SDR Difference (Demucs - U-Net) target={target}")
    ax.set_xlabel("SDR diff (dB)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"sdr_diff_hist_{target}.png"), dpi=150)
    plt.close(fig)

    # 3) Per-file SDR scatter
    fig, ax = plt.subplots(figsize=(8,4))
    idx = np.arange(len(df))
    ax.scatter(idx, df['sdr_unet'], s=18, label='U-Net', color='#4e79a7')
    ax.scatter(idx, df['sdr_demucs'], s=18, label='Demucs', color='#f28e2b')
    ax.set_title(f"Per-file SDR (target={target})")
    ax.set_xlabel("File index (overlap set)")
    ax.set_ylabel("SDR (dB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"sdr_scatter_{target}.png"), dpi=150)
    plt.close(fig)

    # 4) Processing time difference histogram
    if 'time_unet' in df.columns and 'time_demucs' in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(df['time_diff'].dropna(), bins=15, color='#e15759', alpha=0.8)
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        ax.set_title(f"Time Difference (Demucs - U-Net) target={target}")
        ax.set_xlabel("Time diff (s)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"time_diff_hist_{target}.png"), dpi=150)
        plt.close(fig)

    return {'summary': summary, 'merged_csv': merged_csv}


def main():
    parser = argparse.ArgumentParser(description='Compare Demucs and U-Net evaluation logs and plot differences.')
    parser.add_argument('--logs-dir', type=str, default='logs', help='Directory containing evaluation CSV logs')
    parser.add_argument('--output-dir', type=str, default='logs/figures', help='Directory to save merged tables and plots')
    parser.add_argument('--targets', type=str, default='vocals,bass,drums', help='Comma-separated list of targets to compare')
    parser.add_argument('--unet-pattern', type=str, default='unet', help='Substring to select U-Net CSVs (in model_tag)')
    parser.add_argument('--demucs-pattern', type=str, default='demucs', help='Substring to select Demucs CSVs (in model_tag)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    targets = [t.strip() for t in args.targets.split(',') if t.strip()]
    summaries = []
    for t in targets:
        print(f"\n=== Comparing target: {t} ===")
        res = compare_target(args.logs_dir, t, args.unet_pattern, args.demucs_pattern, os.path.join(args.output_dir, t))
        if res and 'summary' in res:
            summaries.append(res['summary'])

    # Print overall recap
    if summaries:
        print("\n=== Overall Summary ===")
        for s in summaries:
            print(
                f"Target={s['target']}: n={s['count']}, "
                f"Avg SDR Unet={s['avg_sdr_unet']:.2f}, Demucs={s['avg_sdr_demucs']:.2f}, Diff(D-U)={s['avg_sdr_diff']:.2f}; "
                f"Avg Time Unet={s['avg_time_unet']:.2f}s, Demucs={s['avg_time_demucs']:.2f}s, Diff(D-U)={s['avg_time_diff']:.2f}s"
            )
    else:
        print("No summaries produced. Ensure logs exist and patterns are correct.")


if __name__ == '__main__':
    main()
