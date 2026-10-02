# Import necessary libraries

import os
import sys
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Most important: MNE

import mne
from mne.preprocessing import ICA
from matplotlib.patches import Patch

# Verify installations

print(f"check MNE version: {mne.__version__}")
print(f"check numpy version: {np.__version__}")
print(f"check pandas version: {pd.__version__}")

# Process Config

class Config:
    """Configuration for EEG Processing"""

    input_dir = "/home/rined/Documents/Programming/Python/eeg_analysis/data"
    output_dir = os.path.expanduser("~/eeg_analysis/results")
    plots_dir = os.path.expanduser("~/eeg_analysis/plots")
    cleaned_dir = os.path.expanduser("~/eeg_analysis/cleaned")

# Normalize channel names (from main script)

def normalize_channel_names(raw):
    rename_dict = {}
    for ch in raw.ch_names:
        ch_clean = ch.strip()
        ch_upper = ch_clean.upper()
        if ch_upper.startswith("FP"):
            rename_dict[ch] = "Fp" + ch_upper[2:].lower()
        elif ch_upper.endswith("Z"):
            rename_dict[ch] = ch_upper[:-1] + "z"
        else:
            rename_dict[ch] = ch_upper
    raw.rename_channels(rename_dict)
    return raw

# ============================================================
# CUSTOM PLOTTING FUNCTIONS
# ============================================================

def plot_raw_vs_cleaned(raw, raw_clean, file_name, start=0, duration=10, n_channels=8):
    """Plot raw vs cleaned with proper time alignment and units"""

    print(f"   Plotting raw vs cleaned for {file_name}...")

    # Resample raw to match cleaned
    if raw.info['sfreq'] != raw_clean.info['sfreq']:
        raw = raw.copy().resample(raw_clean.info['sfreq'])

    sfreq = raw.info['sfreq']
    start_idx = int(start * sfreq)
    end_idx = int((start + duration) * sfreq)
    times = np.arange(start_idx, end_idx) / sfreq

    eeg_picks = mne.pick_types(raw.info, eeg=True)
    ch_indices = eeg_picks[:n_channels]
    ch_names = [raw.ch_names[i] for i in ch_indices]

    raw_data, _ = raw[:, start_idx:end_idx]
    clean_data, _ = raw_clean[:, start_idx:end_idx]

    # Convert volts to microvolts
    raw_data = raw_data * 1e6
    clean_data = clean_data * 1e6

    fig, axes = plt.subplots(n_channels, 2, figsize=(14, 2 * n_channels))
    if n_channels == 1:
        axes = axes.reshape(1, -1)

    all_data = np.concatenate([raw_data[ch_indices].flatten(),
                               clean_data[ch_indices].flatten()])
    y_min, y_max = np.percentile(all_data, [1, 99])
    y_pad = (y_max - y_min) * 0.1

    for i, (ch_idx, ch_name) in enumerate(zip(ch_indices, ch_names)):
        axes[i, 0].plot(times, raw_data[ch_idx], color='#4477AA', linewidth=0.8)
        axes[i, 0].set_ylabel(f'{ch_name} (µV)', fontsize=9)
        axes[i, 0].set_ylim(y_min - y_pad, y_max + y_pad)
        axes[i, 0].grid(True, alpha=0.2)
        axes[i, 0].tick_params(labelsize=8)

        axes[i, 1].plot(times, clean_data[ch_idx], color='#228833', linewidth=0.8)
        axes[i, 1].set_ylabel(f'{ch_name} (µV)', fontsize=9)
        axes[i, 1].set_ylim(y_min - y_pad, y_max + y_pad)
        axes[i, 1].grid(True, alpha=0.2)
        axes[i, 1].tick_params(labelsize=8)

    axes[-1, 0].set_xlabel('Time (s)', fontsize=10)
    axes[-1, 1].set_xlabel('Time (s)', fontsize=10)

    axes[0, 0].set_title(f'{file_name} - Raw EEG (µV)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title(f'{file_name} - Cleaned (ICA applied, µV)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{file_name}_raw_cleaned.png", dpi=200, bbox_inches='tight')
    plt.close()

    print(f"   check Plots saved for {file_name}")

def plot_psd_with_bands(raw_clean, file_name, bands, band_colors):
    """Plot PSD in µV² with band shading"""

    print(f"   Plotting PSD for {file_name}...")

    raw_eeg = raw_clean.copy().pick("eeg")
    psd = raw_eeg.compute_psd(method="welch", fmin=1, fmax=45, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)

    # Convert V² to µV²
    psds = psds * 1e12

    plt.figure(figsize=(10, 6))
    for band, (fmin, fmax) in bands.items():
        plt.axvspan(fmin, fmax, color=band_colors[band], alpha=0.35)

    for ch_idx in range(psds.shape[0]):
        plt.plot(freqs, np.log10(psds[ch_idx] + 1e-20), alpha=0.25, linewidth=1)

    mean_psd = psds.mean(axis=0)
    mean_line, = plt.plot(freqs, np.log10(mean_psd + 1e-20), color="black", linewidth=3, label="Mean")

    band_handles = [
        Patch(facecolor=band_colors[band], alpha=0.35, label=band.capitalize())
        for band in bands.keys()
    ]

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10(Power) (µV²)")
    plt.title(f"{file_name} - Power Spectral Density (cleaned)")
    plt.xlim(1, 45)
    plt.grid(alpha=0.3)
    plt.legend(handles=band_handles + [mean_line], loc="upper right")

    plt.savefig(f"{plots_dir}/{file_name}_psd.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   check PSD plot saved for {file_name}")

def plot_topomaps(bandpower_df, raw_info, file_name, bands):
    """Topomaps in µV² and % with shared scale per band"""

    print(f"   Plotting topomaps for {file_name}...")

    band_order = list(bands.keys())
    metrics = [
        ("absolute_power", "Absolute Power (µV²)"),
        ("relative_power", "Relative Power (%)"),
    ]

    # Compute shared vmin/vmax per metric
    vlims = {}
    for metric, _ in metrics:
        vals = bandpower_df[metric].values
        vlims[metric] = (np.percentile(vals, 2), np.percentile(vals, 98))

    fig = plt.figure(figsize=(5 * len(band_order), 10), constrained_layout=True)
    gs = fig.add_gridspec(nrows=4, ncols=len(band_order),
                          height_ratios=[10, 1.5, 10, 1.5],
                          hspace=0.3, wspace=0.4)

    for row_idx, (metric, row_label) in enumerate(metrics):
        topo_row = row_idx * 2
        cbar_row = topo_row + 1
        vmin, vmax = vlims[metric]

        for col_idx, band in enumerate(band_order):
            ax = fig.add_subplot(gs[topo_row, col_idx])
            cax = fig.add_subplot(gs[cbar_row, col_idx])

            band_df = bandpower_df[bandpower_df["band"] == band]
            values = []
            for ch in raw_info.ch_names:
                if ch in band_df["channel"].values:
                    values.append(band_df.loc[band_df["channel"] == ch, metric].values[0])
            values = np.array(values)

            im, _ = mne.viz.plot_topomap(
                values, raw_info, axes=ax, show=False, contours=0,
                cmap="viridis", vlim=(vmin, vmax)
            )

            if row_idx == 0:
                ax.set_title(f"{band.capitalize()}\n{bands[band][0]}-{bands[band][1]} Hz",
                             fontsize=12, fontweight='bold')

            if col_idx == 0:
                ax.text(-0.25, 0.5, row_label, transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=11, fontweight='bold')

            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            cbar.set_label(row_label, fontsize=9)
            cax.tick_params(labelsize=8)

    plt.suptitle(f"{file_name} - EEG Band Power Topomaps (cleaned)",
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(f"{plots_dir}/{file_name}_topomaps.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   check Topomaps saved for {file_name}")

def extract_band_powers(raw_clean, bands):
    """Extract band powers in µV² and %"""

    raw_eeg = raw_clean.copy().pick("eeg")
    psd = raw_eeg.compute_psd(method="welch", fmin=1, fmax=45, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)
    psds = psds * 1e12

    total_mask = (freqs >= 1) & (freqs <= 45)
    total_power = np.trapezoid(psds[:, total_mask], freqs[total_mask], axis=1)

    bandpower_features = []
    for band, (fmin, fmax) in bands.items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        absolute_power = np.trapezoid(psds[:, band_mask], freqs[band_mask], axis=1)
        relative_power = np.divide(
            absolute_power, total_power,
            out=np.zeros_like(absolute_power),
            where=total_power != 0
        ) * 100   # fraction to percentage

        for ch_name, abs_val, rel_val in zip(raw_eeg.ch_names, absolute_power, relative_power):
            bandpower_features.append({
                "channel": ch_name,
                "band": band,
                "absolute_power": abs_val,
                "relative_power": rel_val,
            })

    return pd.DataFrame(bandpower_features)

def save_bandpower_summary(bandpower_df, file_name, output_dir):
    """Save per-file bandpower CSV. Master file overwrites."""

    print(f"   Saving bandpower summary for {file_name}...")

    abs_pivot = bandpower_df.pivot(index='channel', columns='band', values='absolute_power')
    rel_pivot = bandpower_df.pivot(index='channel', columns='band', values='relative_power')

    summary = pd.DataFrame()
    summary['Channel'] = abs_pivot.index
    for band in abs_pivot.columns:
        summary[f'{band}_abs_µV²'] = abs_pivot[band].values
        summary[f'{band}_rel_%'] = rel_pivot[band].values

    summary['Total_Power_µV²'] = summary[[f'{band}_abs_µV²' for band in abs_pivot.columns]].sum(axis=1)

    csv_path = Path(output_dir) / f'{file_name}_bandpowers.csv'
    summary.to_csv(csv_path, index=False)

    return summary

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    config = Config()

    global plots_dir
    plots_dir = config.plots_dir
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("EEG Time Series and Plotting Pipeline (Cleaned Data)")
    print("=" * 60)

    results_file = Path(config.output_dir) / 'processing_summary.csv'
    if not results_file.exists():
        print(f"\nHiccup: No results found at {results_file}")
        print("Please run EEG_set_analyzer.py first.")
        return

    df = pd.read_csv(results_file)
    print(f"\ncheck Found {len(df)} processed files")

    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    band_colors = {
        "delta": "#4477AA",
        "theta": "#66CCEE",
        "alpha": "#228833",
        "beta": "#CCBB44",
        "gamma": "#EE6677",
    }

    all_bandpower_dfs = []

    for idx, row in df.iterrows():
        file_name = row['File']
        stem = file_name.replace('.set', '')
        cleaned_path = Path(config.cleaned_dir) / f"{stem}_cleaned.fif"

        print(f"\nwrrrrrr Processing {idx+1}/{len(df)}: {file_name}")

        if not cleaned_path.exists():
            print(f"   uhoh Cleaned file not found: {cleaned_path}")
            print(f"   Skipping — rerun EEG_set_analyzer.py to generate cleaned .fif files")
            continue

        try:
            raw_clean = mne.io.read_raw_fif(cleaned_path, preload=True, verbose=False)

            # Reconstruct "raw" for comparison: reload raw .set without cleaning
            raw_path = Path(config.input_dir) / file_name
            raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose=False)
            raw = normalize_channel_names(raw)
            raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")

            # 1. Plot raw vs cleaned
            plot_raw_vs_cleaned(raw, raw_clean, stem, start=0, duration=10, n_channels=8)

            # 2. PSD
            plot_psd_with_bands(raw_clean, stem, bands, band_colors)

            # 3. Band powers
            bandpower_df = extract_band_powers(raw_clean, bands)
            all_bandpower_dfs.append(bandpower_df)

            # 4. Save summary
            save_bandpower_summary(bandpower_df, stem, config.plots_dir)

            # 5. Topomaps
            plot_topomaps(bandpower_df, raw_clean.info, stem, bands)

            print(f"   check All plots complete for {file_name}")

            del raw, raw_clean
            gc.collect()

        except Exception as e:
            print(f"   uhoh Error with {file_name}: {e}")
            import traceback
            traceback.print_exc()

    if all_bandpower_dfs:
        combined = pd.concat(all_bandpower_dfs, ignore_index=True)
        combined.to_csv(f"{plots_dir}/all_bandpowers.csv", index=False)
        print(f"\n Bandpower data saved to: {plots_dir}/all_bandpowers.csv")

    print("\n" + "=" * 60)
    print("OvO PLOTTING COMPLETE")
    print("=" * 60)
    print(f"check Plots saved to: {plots_dir}")

if __name__ == "__main__":
    main()