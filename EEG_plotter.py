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

class Config:
    """Configuration for EEG Processing"""

    input_dir = "/home/rined/Documents/Programming/Python/eeg_analysis/data"  # path
    output_dir = os.path.expanduser("~/eeg_analysis/results")
    plots_dir = os.path.expanduser("~/eeg_analysis/plots")

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

    input_dir = os.path.expanduser("~/eeg_analysis/data")
    output_dir = os.path.expanduser("~/eeg_analysis/results")
    plots_dir = os.path.expanduser("~/eeg_analysis/plots")



# Normalize channel names (from main script)

def normalize_channel_names(raw):
    """Normalize common EEG channel names to match MNE standard montage naming."""

    rename_dict = {}

    for ch in raw.ch_names:
        ch_clean = ch.strip()
        ch_upper = ch_clean.upper()

        # Fp channels: FP1 -> Fp1, FP2 -> Fp2, FPZ -> Fpz
        if ch_upper.startswith("FP"):
            rename_dict[ch] = "Fp" + ch_upper[2:].lower()

        # Midline z channels: FZ -> Fz, CZ -> Cz, CPZ -> CPz, POZ -> POz
        elif ch_upper.endswith("Z"):
            rename_dict[ch] = ch_upper[:-1] + "z"

        # Other standard channels: AF3, F7, C3, P4, O1, etc.
        else:
            rename_dict[ch] = ch_upper

    raw.rename_channels(rename_dict)
    return raw



# ============================================================
# CUSTOM PLOTTING FUNCTIONS
# ============================================================

def plot_raw_vs_cleaned(raw, raw_clean, file_name, start=0, duration=10, n_channels=8):
    """Plot raw vs cleaned data with proper units and labels"""
    
    print(f"   Plotting raw vs cleaned for {file_name}...")
    
    # Get data
    sfreq = raw.info['sfreq']
    start_idx = int(start * sfreq)
    end_idx = int((start + duration) * sfreq)
    times = np.arange(start_idx, end_idx) / sfreq
    
    # Select EEG channels (first n_channels)
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    ch_indices = eeg_picks[:n_channels]
    ch_names = [raw.ch_names[i] for i in ch_indices]
    
    # Get data
    raw_data, _ = raw[:, start_idx:end_idx]
    clean_data, _ = raw_clean[:, start_idx:end_idx]
    
    # Create figure with two columns
    fig, axes = plt.subplots(n_channels, 2, figsize=(14, 2 * n_channels))
    
    # Handle single channel case
    if n_channels == 1:
        axes = axes.reshape(1, -1)
    
    # Find global y-limits for consistent scaling
    all_data = np.concatenate([raw_data[ch_indices].flatten(), 
                               clean_data[ch_indices].flatten()])
    y_min, y_max = np.percentile(all_data, [1, 99])
    y_pad = (y_max - y_min) * 0.1
    
    for i, (ch_idx, ch_name) in enumerate(zip(ch_indices, ch_names)):
        # Raw data
        axes[i, 0].plot(times, raw_data[ch_idx], color='#4477AA', linewidth=0.8)
        axes[i, 0].set_ylabel(f'{ch_name} (µV)', fontsize=9)
        axes[i, 0].set_ylim(y_min - y_pad, y_max + y_pad)
        axes[i, 0].grid(True, alpha=0.2)
        axes[i, 0].tick_params(labelsize=8)
        
        # Cleaned data
        axes[i, 1].plot(times, clean_data[ch_idx], color='#228833', linewidth=0.8)
        axes[i, 1].set_ylabel(f'{ch_name} (µV)', fontsize=9)
        axes[i, 1].set_ylim(y_min - y_pad, y_max + y_pad)
        axes[i, 1].grid(True, alpha=0.2)
        axes[i, 1].tick_params(labelsize=8)
        
        # Add vertical line at start if start > 0
        if start > 0:
            axes[i, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            axes[i, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Set labels for bottom subplots only
    axes[-1, 0].set_xlabel('Time (s)', fontsize=10)
    axes[-1, 1].set_xlabel('Time (s)', fontsize=10)
    
    # Add column titles with units
    axes[0, 0].set_title(f'{file_name} - Raw EEG (µV)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title(f'{file_name} - Cleaned (1-45 Hz, µV)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{file_name}_raw_cleaned.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"   check Plots saved for {file_name}")



def plot_artifact_check(raw, raw_clean, file_name, start=0, duration=10):
    """Check for artifacts and jumps in the data, not all that necessary but worth looking at"""
    
    print(f"   Checking for artifacts in {file_name}...")
    
    # Get data
    sfreq = raw.info['sfreq']
    start_idx = int(start * sfreq)
    end_idx = int((start + duration) * sfreq)
    times = np.arange(start_idx, end_idx) / sfreq
    
    # Select a few channels to check
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    ch_indices = eeg_picks[:6]
    ch_names = [raw.ch_names[i] for i in ch_indices]
    
    raw_data, _ = raw[:, start_idx:end_idx]
    clean_data, _ = raw_clean[:, start_idx:end_idx]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    for i, (ch_idx, ch_name) in enumerate(zip(ch_indices, ch_names)):
        row = i // 2
        col = i % 2
        
        # Plot raw and cleaned overlaid for comparison
        axes[row, col].plot(times, raw_data[ch_idx], color='#4477AA', alpha=0.6, 
                           linewidth=0.8, label='Raw')
        axes[row, col].plot(times, clean_data[ch_idx], color='#228833', alpha=0.8,
                           linewidth=0.8, label='Cleaned')
        axes[row, col].set_title(f'{ch_name}', fontsize=10)
        axes[row, col].grid(True, alpha=0.2)
        axes[row, col].legend(fontsize=8)
        
        # Mark any large jumps (> 3 standard deviations)
        diff = np.diff(clean_data[ch_idx])
        threshold = 3 * np.std(diff)
        jump_indices = np.where(np.abs(diff) > threshold)[0]
        if len(jump_indices) > 0:
            for j in jump_indices[:5]:  # Mark first 5 jumps
                axes[row, col].axvline(x=times[j], color='red', 
                                      linestyle='--', alpha=0.5, linewidth=1)
    
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 1].set_xlabel('Time (s)')
    plt.suptitle(f'{file_name} - Artifact Detection (red lines = large jumps)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{file_name}_artifact_check.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   check Artifact check saved for {file_name}")



def plot_psd_with_bands(raw_clean, file_name, bands, band_colors):
    """Plot PSD with frequency bands highlighted"""
    
    print(f"   Plotting PSD for {file_name}...")
    
    # Compute PSD
    raw_eeg = raw_clean.copy().pick("eeg")
    psd = raw_eeg.compute_psd(method="welch", fmin=1, fmax=45, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Background frequency bands
    for band, (fmin, fmax) in bands.items():
        plt.axvspan(fmin, fmax, color=band_colors[band], alpha=0.35)
    
    # Plot each channel
    for ch_idx, ch_name in enumerate(raw_eeg.ch_names):
        plt.plot(freqs, np.log10(psds[ch_idx]), alpha=0.25, linewidth=1)
    
    # Plot mean across channels
    mean_psd = psds.mean(axis=0)
    mean_line, = plt.plot(freqs, np.log10(mean_psd), color="black", linewidth=3, label="Mean")
    
    # Create legend handles for bands
    band_handles = [
        Patch(facecolor=band_colors[band], alpha=0.35, label=band.capitalize())
        for band in bands.keys()
    ]
    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10(Power)")
    plt.title(f"{file_name} - Power Spectral Density Across Channels")
    plt.xlim(1, 45)
    plt.grid(alpha=0.3)
    plt.legend(handles=band_handles + [mean_line], loc="upper right")
    
    plt.savefig(f"{plots_dir}/{file_name}_psd.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   check PSD plot saved for {file_name}")



def plot_topomaps(bandpower_df, raw_info, file_name, bands):
    """Plot topomaps for absolute and relative power with proper labels"""
    
    print(f"   Plotting topomaps for {file_name}...")
    
    band_order = list(bands.keys())
    metrics = [
        ("absolute_power", "Absolute Power (µV²)"),
        ("relative_power", "Relative Power (%)"),
    ]
    
    fig = plt.figure(figsize=(5 * len(band_order), 10), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=4,
        ncols=len(band_order),
        height_ratios=[10, 1.5, 10, 1.5],
        hspace=0.3,
        wspace=0.4,
    )
    
    for row_idx, (metric, row_label) in enumerate(metrics):
        topo_row = row_idx * 2
        cbar_row = topo_row + 1
        
        for col_idx, band in enumerate(band_order):
            ax = fig.add_subplot(gs[topo_row, col_idx])
            cax = fig.add_subplot(gs[cbar_row, col_idx])
            
            band_df = bandpower_df[bandpower_df["band"] == band]
            
            # Get values for each channel
            values = []
            for ch in raw_info.ch_names:
                if ch in band_df["channel"].values:
                    val = band_df.loc[band_df["channel"] == ch, metric].values[0]
                    values.append(val)
            
            values = np.array(values)
            
            # removed vmin and vmax, letting MNE handle scaling
            im, _ = mne.viz.plot_topomap(
                values,
                raw_info,
                axes=ax,
                show=False,
                contours=0,
                cmap="viridis",
            )
            
            if row_idx == 0:
                ax.set_title(f"{band.capitalize()}\n{bands[band][0]}-{bands[band][1]} Hz", 
                           fontsize=12, fontweight='bold')
            
            if col_idx == 0:
                ax.text(
                    -0.25,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                    fontweight='bold',
                )
            
            # Colorbar with units
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            if metric == "absolute_power":
                cbar.set_label("Power (µV²)", fontsize=9)
            else:
                cbar.set_label("Relative Power (%)", fontsize=9)
            cax.tick_params(labelsize=8)
    
    plt.suptitle(f"{file_name} - EEG Band Power Topomaps", fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(f"{plots_dir}/{file_name}_topomaps.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   check Topomaps saved for {file_name}")



def extract_band_powers(raw_clean, bands):
    """Extract absolute and relative band powers"""
    
    raw_eeg = raw_clean.copy().pick("eeg")
    psd = raw_eeg.compute_psd(method="welch", fmin=1, fmax=45, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)
    
    # Total power (1-45 Hz)
    total_mask = (freqs >= 1) & (freqs <= 45)
    total_power = np.trapz(psds[:, total_mask], freqs[total_mask], axis=1)
    
    bandpower_features = []
    
    for band, (fmin, fmax) in bands.items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        
        absolute_power = np.trapz(
            psds[:, band_mask],
            freqs[band_mask],
            axis=1,
        )
        
        relative_power = np.divide(
            absolute_power,
            total_power,
            out=np.zeros_like(absolute_power),
            where=total_power != 0,
        )
        
        for ch_name, abs_val, rel_val in zip(raw_eeg.ch_names, absolute_power, relative_power):
            bandpower_features.append({
                "channel": ch_name,
                "band": band,
                "absolute_power": abs_val,
                "relative_power": rel_val,
            })
    
    return pd.DataFrame(bandpower_features)



def save_bandpower_summary(bandpower_df, file_name, output_dir):
    """Save bandpower summary as a clean CSV file"""
    
    print(f"   Saving bandpower summary for {file_name}...")
    
    # Pivot the data
    abs_pivot = bandpower_df.pivot(index='channel', columns='band', values='absolute_power')
    rel_pivot = bandpower_df.pivot(index='channel', columns='band', values='relative_power')
    
    # Combine into one DataFrame with clear column names
    summary = pd.DataFrame()
    summary['Channel'] = abs_pivot.index
    
    for band in abs_pivot.columns:
        summary[f'{band}_abs_µV²'] = abs_pivot[band].values
        summary[f'{band}_rel_%'] = rel_pivot[band].values
    
    # Add total power
    summary['Total_Power_µV²'] = summary[[f'{band}_abs_µV²' for band in abs_pivot.columns]].sum(axis=1)
    
    # Save to CSV
    csv_path = Path(output_dir) / f'{file_name}_bandpowers.csv'
    summary.to_csv(csv_path, index=False)
    
    # Also append to master summary
    master_path = Path(output_dir).parent / 'results' / 'all_bandpowers_master.csv'
    if master_path.exists():
        existing = pd.read_csv(master_path)
        all_data = pd.concat([existing, summary], ignore_index=True)
        all_data.to_csv(master_path, index=False)
    else:
        summary.to_csv(master_path, index=False)
    
    # Print a clean summary to console
    print(f"\n    Band Power Summary for {file_name}:")
    print(f"   {'='*50}")
    print(f"   Total Channels: {len(summary)}")
    print(f"   Frequency Bands: {', '.join(abs_pivot.columns)}")
    print(f"   Max Alpha Power: {summary['alpha_abs_µV²'].max():.4f} µV²")
    print(f"   Max Beta Power:  {summary['beta_abs_µV²'].max():.4f} µV²")
    print(f"    Saved to: {csv_path}")
    
    return summary



# ============================================================
# MAIN EXECUTION - PROCESS ALL FILES
# ============================================================

def main():
    config = Config()
    
    global plots_dir
    plots_dir = config.plots_dir
    
    # Create plots directory
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("EEG Time Series and Plotting Pipeline")
    print("=" * 60)
    
    # Load the summary CSV
    results_file = Path(config.output_dir) / 'processing_summary.csv'
    if not results_file.exists():
        print(f"\nHiccup: No results found at {results_file}")
        print("Please run EEG_set_analyzer.py first.")
        return
    
    df = pd.read_csv(results_file)
    print(f"\ncheck Found {len(df)} processed files")
    
    # Frequency bands and colors
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
    
    # Process each file
    all_bandpower_dfs = []
    
    for idx, row in df.iterrows():
        file_name = row['File']
        file_path = Path(config.input_dir) / file_name
        
        print(f"\nwrrrrrr Processing {idx+1}/{len(df)}: {file_name}")
        
        try:
            # Load data
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
            
            # Re-apply basic preprocessing to get raw_clean
            raw_clean = raw.copy()
            
            # Normalize channel names
            raw_clean = normalize_channel_names(raw_clean)
            
            montage = mne.channels.make_standard_montage("standard_1020")
            raw_clean.set_montage(montage, on_missing="ignore")
            
            # Downsample if needed
            target_sfreq = 500
            if raw_clean.info['sfreq'] != target_sfreq:
                raw_clean.resample(target_sfreq)
            
            # Filter for visualization (1-45 Hz)
            raw_clean.filter(l_freq=1, h_freq=45, picks="eeg", verbose=False)
            
            # 1. Plot raw vs cleaned
            plot_raw_vs_cleaned(raw, raw_clean, file_name.replace('.set', ''), start=0, duration=10, n_channels=8)
            
            # 2. Plot artifact check
            plot_artifact_check(raw, raw_clean, file_name.replace('.set', ''), start=0, duration=10)
            
            # 3. Plot PSD with bands
            plot_psd_with_bands(raw_clean, file_name.replace('.set', ''), bands, band_colors)
            
            # 4. Extract band powers
            bandpower_df = extract_band_powers(raw_clean, bands)
            all_bandpower_dfs.append(bandpower_df)
            
            # 5. Save bandpower summary
            save_bandpower_summary(bandpower_df, file_name.replace('.set', ''), config.plots_dir)
            
            # 6. Plot topomaps
            plot_topomaps(bandpower_df, raw_clean.info, file_name.replace('.set', ''), bands)
            
            print(f"   check All plots complete for {file_name}")
            
            # Clean up
            del raw, raw_clean
            gc.collect()
            
        except Exception as e:
            print(f"   uhoh Error with {file_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Combine all bandpower data
    if all_bandpower_dfs:
        combined_bandpower = pd.concat(all_bandpower_dfs, ignore_index=True)
        combined_bandpower.to_csv(f"{plots_dir}/all_bandpowers.csv", index=False)
        print(f"\n Bandpower data saved to: {plots_dir}/all_bandpowers.csv")
    
    print("\n" + "=" * 60)
    print("OvO PLOTTING COMPLETE")
    print("=" * 60)
    print(f"check Plots saved to: {plots_dir}")
    print(f"check Bandpower data saved to: {plots_dir}/all_bandpowers.csv")



if __name__ == "__main__":
    main()