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
from matplotlib.patches import Patch

# Global plot directory

hf_plots_dir = None

# Verify installations

print(f"check MNE version: {mne.__version__}")
print(f"check numpy version: {np.__version__}")
print(f"check pandas version: {pd.__version__}")

# Process Config

class Config:
    """Configuration for High-Frequency EEG Analysis (45-150 Hz)"""

    input_dir = "/home/rined/Documents/Programming/Python/eeg_analysis/data"
    output_dir = os.path.expanduser("~/eeg_analysis/results")
    cleaned_dir = os.path.expanduser("~/eeg_analysis/cleaned")
    hf_plots_dir = os.path.expanduser("~/eeg_analysis/plots_highfreq")

    # High-frequency bands of interest

    hf_bands = {
        "low_gamma":  (30, 45),
        "high_gamma": (45, 80),
        "very_high":  (80, 150),
    }

    # Narrow bands around specific frequencies of interest

    narrow_bands = {
        "around_40":  (38, 42),
        "around_60":  (58, 62),    # mains
        "around_80":  (78, 82),    # 2nd harmonic of 40
        "around_120": (118, 122),  # 2nd harmonic of 60
    }

# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def compute_high_freq_psd(raw_clean, fmax=150):
    """Compute PSD from 30-150 Hz in µV²"""

    raw_eeg = raw_clean.copy().pick("eeg")
    psd = raw_eeg.compute_psd(
        method="welch",
        fmin=30,
        fmax=fmax,
        n_per_seg=512,
        verbose=False
    )
    psds, freqs = psd.get_data(return_freqs=True)
    psds = psds * 1e12   # V² → µV²

    return raw_eeg, psds, freqs

def extract_hf_band_powers(psds, freqs, hf_bands):
    """Extract absolute and relative power for high-frequency bands"""

    total_mask = (freqs >= 30) & (freqs <= 150)
    total_power = np.trapz(psds[:, total_mask], freqs[total_mask], axis=1)

    features = []
    for band, (fmin, fmax) in hf_bands.items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        abs_power = np.trapz(psds[:, band_mask], freqs[band_mask], axis=1)
        rel_power = np.divide(
            abs_power, total_power,
            out=np.zeros_like(abs_power),
            where=total_power != 0
        ) * 100
        features.append({
            "band": band,
            "fmin": fmin,
            "fmax": fmax,
            "abs_power": abs_power,
            "rel_power": rel_power,
        })
    return features, total_power

def extract_narrow_peaks(psds, freqs, narrow_bands):
    """Extract peak power in narrow windows around key frequencies"""

    peaks = {}
    for name, (fmin, fmax) in narrow_bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if mask.sum() > 0:
            peak_power = np.max(psds[:, mask], axis=1)
            peak_freq = freqs[mask][np.argmax(psds[:, mask], axis=1)]
            peaks[name] = {
                "peak_power": peak_power,
                "peak_freq": peak_freq,
            }
    return peaks

def check_harmonic_ratio(peaks, ch_names):
    """Check for 80 Hz harmonic of 40 Hz, and 120 Hz harmonic of 60 Hz"""

    print("\n    Harmonic check:")
    if "around_40" in peaks and "around_80" in peaks:
        ratio_80_40 = peaks["around_80"]["peak_power"] / (peaks["around_40"]["peak_power"] + 1e-20)
        print(f"    80/40 ratio (mean): {np.mean(ratio_80_40):.3f}")
        print(f"    80/40 ratio (max):  {np.max(ratio_80_40):.3f}")
        print(f"    → If ~1.0, likely device harmonic")

    if "around_60" in peaks and "around_120" in peaks:
        ratio_120_60 = peaks["around_120"]["peak_power"] / (peaks["around_60"]["peak_power"] + 1e-20)
        print(f"    120/60 ratio (mean): {np.mean(ratio_120_60):.3f}")
        print(f"    120/60 ratio (max):  {np.max(ratio_120_60):.3f}")
        print(f"    → If ~1.0, likely mains harmonic")

# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_hf_psd(psds, freqs, file_name, hf_bands, narrow_bands, band_colors):
    """Plot PSD 30-150 Hz with high-frequency bands shaded"""

    print(f"   Plotting high-frequency PSD for {file_name}...")

    plt.figure(figsize=(12, 6))

    for band, (fmin, fmax) in hf_bands.items():
        plt.axvspan(fmin, fmax, color=band_colors.get(band, "#999999"), alpha=0.25)

    for name, (fmin, fmax) in narrow_bands.items():
        plt.axvspan(fmin, fmax, color="red", alpha=0.10)

    for ch_idx in range(psds.shape[0]):
        plt.plot(freqs, np.log10(psds[ch_idx] + 1e-20), alpha=0.15, linewidth=0.8)

    mean_psd = psds.mean(axis=0)
    mean_line, = plt.plot(freqs, np.log10(mean_psd + 1e-20),
                          color="black", linewidth=2.5, label="Mean")

    band_handles = [
        Patch(facecolor=band_colors.get(b, "#999999"), alpha=0.25,
              label=b.replace("_", " ").title())
        for b in hf_bands.keys()
    ]

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10(Power) (µV²)")
    plt.title(f"{file_name} - High-Frequency PSD (30-150 Hz, cleaned)")
    plt.xlim(30, 150)
    plt.grid(alpha=0.3)
    plt.legend(handles=band_handles + [mean_line], loc="upper right", fontsize=9)

    plt.savefig(f"{hf_plots_dir}/{file_name}_hf_psd.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   check HF PSD saved for {file_name}")

def plot_120hz_zoom(psds, freqs, file_name):
    """Zoom in on the 120 Hz region specifically"""

    print(f"   Plotting 120 Hz zoom for {file_name}...")

    mask = (freqs >= 100) & (freqs <= 140)
    freqs_z = freqs[mask]
    psds_z = psds[:, mask]

    plt.figure(figsize=(10, 5))

    for ch_idx in range(psds_z.shape[0]):
        plt.plot(freqs_z, np.log10(psds_z[ch_idx] + 1e-20), alpha=0.20, linewidth=0.8)

    mean_psd = psds_z.mean(axis=0)
    plt.plot(freqs_z, np.log10(mean_psd + 1e-20), color="black", linewidth=2.5, label="Mean")

    plt.axvline(x=120, color="red", linestyle="--", alpha=0.7, label="120 Hz")
    plt.axvline(x=60, color="orange", linestyle=":", alpha=0.5, label="60 Hz")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10(Power) (µV²)")
    plt.title(f"{file_name} - 120 Hz Region Zoom (100-140 Hz, cleaned)")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)

    plt.savefig(f"{hf_plots_dir}/{file_name}_120hz_zoom.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   check 120 Hz zoom saved for {file_name}")

def plot_hf_topomaps(hf_features, ch_names, raw_info, file_name, hf_bands):
    """Topomaps for high-frequency bands"""

    print(f"   Plotting high-frequency topomaps for {file_name}...")

    band_order = list(hf_bands.keys())
    n_bands = len(band_order)

    fig = plt.figure(figsize=(5 * n_bands, 6), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=n_bands,
                          height_ratios=[10, 1.5], hspace=0.3, wspace=0.4)

    # Collect all abs values for shared scale
    all_abs = np.concatenate([f["abs_power"] for f in hf_features])
    vmin, vmax = np.percentile(all_abs, [2, 98])

    for col_idx, (band, feature) in enumerate(zip(band_order, hf_features)):
        ax = fig.add_subplot(gs[0, col_idx])
        cax = fig.add_subplot(gs[1, col_idx])

        values = feature["abs_power"]

        im, _ = mne.viz.plot_topomap(
            values, raw_info, axes=ax, show=False, contours=0,
            cmap="viridis", vlim=(vmin, vmax)
        )

        ax.set_title(f"{band.replace('_', ' ').title()}\n{hf_bands[band][0]}-{hf_bands[band][1]} Hz",
                     fontsize=11, fontweight='bold')

        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("Power (µV²)", fontsize=9)
        cax.tick_params(labelsize=8)

    plt.suptitle(f"{file_name} - High-Frequency Topomaps (cleaned)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(f"{hf_plots_dir}/{file_name}_hf_topomaps.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   check HF topomaps saved for {file_name}")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    config = Config()

    global hf_plots_dir
    hf_plots_dir = config.hf_plots_dir

    Path(hf_plots_dir).mkdir(parents=True, exist_ok=True)
    ...

    print("\n" + "=" * 60)
    print("EEG High-Frequency Analysis (45-150 Hz)")
    print("=" * 60)

    results_file = Path(config.output_dir) / 'processing_summary.csv'
    if not results_file.exists():
        print(f"\nHiccup: No results found at {results_file}")
        print("Please run EEG_set_analyzer.py first.")
        return

    df = pd.read_csv(results_file)
    print(f"\ncheck Found {len(df)} processed files")

    band_colors = {
        "low_gamma":  "#EE6677",
        "high_gamma": "#AA3377",
        "very_high":  "#663399",
    }

    all_hf_rows = []

    for idx, row in df.iterrows():
        file_name = row['File']
        stem = file_name.replace('.set', '')
        cleaned_path = Path(config.cleaned_dir) / f"{stem}_cleaned.fif"

        print(f"\nwrrrrrr HF Processing {idx+1}/{len(df)}: {file_name}")

        if not cleaned_path.exists():
            print(f"   uhoh Cleaned file not found: {cleaned_path}")
            continue

        try:
            raw_clean = mne.io.read_raw_fif(cleaned_path, preload=True, verbose=False)

            # 1. Compute high-frequency PSD
            raw_eeg, psds, freqs = compute_high_freq_psd(raw_clean, fmax=150)

            # 2. Extract HF band powers
            hf_features, total_hf_power = extract_hf_band_powers(psds, freqs, config.hf_bands)

            # 3. Extract narrow peaks
            peaks = extract_narrow_peaks(psds, freqs, config.narrow_bands)

            # 4. Harmonic checks
            check_harmonic_ratio(peaks, raw_eeg.ch_names)

            # 5. Save rows for master CSV
            for feature in hf_features:
                for ch_name, abs_val, rel_val in zip(
                    raw_eeg.ch_names,
                    feature["abs_power"],
                    feature["rel_power"]
                ):
                    row_data = {
                        "File": file_name,
                        "Channel": ch_name,
                        "Band": feature["band"],
                        "Abs_Power_µV²": abs_val,
                        "Rel_Power_%": rel_val,
                    }
                    if "around_120" in peaks:
                        row_data["Peak_120Hz_µV²"] = peaks["around_120"]["peak_power"][raw_eeg.ch_names.index(ch_name)]
                    if "around_40" in peaks:
                        row_data["Peak_40Hz_µV²"] = peaks["around_40"]["peak_power"][raw_eeg.ch_names.index(ch_name)]
                    if "around_80" in peaks:
                        row_data["Peak_80Hz_µV²"] = peaks["around_80"]["peak_power"][raw_eeg.ch_names.index(ch_name)]
                    all_hf_rows.append(row_data)

            # 6. Plots
            plot_hf_psd(psds, freqs, stem, config.hf_bands, config.narrow_bands, band_colors)
            plot_120hz_zoom(psds, freqs, stem)
            plot_hf_topomaps(hf_features, raw_eeg.ch_names, raw_eeg.info, stem, config.hf_bands)

            print(f"   check HF analysis complete for {file_name}")

            del raw_clean, raw_eeg
            gc.collect()

        except Exception as e:
            print(f"   uhoh Error with {file_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save master HF summary
    if all_hf_rows:
        hf_df = pd.DataFrame(all_hf_rows)
        hf_csv = Path(config.hf_plots_dir) / "high_frequency_summary.csv"
        hf_df.to_csv(hf_csv, index=False)
        print(f"\n HF summary saved to: {hf_csv}")

    print("\n" + "=" * 60)
    print("OvO HIGH-FREQUENCY ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"check Plots saved to: {config.hf_plots_dir}")

if __name__ == "__main__":
    main()