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



# Verify installations

print(f"check MNE version: {mne.__version__}")
print(f"check numpy version: {np.__version__}")
print(f"check pandas version: {pd.__version__}")



# Preprocessing, taken from Google Colab

def detect_bad_channels_simple(raw, z_thresh=3.5, delta_ratio_thresh=5):        # Find any noisemakers
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    ch_names = [raw.ch_names[p] for p in picks]
    data = raw.get_data(picks=picks)

    def rz(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return np.zeros_like(x) if mad == 0 else 0.6745 * (x - med) / mad

    # Variance outliers
    log_var = np.log10(np.var(data, axis=1) + 1e-20)
    var_z = rz(log_var)

    # Delta power outliers
    psd = raw.compute_psd(method="welch", fmin=0.5, fmax=40, picks=picks, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)

    delta_mask = (freqs >= 0.5) & (freqs <= 4)
    total_mask = (freqs >= 0.5) & (freqs <= 40)

    # FIXED: np.trapezoid changed to np.trapz in newer NumPy versions
    delta_power = np.trapz(psds[:, delta_mask], freqs[delta_mask], axis=1)
    total_power = np.trapz(psds[:, total_mask], freqs[total_mask], axis=1)
    delta_ratio = delta_power / (total_power + 1e-20)

    delta_z = rz(np.log10(delta_power + 1e-20))

    bad_channels = [
        ch_names[i]
        for i in range(len(ch_names))
        if abs(var_z[i]) > z_thresh
        or delta_z[i] > z_thresh
        or delta_ratio[i] > delta_ratio_thresh * np.median(delta_ratio)
    ]

    return bad_channels

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



# Process Config

class Config:
    """Configuration for EEG Processing"""

    input_dir = os.path.expanduser("~/eeg_analysis/data")          # Input directory for raw EEG data
    output_dir = os.path.expanduser("~/eeg_analysis/results")      # Output directory for processed results, the useful kind

    
    
    # Filtering

    l_freq = 1.0      # Low frequency for bandpass filter
    h_freq = 100.0    # High frequency for bandpass filter
    notch = 60.0      # This is a detail I would have missed without AI insight. The frequency of power line noise may mess with the signal an EEG picks up. So this will also be filtered out if appearing.


    # Frequency bands

    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }




# Main Processing - MEMORY SAFE VERSION

def process_eeg_data(file_path, config):
    """Processes EEG data from a given .set/.fdt file - MEMORY SAFE"""
    
    try:
        raw = mne.io.read_raw_eeglab(file_path, preload=True)  # Load raw EEG data
        print(f"    - Duration: {raw.times[-1]:.2f} s")
        print(f"    - Sampling Rate: {raw.info['sfreq']} Hz")
        print(f"    - Channels: {len(raw.ch_names)}")

        # Make a copy of the raw data so the original recording stays unchanged
        raw_clean = raw.copy()
        # Free the original raw to save memory
        del raw
        gc.collect()

        # Normalize channel names
        raw_clean = normalize_channel_names(raw_clean)

        # Set electrode locations
        # The montage tells MNE where the EEG electrodes are located on the scalp.
        # This is needed for topographic plots and bad-channel interpolation.
        montage = mne.channels.make_standard_montage("standard_1020")
        raw_clean.set_montage(montage, on_missing="ignore")

        # Optional: downsample the data to reduce file size and speed up processing.
        # This is often unnecessary if the original sampling rate is already appropriate.
        target_sfreq = 500
        if raw_clean.info['sfreq'] != target_sfreq:
            raw_clean.resample(target_sfreq)

        # Band-pass filter the data.
        # Keeps frequencies of interest while reducing slow drifts and high-frequency noise.
        raw_clean.filter(l_freq=config.l_freq, h_freq=config.h_freq, picks="eeg")
        print(f"check Filtered: {config.l_freq}-{config.h_freq} Hz")

        # Apply notch filter to reduce line noise, 60 Hz for American standard.
        raw_clean.notch_filter(freqs=config.notch)
        print(f"check Notch Filtered: {config.notch} Hz")

        # Detect bad channels.
        # Bad channels may be noisy, flat, or unstable because of poor contact,
        # movement, or electrode issues. Visual inspection is still important,
        # but simple automated methods can provide a useful first pass.
        bad_channels = detect_bad_channels_simple(raw_clean)
        raw_clean.info["bads"] = bad_channels
        print(f"check Detected bad channels: {bad_channels if bad_channels else 'None'}")

        # Re-reference the EEG to the average of all EEG channels.
        # EEG is always measured relative to a reference, so this step affects the signal values.
        raw_clean.set_eeg_reference("average", projection=False)

        # Artifact removal with ICA
        # EEG often contains non-neural activity such as eye or muscle artifacts.
        # Here we use ICA to identify and remove some of these components.

        # Create the ICA object - using slightly fewer components for memory
        ica = ICA(
            n_components=0.95,    # Reduced slightly from 0.99 for memory
            method="infomax",
            fit_params=dict(extended=True),
            random_state=42,
            max_iter="auto"
        )

        # Fit ICA using only good EEG channels
        picks = mne.pick_types(
            raw_clean.info,
            eeg=True,
            exclude="bads"
        )

        ica.fit(raw_clean, picks=picks)

        # Try to identify components related to eye activity
        eog_indices = []
        try:
            eog_indices, eog_scores = ica.find_bads_eog(
                raw_clean,
                threshold=3.0
            )
        except Exception as error:
            print("EOG detection failed:", error)

        # Try to identify components related to muscle activity
        muscle_indices = []
        try:
            muscle_indices, muscle_scores = ica.find_bads_muscle(
                raw_clean,
                threshold=0.8
            )
        except Exception as error:
            print("Muscle detection failed:", error)

        # Combine all detected artifact components
        artifact_indices = sorted(set(eog_indices + muscle_indices))

        print("EOG components:", eog_indices)
        print("Muscle components:", muscle_indices)
        print("All artifact components to remove:", artifact_indices)

        # Mark these components for removal
        ica.exclude = artifact_indices

        # Apply ICA cleaning to the data
        if artifact_indices:
            raw_clean = ica.apply(raw_clean.copy())
            print(f"check ICA cleaning complete. Removed {len(artifact_indices)} components")
        else:
            print("check No ICA components to remove")

        # Interpolate channels that were marked as bad.
        # This restores the full channel layout, which is useful for scalp maps and later analyses.
        if raw_clean.info["bads"]:
            raw_clean.interpolate_bads(reset_bads=True)

        # Compute Power Spectral Density (PSD) for each band for mapping and further analysis
        # Using smaller segment size for memory efficiency
        psd = raw_clean.compute_psd(
            method='welch',
            fmin=0.5,
            fmax=45,
            n_per_seg=512,        # Reduced from 1024 for memory
            verbose=False
        )
        print(f"check Computed PSD")

        # Extract frequency bands
        band_data = {}
        for band, (low, high) in config.bands.items():
            band_raw = raw_clean.copy()
            band_raw.filter(
                l_freq=low, 
                h_freq=high,
                method='fir',
                fir_design='firwin',
                skip_by_annotation='edge',
                verbose=False
            )
            band_data[band] = band_raw
            print(f"    - Processed {band} band: {low}-{high} Hz")

        # Extract basic features as listed
        features = {
            'n_channels': len(raw_clean.ch_names),
            'duration': raw_clean.times[-1],
            'sampling_rate': raw_clean.info['sfreq'],
            'n_bad_channels': len(raw_clean.info["bads"]),
            'bad_channels': raw_clean.info["bads"],
            'n_ica_removed': len(artifact_indices) if artifact_indices else 0
        }
        
        # Store results but DON'T keep the raw data to save memory
        result = {
            'success': True,
            'features': features,
            'file_name': Path(file_path).name
        }
        
        # Clean up large objects before returning
        del raw_clean
        del ica
        gc.collect()
        
        return result
        
    except FileNotFoundError:
        print(f"uhoh File not found: {file_path}")
        return {'success': False, 'error': 'File not found'}
        
    except Exception as e:
        print(f"uhoh Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}



# Main execution, find the files and work magic

def main():
    config = Config()
    
    print("\n" + "=" * 60)
    print("EEG Data Processor (Memory Safe)")
    print("=" * 60)
    
    # Check input directory
    input_dir = Path(config.input_dir)
    if not input_dir.exists():
        print(f"\nHiccup: Input directory not found. Creating: {input_dir}")
        input_dir.mkdir(parents=True, exist_ok=True)
        print("📂 Please place your .set and .fdt files here.")
        return
    
    # Find .set files
    set_files = list(input_dir.glob("*.set"))
    if not set_files:
        print(f"\nHiccup: No .set files found in {input_dir}")
        print("📂 Please place your .set and .fdt files in this folder.")
        return
    
    print(f"\ncheck Found {len(set_files)} .set file(s):")
    for f in set_files[:10]:  # Only show first 10 to keep output clean
        print(f"   📄 {f.name}")
    if len(set_files) > 10:
        print(f"   ... and {len(set_files) - 10} more")
    
    # Process all files
    all_results = []
    successful = []
    failed = []
    
    # Process all files (no testing limit)
    process_files = set_files    # Process all files
    
    print(f"\nOvO Processing {len(process_files)} file(s)...")
    
    for idx, set_file in enumerate(process_files):
        print(f"\n📊 Processing {idx+1}/{len(process_files)}")
        print(f"{'='*60}")
        print(f"OvO Processing: {set_file.name}")
        print(f"{'='*60}")
        
        result = process_eeg_data(set_file, config)
        
        if result and result.get('success'):
            all_results.append(result)
            successful.append(result['file_name'])
            print(f"   check Success!")
        else:
            failed.append(result.get('file_name', str(set_file)))
            print(f"   uhoh Failed: {result.get('error', 'Unknown error') if result else 'No result'}")
        
        # Force garbage collection after each file to free memory
        gc.collect()
        # ^ Perhaps the most important line. Without this change, my hardware was overloaded, and vs code crashed 
    
    # Summary
    print("\n" + "=" * 60)
    print("OvO PROCESSING SUMMARY")
    print("=" * 60)
    print(f"   Total files: {len(process_files)}")
    print(f"   check Successful: {len(successful)}")
    print(f"   uhoh Failed: {len(failed)}")
    
    if successful:
        print(f"\n   check Successfully processed:")
        for f in successful:
            print(f"      - {f}")
    
    if failed:
        print(f"\n   uhoh Failed:")
        for f in failed:
            print(f"      - {f}")
    
    # Save results summary
    if all_results:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary DataFrame
        summary_data = []
        for r in all_results:
            summary_data.append({
                'File': r['file_name'],
                'Channels': r['features']['n_channels'],
                'Duration': r['features']['duration'],
                'Sampling_Rate': r['features']['sampling_rate'],
                'Bad_Channels': r['features']['n_bad_channels'],
                'Bad_Channel_Names': str(r['features']['bad_channels']),
                'ICA_Removed': r['features']['n_ica_removed']
            })
        
        df = pd.DataFrame(summary_data)
        csv_path = output_dir / 'processing_summary.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n📁 Summary saved to: {csv_path}")
    
    if all_results:
        print(f"\ncheck Processed {len(all_results)} files successfully!")
    else:
        print("\nuhoh No files processed successfully")



if __name__ == "__main__":
    main()