# Import libraries

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

# Process Config

class Config:
    """Configuration for EEG Processing"""

    input_dir = "/home/rined/Documents/Programming/Python/eeg_analysis/data"
    output_dir = os.path.expanduser("~/eeg_analysis/results")
    cleaned_dir = os.path.expanduser("~/eeg_analysis/cleaned")   #  save cleaned .fif here

    # Filtering

    l_freq = 1.0
    h_freq = 100.0
    notch = 60.0

    # Frequency bands (delta starts at 1 Hz to match high-pass)

    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

# Preprocessing

def detect_bad_channels_simple(raw, z_thresh=3.5, delta_ratio_thresh=5):
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    ch_names = [raw.ch_names[p] for p in picks]
    data = raw.get_data(picks=picks)

    def rz(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return np.zeros_like(x) if mad == 0 else 0.6745 * (x - med) / mad

    log_var = np.log10(np.var(data, axis=1) + 1e-20)
    var_z = rz(log_var)

    psd = raw.compute_psd(method="welch", fmin=0.5, fmax=40, picks=picks, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)

    delta_mask = (freqs >= 0.5) & (freqs <= 4)
    total_mask = (freqs >= 0.5) & (freqs <= 40)

    delta_power = np.trapezoid(psds[:, delta_mask], freqs[delta_mask], axis=1)
    total_power = np.trapezoid(psds[:, total_mask], freqs[total_mask], axis=1)
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

# Main Processing

def process_eeg_data(file_path, config):
    """Processes EEG data - MEMORY SAFE, saves cleaned .fif"""

    try:
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        print(f"    - Duration: {raw.times[-1]:.2f} s")
        print(f"    - Sampling Rate: {raw.info['sfreq']} Hz")
        print(f"    - Channels: {len(raw.ch_names)}")

        raw_clean = raw.copy()
        del raw
        gc.collect()

        raw_clean = normalize_channel_names(raw_clean)

        montage = mne.channels.make_standard_montage("standard_1020")
        raw_clean.set_montage(montage, on_missing="ignore")

        target_sfreq = 500
        if raw_clean.info['sfreq'] != target_sfreq:
            raw_clean.resample(target_sfreq)

        raw_clean.filter(l_freq=config.l_freq, h_freq=config.h_freq, picks="eeg")
        print(f"check Filtered: {config.l_freq}-{config.h_freq} Hz")

        raw_clean.notch_filter(freqs=config.notch)
        print(f"check Notch Filtered: {config.notch} Hz")

        bad_channels = detect_bad_channels_simple(raw_clean)
        raw_clean.info["bads"] = bad_channels
        print(f"check Detected bad channels: {bad_channels if bad_channels else 'None'}")

        raw_clean.set_eeg_reference("average", projection=False)

        ica = ICA(
            n_components=0.95,
            method="infomax",
            fit_params=dict(extended=True),
            random_state=42,
            max_iter="auto"
        )

        picks = mne.pick_types(raw_clean.info, eeg=True, exclude="bads")
        ica.fit(raw_clean, picks=picks)

        eog_indices = []
        try:
            eog_channels = mne.pick_types(raw_clean.info, eog=True)
            if len(eog_channels) > 0:
                eog_indices, _ = ica.find_bads_eog(raw_clean, threshold=3.0)
            else:
                frontal_channels = [ch for ch in ['Fp1', 'Fp2', 'Fz', 'AF3', 'AF4', 'F7', 'F8']
                                    if ch in raw_clean.ch_names]
                if frontal_channels:
                    eog_indices, _ = ica.find_bads_eog(
                        raw_clean, ch_name=frontal_channels, threshold=3.0
                    )
        except Exception as error:
            print("EOG detection failed:", error)

        muscle_indices = []
        try:
            muscle_indices, _ = ica.find_bads_muscle(raw_clean, threshold=0.8)
        except Exception as error:
            print("Muscle detection failed:", error)

        artifact_indices = sorted(set(eog_indices + muscle_indices))
        print("EOG components:", eog_indices)
        print("Muscle components:", muscle_indices)
        print("All artifact components to remove:", artifact_indices)

        ica.exclude = artifact_indices

        if artifact_indices:
            raw_clean = ica.apply(raw_clean.copy())
            print(f"check ICA cleaning complete. Removed {len(artifact_indices)} components")
        else:
            print("check No ICA components to remove")

        if raw_clean.info["bads"]:
            raw_clean.interpolate_bads(reset_bads=True)

        # Save cleaned data so plotter uses real cleaned signal
        cleaned_dir = Path(config.cleaned_dir)
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        cleaned_path = cleaned_dir / (Path(file_path).stem + "_cleaned.fif")
        raw_clean.save(cleaned_path, overwrite=True, verbose=False)
        print(f"    - Saved cleaned data: {cleaned_path.name}")

        # Compute PSD on cleaned data
        psd = raw_clean.compute_psd(
            method='welch',
            fmin=0.5,
            fmax=45,
            n_per_seg=512,
            verbose=False
        )
        print(f"check Computed PSD")

        features = {
            'n_channels': len(raw_clean.ch_names),
            'duration': raw_clean.times[-1],
            'sampling_rate': raw_clean.info['sfreq'],
            'n_bad_channels': len(raw_clean.info["bads"]),
            'bad_channels': raw_clean.info["bads"],
            'n_ica_removed': len(artifact_indices) if artifact_indices else 0
        }

        result = {
            'success': True,
            'features': features,
            'file_name': Path(file_path).name
        }

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

# Main execution

def main():
    config = Config()

    print("\n" + "=" * 60)
    print("EEG Data Processor (Memory Safe)")
    print("=" * 60)

    input_dir = Path(config.input_dir)
    if not input_dir.exists():
        print(f"\nHiccup: Input directory not found. Creating: {input_dir}")
        input_dir.mkdir(parents=True, exist_ok=True)
        print("Please place .set and .fdt files here.")
        return

    set_files = list(input_dir.glob("*.set"))
    if not set_files:
        print(f"\nHiccup: No .set files found in {input_dir}")
        return

    print(f"\ncheck Found {len(set_files)} .set file(s):")
    for f in set_files[:10]:
        print(f"   File: {f.name}")
    if len(set_files) > 10:
        print(f"   ... and {len(set_files) - 10} more")

    all_results = []
    successful = []
    failed = []

    process_files = set_files

    print(f"\nOvO Processing {len(process_files)} file(s)...")

    for idx, set_file in enumerate(process_files):
        print(f"\nOvO Processing {idx+1}/{len(process_files)}")
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

        gc.collect()

    print("\n" + "=" * 60)
    print("OvO PROCESSING SUMMARY")
    print("=" * 60)
    print(f"   Total files: {len(process_files)}")
    print(f"   check Successful: {len(successful)}")
    print(f"   uhoh Failed: {len(failed)}")

    if all_results:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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
        print(f"\nDone! Summary saved to: {csv_path}")

    if all_results:
        print(f"\ncheck Processed {len(all_results)} files successfully!")
    else:
        print("\nuhoh No files processed successfully")

if __name__ == "__main__":
    main()