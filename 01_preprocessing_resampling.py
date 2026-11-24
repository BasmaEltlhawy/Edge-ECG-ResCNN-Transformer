import wfdb
import numpy as np
import os
from scipy.signal import resample, butter, filtfilt, iirnotch
from tqdm import tqdm
import neurokit2 as nk
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# === CONFIG ===
MITDB_PATH = r"D:\graduation project\data\raw data\mit-bih-arrhythmia-database-1.0.0 (1)\mit-bih-arrhythmia-database-1.0.0"
OUTPUT_PATH = r"D:\graduation project\data\data out\mitbih_rpeaks_250hz_5sec_labeled_onehot.npz"
WINDOW_SIZE = 1250  # 5 seconds @ 250Hz
TARGET_FS = 250

# === FILTERING FUNCTIONS ===
def bandpass_filter(signal, lowcut=0.5, highcut=45.0, fs=250, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, freq=50.0, fs=250, quality=30.0):
    b, a = iirnotch(freq / (fs / 2), quality)
    return filtfilt(b, a, signal)

# === BEAT TYPE MAPPING ===
BEAT_MAP = {
    'N': 'Normal',
    'L': 'LBBB',
    'R': 'RBBB',
    'V': 'PVC',
    'A': 'APB',
    '/': 'Paced',
    'F': 'Fusion',
    'E': 'Escape',
    'j': 'Nodal Escape',
    'a': 'Atrial Premature',
    'J': 'Junctional Premature'
}

# === STEP 1: PROCESS RECORDS ===
segments = []
labels = []

# 🔁 Loop through .dat files only
records = sorted(list(set(f.split('.')[0] for f in os.listdir(MITDB_PATH) if f.endswith('.dat'))))

for record in tqdm(records, desc="Processing MIT-BIH"):
    try:
        # Load signal and annotation
        signal, fields = wfdb.rdsamp(os.path.join(MITDB_PATH, record), channels=[0])
        annotation = wfdb.rdann(os.path.join(MITDB_PATH, record), 'atr')

        signal = signal.flatten()
        original_fs = fields['fs']

        # Resample to 250Hz
        if original_fs != TARGET_FS:
            signal = resample(signal, int(len(signal) * TARGET_FS / original_fs))
            annotation.sample = [int(s * TARGET_FS / original_fs) for s in annotation.sample]

        # Filtering
        signal = bandpass_filter(signal, fs=TARGET_FS)
        signal = notch_filter(signal, fs=TARGET_FS)

        # R-peak detection
        rpeaks = nk.ecg_peaks(signal, sampling_rate=TARGET_FS)[1]['ECG_R_Peaks']

        # Segment around R-peaks
        half_window = WINDOW_SIZE // 2
        for r in rpeaks:
            start = r - half_window
            end = r + half_window
            if start >= 0 and end <= len(signal):
                # Find nearest annotation
                beat_idx = np.argmin(np.abs(np.array(annotation.sample) - r))
                beat_symbol = annotation.symbol[beat_idx]
                if beat_symbol in BEAT_MAP:
                    label = BEAT_MAP[beat_symbol]
                    segments.append(signal[start:end])
                    labels.append(label)

    except Exception as e:
        print(f"❌ Error in record {record}: {e}")

# Convert to numpy arrays
X = np.array(segments)
y = np.array(labels)

# === STEP 2: Class distribution
print("\n📊 Class distribution:")
class_counts = Counter(y)
for label, count in class_counts.items():
    print(f"{label}: {count}")

# Plot
plt.figure(figsize=(10, 5))
plt.bar(class_counts.keys(), class_counts.values())
plt.title("Class Distribution (250Hz - 5s Segments)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# === STEP 3: One-hot encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Save compressed
np.savez_compressed(OUTPUT_PATH, X=X, y=y_onehot, label_names=encoder.classes_)

print(f"\n✅ Saved {len(X)} labeled segments to: {OUTPUT_PATH}")
print(f"🔤 Classes: {list(encoder.classes_)}")


