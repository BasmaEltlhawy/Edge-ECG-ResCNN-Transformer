import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# === Load original data ===
INPUT_PATH = r"D:\graduation project\data\data out\mitbih_rpeaks_250hz_5sec_labeled_onehot.npz"
OUTPUT_PATH = r"D:\graduation project\data\data out\mitbih_cleaned_6plus1_balanced.npz"

data = np.load(INPUT_PATH, allow_pickle=True)
X = data['X']
y_original = data['y']

# === Decode raw labels if needed ===
if 'label_names' in data:
    label_names = data['label_names']
else:
    label_names = None

if y_original.ndim == 2:
    y_labels = np.argmax(y_original, axis=1)
    encoder_tmp = LabelEncoder()
    encoder_tmp.fit(label_names)
    y_text = encoder_tmp.inverse_transform(y_labels)
else:
    y_text = y_original.astype(str)

# === Group labels: keep 6 classes, others → "Other"
main_classes = ['Normal', 'PVC', 'LBBB', 'RBBB', 'Paced', 'APB']
y_grouped = []

for label in y_text:
    if label in main_classes:
        y_grouped.append(label)
    else:
        y_grouped.append("Other")

y_grouped = np.array(y_grouped)

# === Undersample "Normal" to 9000
X_new = []
y_new = []

for cls in np.unique(y_grouped):
    cls_indices = np.where(y_grouped == cls)[0]
    if cls == "Normal":
        cls_indices = np.random.choice(cls_indices, 9000, replace=False)
    X_new.append(X[cls_indices])
    y_new.extend([cls] * len(cls_indices))

X_new = np.concatenate(X_new, axis=0)
y_new = np.array(y_new)

# === Shuffle data
X_new, y_new = shuffle(X_new, y_new, random_state=42)

# === Encode and one-hot
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_new)
y_onehot = to_categorical(y_encoded)

# === Show class distribution
print("\n📊 Final Class Distribution (after undersampling):")
class_counts = Counter(y_new)
for label, count in class_counts.items():
    print(f"{label}: {count}")

plt.figure(figsize=(10, 5))
plt.bar(class_counts.keys(), class_counts.values())
plt.title("Balanced Class Distribution (Normal = 9000)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Save final file
np.savez_compressed(OUTPUT_PATH, X=X_new, y=y_onehot, label_names=encoder.classes_)
print(f"\n✅ Final balanced dataset saved to: {OUTPUT_PATH}")
print(f"🧾 Classes: {list(encoder.classes_)}")


