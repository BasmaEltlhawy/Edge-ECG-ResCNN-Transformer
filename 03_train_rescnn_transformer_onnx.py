import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, LayerNormalization, ReLU, Add, Dense,
    MultiHeadAttention, Softmax, Multiply, GlobalAveragePooling1D, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import tf2onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

# ==========================================
# 1. Configuration & Paths
# ==========================================
DATA_PATH = "./data/mitbih_cleaned_6plus1_balanced.npz"
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Strict Patient-Independent Record Partitioning (Zero Data Leakage)
TRAIN_RECORDS = [
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
    201, 203, 205, 207, 208, 209, 215, 220, 223, 230, 102
]  # 23 Records
VAL_RECORDS = [
    104, 107, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 217, 219
]  # 15 Records
TEST_RECORDS = [
    100, 103, 105, 221, 222, 228, 231, 232, 233
]  # 9 Independent Records (DS2)

# ==========================================
# 2. Data Loading & Patient-Independent Split
# ==========================================
data = np.load(DATA_PATH, allow_pickle=True)
X = np.expand_dims(data['X'], axis=-1)  # Shape: (N, 1250, 1)
y = np.argmax(data['y'], axis=1) if data['y'].ndim == 2 else data['y']

if 'record_ids' in data:
    record_ids = data['record_ids']
    train_mask = np.isin(record_ids, TRAIN_RECORDS)
    val_mask = np.isin(record_ids, VAL_RECORDS)
    test_mask = np.isin(record_ids, TEST_RECORDS)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
else:
    raise KeyError("Dataset must include 'record_ids' to enforce patient-independent splitting.")

# Compute balanced class weights on training set only
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights_array))

print(f"Train samples: {X_train.shape[0]} | Val samples: {X_val.shape[0]} | Test samples: {X_test.shape[0]}")

# ==========================================
# 3. Hybrid ResCNN-Transformer Architecture
# ==========================================
def residual_block(x, filters, kernel_size, stride, reg=l2(1e-4)):
    shortcut = x
    x = Conv1D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=reg)(x)
    x = LayerNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(filters, kernel_size, strides=1, padding='same', kernel_regularizer=reg)(x)
    x = LayerNormalization()(x)
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = Conv1D(filters, kernel_size=1, strides=stride, padding='same', kernel_regularizer=reg)(shortcut)
        shortcut = LayerNormalization()(shortcut)
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x

def transformer_encoder(x, head_size=64, num_heads=4, ff_dim=1024, reg=l2(1e-4)):
    # Pre-LN Multi-Head Self-Attention
    norm1 = LayerNormalization()(x)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(norm1, norm1)
    x = Add()([x, attn])
    
    # Pre-LN Feed-Forward Network
    norm2 = LayerNormalization()(x)
    ff = Dense(ff_dim, activation=tf.keras.activations.gelu, kernel_regularizer=reg)(norm2)
    ff = Dense(x.shape[-1], kernel_regularizer=reg)(ff)
    x = Add()([x, ff])
    return x

def attention_pooling(x):
    scores = Dense(1, activation='tanh')(x)
    weights = Softmax(axis=1)(scores)
    weighted = Multiply()([x, weights])
    return GlobalAveragePooling1D()(weighted)

def build_model(input_shape=(1250, 1), num_classes=7):
    reg = l2(1e-4)
    inputs = Input(shape=input_shape, name="ecg_input")

    # Initial Conv1D Stem (1250 -> 625)
    x = Conv1D(64, 7, strides=2, padding='same', kernel_regularizer=reg)(inputs)
    x = LayerNormalization()(x)
    x = ReLU()(x)

    # Hierarchical Residual Backbone (625x64 -> 313x128 -> 313x256)
    x = residual_block(x, 64, 3, 1, reg)
    x = residual_block(x, 64, 3, 1, reg)
    x = residual_block(x, 128, 3, 2, reg)
    x = residual_block(x, 128, 3, 1, reg)
    x = residual_block(x, 256, 3, 1, reg)

    # Stacked Pre-LN Transformer Encoders
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=1024, reg=reg)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=1024, reg=reg)

    # Trainable Attention Pooling & Classification Head
    x = attention_pooling(x)
    x = LayerNormalization()(x)

    x = Dense(256, activation=tf.keras.activations.gelu, kernel_regularizer=reg)(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation=tf.keras.activations.gelu, kernel_regularizer=reg)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation=tf.keras.activations.gelu, kernel_regularizer=reg)(x)

    outputs = Dense(num_classes, activation='softmax', name="arrhythmia_output")(x)
    return Model(inputs, outputs, name="ResCNN_Transformer_ECG")

# ==========================================
# 4. Model Compilation & Training
# ==========================================
model = build_model(input_shape=(1250, 1), num_classes=7)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

best_model_path = os.path.join(MODEL_DIR, "best_rescnn_transformer.keras")
callbacks = [
    ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    class_weight=class_weights,
    callbacks=callbacks
)

# ==========================================
# 5. Independent Patient-Wise Test Evaluation
# ==========================================
y_test_probs = model.predict(X_test)
y_test_preds = np.argmax(y_test_probs, axis=1)

test_acc = accuracy_score(y_test, y_test_preds)
weighted_f1 = f1_score(y_test, y_test_preds, average='weighted')

print(f"\nIndependent Test Accuracy: {test_acc * 100:.2f}%")
print(f"Independent Weighted F1-Score: {weighted_f1:.4f}")
print("\nDetailed Per-Class Classification Report:\n")
print(classification_report(y_test, y_test_preds, digits=4))

# ==========================================
# 6. ONNX Export & Post-Training INT8 Quantization
# ==========================================
fp32_onnx_path = os.path.join(MODEL_DIR, "ecg_model_fp32.onnx")
int8_onnx_path = os.path.join(MODEL_DIR, "ecg_model_int8.onnx")

spec = (tf.TensorSpec((1, 1250, 1), tf.float32, name="ecg_input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open(fp32_onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

class ECGCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_data):
        self.data = iter([{"ecg_input": sample[np.newaxis, ...].astype(np.float32)} for sample in calibration_data])

    def get_next(self):
        return next(self.data, None)

# Calibrate using 500 representative samples from training set
calibration_reader = ECGCalibrationDataReader(X_train[:500])
quantize_static(
    model_input=fp32_onnx_path,
    model_output=int8_onnx_path,
    calibration_data_reader=calibration_reader,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)
print(f"\nQuantized INT8 ONNX model saved successfully to: {int8_onnx_path}")