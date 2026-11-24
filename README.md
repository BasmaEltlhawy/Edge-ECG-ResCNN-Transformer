# Edge-Computing Framework for ECG Arrhythmia Classification
Official implementation of the quantized hybrid ResCNN-Transformer framework for real-time ECG arrhythmia classification on Raspberry Pi 4B.

## Contents
- Signal resampling (360 Hz to 250 Hz) and causal Butterworth filtering scripts
- Hybrid ResCNN-Transformer model architecture and training pipeline
- Post-training static INT8 quantization via ONNX Runtime
- Real-time PyQt5 edge deployment interface
