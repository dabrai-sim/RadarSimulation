# Hidden Metal Object Detection using mmWave Radar & Deep Learning

## Project Overview

This project implements an end-to-end AI-powered radar system for detecting hidden metal objects in cluttered environments. Using millimeter-wave (mmWave) FMCW radar simulation and Convolutional Neural Networks (CNNs), the system can classify objects as metal or non-metal from Range-Doppler maps.

## Objectives

1. Simulate realistic 77 GHz FMCW radar signals
2. Generate Range-Doppler maps for various scenarios
3. Train a CNN classifier to distinguish metal from non-metal objects
4. Deploy detection pipeline for cluttered scenes with occlusion
5. Evaluate performance under different noise and clutter conditions

## System Architecture
<img width="461" height="1832" alt="image" src="https://github.com/user-attachments/assets/98e2c1c4-e675-42f9-9ea9-52b92a3d2bad" />

## 📁 Project Structure
```
├── radar_simulation.pdf          # Part 1: Radar signal generation & processing
├── classification_model.pdf      # Part 2: CNN training on clean data
├── hidden_object_detection.pdf   # Part 3: Deployment on cluttered scenes
├── classification_model_data/    # Generated datasets
├── classification_model_outputs/ # Trained models & metrics
└── outputs_hidden/               # Detection results & visualizations
```
## 🔬 Technical Implementation

### Part 1: Radar Simulation (`radar_simulation.pdf`)

**Key Components:**
- **RadarSignalGenerator**: FMCW signal synthesis
  - Carrier frequency: 77 GHz
  - Bandwidth: 4 GHz
  - Sweep time: 40 μs
  - Range resolution: 0.037 m
  - Velocity resolution: 0.380 m/s

- **RadarProcessor**: Signal processing pipeline
  - Range FFT with Hann/Hamming windowing
  - Doppler FFT with zero-Doppler suppression
  - Magnitude computation (dB scale)

- **ScenarioGenerator**: Diverse test scenarios
  - Empty room (noise baseline)
  - Single metal object
  - Multiple objects (metal + non-metal)
  - Cluttered scenes with occlusion

### Part 2: Classification Model (`classification_model.pdf`)

**Dataset Generation:**
- 1,500 synthetic samples (750 metal, 750 non-metal)
- Metal RCS: 18-30 (strong reflections)
- Non-metal RCS: 0.1-1.5 (weak reflections)
- Clear class separation ensured

**CNN Architecture:**
```
Input (128×256×1) → Conv2D(16) → BN → MaxPool → Dropout(0.2)
                  → Conv2D(32) → BN → MaxPool → Dropout(0.4)
                  → Conv2D(64) → BN → MaxPool → Dropout(0.5)
                  → GlobalAvgPool → Dense(64) → Dropout(0.6)
                  → Dense(2, softmax)
```

**Training Strategy:**
- Data augmentation: noise injection, Doppler/range shifts, amplitude scaling
- Callbacks: EarlyStopping (patience=20), ReduceLROnPlateau
- Optimizer: Adam (lr=0.0003)
- Total parameters: 28,290

**Results:**
- Test Accuracy: **100%**
- Precision: **100%**
- Recall: **100%**
- F1-Score: **100%**
- ROC AUC: **1.0**

### Part 3: Hidden Object Detection (`hidden_object_detection.pdf`)

**Advanced Processing:**
- **CFAR Detector**: Adaptive thresholding based on local noise
- **Background Subtraction**: Percentile filtering to remove static clutter
- **Morphological Filtering**: Binary opening/closing for noise cleanup
- **Adaptive Noise Filter**: Wiener-like edge-preserving filter

**Detection Pipeline:**
1. Preprocess scene (background subtraction + noise filtering)
2. Apply CFAR to identify peaks
3. Segment connected components
4. Classify each region with CNN
5. Aggregate detections

**Deployment Results:**
- Mean Detection Rate: **16.1%** 
- Mean Precision: **23.4%**
- Mean Recall: **16.1%**
- Mean F1-Score: **19.1%**

**Performance by Occlusion:**
- Low: 26.1% detection rate
- Medium: 13.3%
- High: 8.2%

## Key Results

| Metric | Training (Clean) | Deployment (Cluttered) |
|--------|------------------|------------------------|
| Accuracy | 100% | N/A |
| Precision | 100% | 23.4% |
| Recall | 100% | 16.1% |
| F1-Score | 100% | 19.1% |

**Critical Finding:** The model exhibits severe performance degradation in real-world scenarios despite perfect training accuracy.

## Usage

### 1. Generate Radar Data
```python
from radar_simulation import RadarSignalGenerator, RadarProcessor

radar_gen = RadarSignalGenerator()
signal = radar_gen.generate_2d_signal(targets=[...])
```

### 2. Train Classification Model
```python
from classification_model import RadarCNN, train_model

model, history = train_model(X_train, y_train, X_val, y_val, input_shape)
```

### 3. Run Hidden Object Detection
```python
from hidden_object_detection import HiddenObjectDetectionPipeline

pipeline = HiddenObjectDetectionPipeline(model_path, processor)
detections = pipeline.segment_and_classify(rd_map, detection_map)
```

## Limitations

1. **Domain Gap**: Model trained on clean isolated objects fails on cluttered scenes
2. **Low Recall**: Misses 84% of metal objects in occluded environments
3. **CFAR Sensitivity**: Detection threshold may be too conservative
4. **No Temporal Processing**: Single-frame analysis without tracking
5. **Synthetic Data Only**: Not validated on real radar hardware

## Proposed Improvements

1. **Domain Adaptation**:
   - Train on cluttered scenes from the start
   - Use mixed datasets (clean + occluded)
   - Apply stronger augmentation (realistic clutter injection)

2. **Advanced Detection**:
   - Tune CFAR parameters (reduce PFA threshold)
   - Implement multi-scale processing
   - Add temporal tracking (Kalman filtering)

3. **Model Architecture**:
   - Use attention mechanisms to focus on metal signatures
   - Implement object detection (YOLO/Faster R-CNN) instead of patch classification
   - Add multi-task learning (detection + classification simultaneously)

4. **Real-World Validation**:
   - Collect data from actual mmWave radar (TI AWR1843, IWR6843)
   - Fine-tune on real measurements
   - Deploy on embedded hardware (Jetson Nano, Raspberry Pi)

## 📚 Dependencies
```
numpy
matplotlib
seaborn
scipy
tensorflow
keras
scikit-learn
```

## 🙏 Acknowledgments

- Radar simulation based on FMCW principles from TI mmWave SDK
- CNN architecture inspired by ResNet and VGG
