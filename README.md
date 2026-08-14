# RF-DETR Weapon Detection

**Real-time multi-class weapon detection for CCTV and video surveillance using RF-DETR, PyTorch, OpenCV, and Supervision.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![RF-DETR](https://img.shields.io/badge/Model-RF--DETR%20Medium-green)
![Classes](https://img.shields.io/badge/Classes-5-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

This project implements a **five-class weapon detection system based on RF-DETR Medium**, designed primarily for detecting weapons and weapon-related objects in CCTV footage, recorded videos, and webcam streams.

The system was developed with surveillance environments in mind, where detection performance can be affected by:

- Variable illumination
- Low-light and infrared CCTV footage
- Motion blur
- Lens blur
- Video compression artifacts
- Perspective distortion
- Fisheye and wide-angle cameras
- Small or distant weapons
- Partial weapon occlusion
- Shadows
- Rain, fog, and environmental degradation
- Different camera and recording conditions

The training pipeline therefore uses a CCTV-oriented augmentation strategy together with RF-DETR's end-to-end object detection architecture.

---

## Detection Classes

The model detects the following five classes:

| ID | Class |
|---:|---|
| 0 | Handgun |
| 1 | Heavy Weapon |
| 2 | Knife |
| 3 | Rifle |
| 4 | Rocket_Grenade Launcher |

The class definitions are shared between the training configuration and inference pipeline.

---

## Model

### RF-DETR Medium

The project uses **RF-DETR Medium** as the trained detection architecture.

The configured model uses:

- **Architecture:** RF-DETR Medium
- **Backbone:** DINOv2 Windowed Medium
- **Number of classes:** 5
- **Input resolution:** 672 × 672
- **Number of queries:** 300
- **Number of selected detections:** 300
- **Mixed precision:** enabled during training
- **EMA:** enabled
- **Device:** CUDA when available

The training configuration records the model as `RFDETRMedium` with a DINOv2 Windowed Medium encoder and five detection classes.

---

## Performance

The best recorded validation performance was obtained at **epoch 28**.

| Metric | Score |
|---|---:|
| Precision | **95.72%** |
| Recall | **93.55%** |
| F1 Score | **94.62%** |
| mAP@0.5 | **96.61%** |
| mAP@0.5:0.95 | **84.82%** |
| Best Epoch | **28** |

The best epoch is selected according to the highest recorded **mAP@0.5:0.95**.

> **Important:** These metrics represent the recorded training/validation results available in this repository's training artifacts. They should not be interpreted as a guarantee of performance on unseen CCTV environments or operational surveillance deployments.

---

## Training Configuration

The training pipeline is configurable through `config.py`.

### Core Training Parameters

| Parameter | Value |
|---|---:|
| Epochs | 100 |
| Batch size | 8 |
| Effective batch size | 16 |
| Gradient accumulation | 2 |
| Input resolution | 672 × 672 |
| Initial learning rate | 1e-4 |
| Encoder learning rate | 1.5e-4 |
| Weight decay | 1e-4 |
| Early stopping patience | 10 epochs |
| Early stopping minimum delta | 0.005 |
| Random seed | 42 |
| Precision | 16-bit mixed precision |
| EMA | Enabled |
| Checkpoint interval | Every 5 epochs |

Training uses CUDA automatically when a compatible NVIDIA GPU is available.

---

## CCTV-Oriented Data Augmentation

A major component of this project is the use of domain-specific augmentation designed around real-world CCTV conditions.

The augmentation pipeline includes:

### Photometric Augmentation

- Random brightness and contrast
- Hue, saturation, and value shifts
- RGB color shifts
- CLAHE
- Sharpening
- Grayscale/infrared simulation

### Blur, Noise, and Compression

- Gaussian blur
- Motion blur
- Sensor noise
- Image compression
- Downscaling

### Geometric Augmentation

- Horizontal flipping
- Affine transformations
- Perspective transformations
- Random scaling
- Optical/fisheye distortion

### Environmental Augmentation

- Rain
- Fog
- Shadows

### Occlusion Augmentation

- Coarse dropout

The augmentation strategy was specifically configured to expose the detector to challenging surveillance conditions, including low illumination, compression artifacts, small/distant weapons, camera distortion, environmental degradation, and partial occlusion.

---

## Why CCTV-Specific Augmentation?

Weapons in surveillance footage often occupy only a small portion of an image and may be partially hidden by people, clothing, objects, shadows, or camera artifacts.

For example:

- A handgun may only expose its grip.
- A knife may be partially hidden by clothing or a person's hand.
- A rifle may be partially occluded by the carrier's body.
- Distant weapons may contain very few usable pixels.
- Aggressive CCTV compression can destroy thin structures such as knife blades.
- Motion blur can significantly reduce weapon detail.
- Fisheye and wide-angle cameras can distort object geometry.

The augmentation configuration therefore attempts to reproduce these conditions during training rather than relying exclusively on clean images.

---

## Project Structure

A recommended repository structure is:

```text
RF-DETR-Weapon-Detection/
│
├── config.py
├── train.py
├── test.py
│
├── dataset/
│   └── ...
│
├── runs/
│   ├── metrics.csv
│   ├── train_metadata.json
│   └── training_config.json
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

### Main Files

| File | Purpose |
|---|---|
| `config.py` | Central configuration for dataset paths, classes, model variant, training parameters, and augmentations |
| `train.py` | RF-DETR training and checkpoint/metrics management |
| `test.py` | Video/webcam inference and visualization |
| `metrics.csv` | Per-epoch training/validation metrics generated by the training framework |
| `train_metadata.json` | Best-epoch summary containing precision, recall, F1, mAP metrics, and weight paths |
| `training_config.json` | Recorded training and model configuration |
| `LICENSE` | MIT License |
| `README.md` | Project documentation |

---

## Dataset

The training pipeline expects the dataset in **YOLO format** and is configured to work with a Roboflow-exported dataset structure.

Expected structure:

```text
dataset/
└── Weapons_Detection_New_v2_version_2_yolo26/
    ├── train/
    │   ├── images/
    │   └── labels/
    │
    ├── valid/
    │   ├── images/
    │   └── labels/
    │
    ├── test/
    │   ├── images/
    │   └── labels/
    │
    └── data.yaml
```

The training configuration uses the Roboflow dataset loader and YOLO-format bounding-box annotations.

> **Dataset availability:** The dataset itself is not necessarily included in this repository. If the dataset is subject to separate licensing, ownership, privacy, or distribution restrictions, obtain it from its authorized source rather than assuming it is included with this project.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU training/inference, install a CUDA-compatible PyTorch build appropriate for your NVIDIA driver and CUDA environment.

---

## Requirements

The project relies on the following major components:

- Python
- PyTorch
- RF-DETR
- OpenCV
- NumPy
- Supervision
- Pillow
- Albumentations

A complete dependency list should be maintained in `requirements.txt`.

---

## Training

After preparing the dataset and configuring the paths in `config.py`, start training with:

```bash
python train.py
```

The training script:

1. Loads the configured RF-DETR model.
2. Detects whether CUDA is available.
3. Sets reproducibility seeds.
4. Loads or automatically discovers a valid checkpoint when available.
5. Trains using the configured augmentation pipeline.
6. Uses mixed precision when training on CUDA.
7. Maintains exponential moving average weights.
8. Performs validation during training.
9. Applies early stopping.
10. Saves periodic and best-performing checkpoints.
11. Extracts the best validation metrics.
12. Writes the best-model metadata to `train_metadata.json`.

---

## Resuming Training

The training pipeline supports both explicit and automatic checkpoint recovery.

To explicitly resume from a checkpoint:

```bash
python train.py --resume path/to/checkpoint.pth
```

When no explicit checkpoint is supplied, the training pipeline attempts to locate an existing checkpoint in the configured output directory.

This allows interrupted training sessions to be continued without manually reconstructing the training state.

---

## Checkpoints and Model Weights

### Why are the weights not included?

The trained RF-DETR checkpoints and model weights are **intentionally excluded from this GitHub repository**.

The generated checkpoint files are large — the repository contains checkpoints exceeding GitHub's standard **100 MiB per-file limit**. Examples include:

```text
checkpoint_4.ckpt
checkpoint_9.ckpt
checkpoint_14.ckpt
checkpoint_19.ckpt
checkpoint_24.ckpt
checkpoint_29.ckpt
checkpoint_best_ema.pth
checkpoint_best_regular.pth
checkpoint_best_total.pth
last.ckpt
```

Including these files directly in a normal Git repository would therefore cause GitHub to reject the files.

The repository consequently contains the **source code, configuration, training metadata, and documentation**, while the large binary model artifacts remain outside the GitHub repository.

### Important

The following files should **not** be committed to the repository:

```text
*.pth
*.ckpt
*.pt
```

unless they are stored using an appropriate large-file solution such as Git LFS or an external model/artifact registry.

### Recommended `.gitignore`

```gitignore
# Model checkpoints / weights
*.pth
*.ckpt
*.pt

# Training outputs
runs/
wandb/
mlruns/
tensorboard/
```

If you want other developers to run inference immediately, publish the trained weights through an appropriate model/artifact hosting service and provide the download location in this section.

---

## Running Inference

The inference script is implemented in `test.py`.

It supports:

- Webcam input
- A single video file
- Multiple video files

The inference pipeline uses:

- OpenCV for video capture
- RF-DETR for object detection
- Supervision for bounding-box and label visualization

RF-DETR's prediction output is passed directly to Supervision for annotation. No separate NMS stage is required by the inference pipeline.

---

## Video Inference

Configure the video directory in `test.py`, then run:

```bash
python test.py
```

The script searches for common video formats including:

```text
.mp4
.avi
.mov
.mkv
```

and their uppercase variants.

If no videos are found, the script falls back to webcam inference.

---

## Inference Controls

During video inference:

| Key | Action |
|---|---|
| `Q` | Skip to the next video |
| `E` | Exit all videos |
| `D` | Fast-forward by 30 frames |
| `A` | Rewind by 30 frames |

The default production inference configuration uses a confidence threshold of `0.75` in the main video-processing path.

The underlying inference function supports configurable confidence thresholds.

Example:

```python
run_video_inference(
    video_source="path/to/video.mp4",
    conf_thres=0.75
)
```

---

## Using the Model with Your Own Video

You can modify the video directory in `test.py`:

```python
video_dir = r"path/to/your/videos"
```

Then run:

```bash
python test.py
```

The script will automatically search the directory for supported video formats.

---

## Webcam Inference

The inference function supports webcam input:

```python
run_video_inference(
    video_source=None,
    conf_thres=0.75
)
```

When `video_source=None`, OpenCV's default camera device is used.

---

## Detection Visualization

Detections are rendered using Supervision.

Each detection displays:

```text
<Class Name> <Confidence>
```

For example:

```text
Handgun 0.91
Rifle 0.87
Knife 0.78
```

The visualization also overlays video/frame information and the configured confidence threshold.

---

## Reproducibility

The training and inference code uses a fixed random seed:

```python
RANDOM_SEED = 42
```

Seeds are applied to:

- Python's `random`
- NumPy
- PyTorch
- CUDA

when CUDA is available.

This improves reproducibility between training runs, although complete bit-for-bit reproducibility can still depend on the underlying hardware, CUDA/PyTorch versions, and execution environment.

---

## Training Artifacts

The training pipeline generates several useful artifacts.

### `metrics.csv`

Contains per-epoch/logging-step training and validation metrics.

The training script consolidates these records to identify the best epoch according to mAP@0.5:0.95.

### `train_metadata.json`

Stores the best-model summary:

```json
{
    "epoch": 28,
    "precision": 0.957201,
    "recall": 0.935459,
    "f1_score": 0.946205,
    "mAP50": 0.966075,
    "mAP50-95": 0.848201
}
```

It also records the expected locations of the best, regular, EMA, and last checkpoint artifacts.

### `training_config.json`

Contains the recorded training configuration, including:

- Learning rates
- Batch size
- Gradient accumulation
- Epoch count
- EMA configuration
- Early stopping
- Dataset path
- Augmentation configuration
- Model architecture
- Encoder
- Resolution
- Number of classes

---

## Model Selection

The configuration supports multiple RF-DETR variants:

```text
RFDETRNano
RFDETRSmall
RFDETRBase
RFDETRMedium
RFDETRLarge
RFDETRXLarge
```

The current project configuration uses:

```python
MODEL_VARIANT = "RFDETRSmall"
```

RF-DETR Medium was selected as a practical balance between detection capability and computational requirements, particularly for deployment-oriented workloads.

---

## Deployment Considerations

This project is intended as a foundation for real-time or near-real-time weapon detection in surveillance environments.

For deployment, consider:

### Hardware

- NVIDIA GPU acceleration
- Sufficient VRAM for the selected RF-DETR variant
- Adequate CPU resources for video decoding and preprocessing
- Sufficient RAM for video buffering and data loading

### Video Pipeline

For production CCTV systems, consider:

- RTSP stream ingestion
- Frame skipping or adaptive sampling
- Multi-camera scheduling
- GPU-accelerated decoding
- Detection result logging
- Event-based alerting
- Temporal detection smoothing
- False-positive filtering
- Camera-specific confidence thresholds

### Edge Deployment

The current RF-DETR Medium configuration provides a balanced model variant suitable for resource-constrained deployment compared with larger RF-DETR variants.

However, actual throughput depends on:

- GPU
- Input resolution
- Number of cameras
- Video FPS
- Decoder performance
- Preprocessing overhead
- Detection threshold
- Concurrent inference streams

Benchmark the complete deployment pipeline on the target hardware before production use.

---

## Limitations

The model should not be considered a perfect or universal weapon detector.

Potential failure cases include:

- Extremely small weapons
- Heavy motion blur
- Severe compression
- Strong occlusion
- Poor illumination
- Unusual camera angles
- Severe fisheye distortion
- Objects visually similar to weapons
- Novel weapon types not sufficiently represented in training data
- Dense crowds
- Severe weather or environmental degradation

Model performance can also vary significantly between cameras and environments that differ from the training distribution.

For operational deployment, additional validation using representative footage from the target environment is strongly recommended.

---

## Safety and Responsible Use

This project is intended for **research, computer-vision development, surveillance-system prototyping, and authorized security applications**.

Weapon detection predictions should be treated as automated visual detections rather than definitive determinations.

A production security system should incorporate appropriate human review, operational safeguards, threshold calibration, auditing, and false-positive/false-negative analysis before taking consequential action based on model output.

Ensure that deployment complies with applicable laws, regulations, privacy requirements, organizational policies, and the rights of individuals captured by surveillance systems.

---

## Reproducing the Reported Results

To reproduce the training configuration:

1. Obtain the authorized dataset.
2. Place it under the expected dataset directory.
3. Install the required Python dependencies.
4. Configure the correct dataset path in `config.py`.
5. Ensure a compatible CUDA/PyTorch environment is available if GPU training is desired.
6. Run:

```bash
python train.py
```

The training configuration used for the reported run is preserved in `training_config.json`.

---

## Future Improvements

Potential future improvements include:

- Exporting the trained detector to ONNX
- TensorRT optimization
- FP16/INT8 inference optimization
- Jetson/edge deployment
- RTSP camera support
- Multi-camera inference
- Temporal detection tracking
- Alert/event management
- Per-class threshold optimization
- Hard-negative mining
- Additional CCTV-specific data collection
- Expanded weapon categories
- More extensive cross-camera validation
- Quantitative real-time FPS benchmarking

---

## Acknowledgements

This project is built around the **RF-DETR** object detection framework and uses supporting open-source computer-vision libraries including PyTorch, OpenCV, Albumentations, Supervision, NumPy, and Pillow.

The training configuration also uses a pretrained RF-DETR checkpoint as the starting point for model training.

Third-party libraries and pretrained model components remain subject to their respective licenses and terms.

---

## License

This project is released under the **MIT License**.

Copyright (c) 2026 Shakiru Sikiru

```text
MIT License

Copyright (c) 2026 Shakiru Sikiru

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Disclaimer

This repository provides software and research artifacts for object-detection experimentation. No guarantee is made that the detector will identify every weapon, avoid every false detection, or perform identically across all environments, cameras, datasets, or hardware configurations.

Always validate the system against the intended deployment environment before relying on its predictions for operational decisions.