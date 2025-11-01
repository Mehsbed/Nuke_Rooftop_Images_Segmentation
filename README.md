# 🏠 Rooftop Segmentation with DeepLabV3Plus

---

<div align="center">

# 📅 **WEEK 1 PROGRESS**

### **Internship Project - EduNet**

**This README documents the Week 1 implementation and progress of the Rooftop Segmentation project.**

**Internship Duration**: [27th October 2025] to [27th November 2025]  
**Organization**: EduNet  
**Week**: 1 of 3

</div>

---

A deep learning project for semantic segmentation of rooftops in aerial/satellite images using DeepLabV3Plus architecture with PyTorch.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Week](https://img.shields.io/badge/Week-1-yellow.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange.svg)

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Dataset Structure](#-dataset-structure)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Training Details](#-training-details)
- [Results](#-results)
- [Key Features Explained](#-key-features-explained)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

This project implements a state-of-the-art semantic segmentation model to detect and segment rooftops in aerial/satellite images. The model uses DeepLabV3Plus with a pretrained ResNet50 backbone and includes various optimizations for efficient training and better performance.

**Key Highlights:**
- ✅ Semantic segmentation (pixel-wise classification)
- ✅ Pretrained ResNet50 encoder for transfer learning
- ✅ Mixed precision training for faster training
- ✅ Combined loss function (CrossEntropy + Dice Loss)
- ✅ Learning rate scheduling
- ✅ GPU acceleration support
- ✅ Comprehensive evaluation metrics (IoU, Pixel Accuracy)

## ✨ Features

### Model Features
- **DeepLabV3Plus Architecture**: State-of-the-art segmentation model
- **Pretrained Backbone**: ResNet50 encoder pretrained on ImageNet
- **Transfer Learning**: Leverages pretrained weights for better performance

### Training Features
- **Mixed Precision Training**: 2x faster training with FP16
- **Combined Loss Function**: CrossEntropy (60%) + Dice Loss (40%)
- **Learning Rate Scheduling**: Automatic LR reduction on plateau
- **Gradient Clipping**: Prevents gradient explosion
- **Early Stopping**: Stops training when validation metrics plateau
- **Model Checkpointing**: Saves best models based on IoU and Loss

### Evaluation Features
- **Pixel Accuracy (PA)**: Percentage of correctly classified pixels
- **Mean Intersection over Union (mIoU)**: Better metric for segmentation
- **Learning Curves Visualization**: Track training progress
- **Inference Visualization**: Side-by-side comparison of predictions

## 📦 Requirements

### Python Version
- Python 3.7 or higher

### Core Libraries
```
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.0
Pillow>=8.0.0
matplotlib>=3.3.0
albumentations>=1.0.0
opencv-python>=4.5.0
segmentation-models-pytorch>=0.3.0
tqdm>=4.60.0
```

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **CPU**: Can run on CPU but significantly slower
- **Memory**: Minimum 8GB RAM, 16GB+ recommended
- **Storage**: Space for dataset and saved models

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Mehsbed/Nuke_Rooftop_Images_Segmentation.git
cd Nuke_Rooftop_Images_Segmentation
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install PyTorch** (with CUDA support for GPU):
```bash
# For CUDA 11.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111

# For CPU only
pip install torch torchvision torchaudio
```

4. **Install other dependencies**:
```bash
pip install numpy pillow matplotlib albumentations opencv-python segmentation-models-pytorch tqdm
```

Or install from requirements.txt (if provided):
```bash
pip install -r requirements.txt
```

## 📁 Dataset Structure

Organize your dataset in the following structure:

```
dataset/
├── images/
│   └── images/
│       ├── image_001.tif
│       ├── image_002.tif
│       └── ...
└── label/
    └── label/
        ├── image_001_label.tif
        ├── image_002_label.tif
        └── ...
```

**Dataset Requirements:**
- Images should be in RGB format (.tif, .jpg, .png, etc.)
- Labels should be grayscale masks (.tif, .png, etc.)
- Label files should follow naming: `{image_name}_label.tif`
- Labels should have pixel values: 0 (background), 255 (rooftop)
- Recommended image size: 256×256 pixels (will be resized automatically)

## 💻 Usage

### Running the Notebook

1. **Open Jupyter Notebook**:
```bash
jupyter notebook main.ipynb
```

2. **Run cells sequentially**:
   - **Cell 0**: Data loading and setup
   - **Cell 2**: Data visualization (optional)
   - **Cell 4**: Model setup and configuration
   - **Cell 6**: Training the model
   - **Cell 8**: Visualize learning curves
   - **Cell 10**: Run inference on test set

### Configuration

You can modify these parameters in the code:

```python
# Data Configuration
root = "dataset"                          # Dataset directory
batch_size = 32                           # Batch size
split = [0.9, 0.05, 0.05]                # Train/Val/Test split
image_size = 256                          # Image dimensions

# Model Configuration
encoder_name = "resnet50"                 # Backbone: resnet50, resnet101, efficientnet-b0
encoder_weights = "imagenet"              # Pretrained weights
num_classes = 2                            # Background + Rooftop

# Training Configuration
epochs = 50                                # Number of training epochs
learning_rate = 1e-3                       # Initial learning rate
weight_decay = 1e-4                        # L2 regularization
gradient_clip = 1.0                        # Gradient clipping threshold
```

### Training

The training process will:
1. Load and preprocess the dataset
2. Split into train/validation/test sets
3. Initialize the model with pretrained weights
4. Train with mixed precision
5. Validate after each epoch
6. Save best models based on IoU and Loss
7. Display progress and metrics

### Inference

After training, the inference cell will:
1. Load the best trained model
2. Run predictions on test set
3. Visualize results side-by-side (Original, Ground Truth, Prediction)

## 📂 Project Structure

```
projects/
├── main.ipynb                    # Main notebook with all code
├── README.md                     # This file
├── CODE_EXPLANATION.md           # Detailed code explanation
├── dataset/                      # Dataset directory
│   ├── images/
│   │   └── images/
│   └── label/
│       └── label/
└── saved_models/                 # Trained models (created after training)
    ├── rooftop_best_model_iou.pt
    └── rooftop_best_model_loss.pt
```

## 🏗️ Model Architecture

### DeepLabV3Plus

DeepLabV3Plus is a state-of-the-art semantic segmentation architecture that combines:
- **Encoder**: ResNet50 backbone for feature extraction
- **Atrous Spatial Pyramid Pooling (ASPP)**: Multi-scale feature extraction
- **Decoder**: Upsampling path for high-resolution segmentation

**Architecture Components:**
- **Input**: 3-channel RGB images (256×256)
- **Encoder**: ResNet50 (pretrained on ImageNet)
- **ASPP**: Captures multi-scale context
- **Decoder**: Refines segmentation boundaries
- **Output**: 2-channel logits (background, rooftop)

**Total Parameters**: ~40M (with ResNet50)

## 🎓 Training Details

### Loss Function

**CombinedLoss = 0.6 × CrossEntropy + 0.4 × Dice Loss**

- **CrossEntropy Loss**: Standard classification loss
- **Dice Loss**: Better for segmentation, handles class imbalance
- **Combined**: Gets benefits of both losses

### Optimizer

**AdamW** with:
- Learning Rate: 1e-3
- Weight Decay: 1e-4
- Adaptive learning rates per parameter

### Learning Rate Scheduling

**ReduceLROnPlateau**:
- Monitors validation loss
- Reduces LR by factor of 0.5 when loss plateaus
- Patience: 5 epochs
- Minimum LR: 1e-6

### Mixed Precision Training

Uses **FP16** (half precision) for:
- 2x faster training
- 50% less GPU memory
- Minimal accuracy loss

### Training Optimizations

- **Gradient Clipping**: Prevents gradient explosion (max norm: 1.0)
- **Non-blocking Transfers**: Faster GPU data transfer
- **Early Stopping**: Stops if no improvement for 10 epochs
- **Model Checkpointing**: Saves best models based on IoU and Loss

## 📊 Results

### Metrics Explained

1. **Pixel Accuracy (PA)**
   - Percentage of correctly classified pixels
   - Formula: `Correct Pixels / Total Pixels`
   - Range: 0 to 1 (higher is better)

2. **Mean Intersection over Union (mIoU)**
   - Better metric for segmentation
   - Measures overlap between prediction and ground truth
   - Formula: `IoU = Intersection / Union`
   - Range: 0 to 1 (higher is better)

3. **Loss**
   - Combined loss value
   - Lower is better

### Expected Performance

With the default configuration:
- **Training Time**: ~2-3 hours (on GPU)
- **mIoU**: 0.70-0.85 (depends on dataset quality)
- **Pixel Accuracy**: 0.90-0.95

### Visualizing Results

The notebook includes:
- **Learning Curves**: Track training progress over epochs
- **Inference Results**: Visual comparison of predictions vs ground truth

## 🔑 Key Features Explained

### Why Pretrained Backbone?
- ResNet50 learned useful features on ImageNet (edges, textures, shapes)
- Transfer learning: Reuse features, only learn segmentation-specific parts
- Faster convergence and better results

### Why Combined Loss?
- CrossEntropy: Good for classification
- Dice Loss: Better for segmentation with class imbalance
- Combined: Gets benefits of both

### Why Mixed Precision?
- Uses 16-bit floats instead of 32-bit
- 2x faster training
- 50% less GPU memory

### Why IoU Metric?
- Pixel Accuracy can be misleading (e.g., most pixels are background)
- IoU measures overlap, better for segmentation evaluation
- Handles class imbalance better

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Some areas for improvement:

- [ ] Add more backbone options (EfficientNet, Vision Transformer)
- [ ] Implement data augmentation strategies
- [ ] Add more evaluation metrics
- [ ] Support for multi-class segmentation
- [ ] Export model to ONNX/TensorRT for deployment

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- **DeepLabV3Plus**: [Encoder-Decoder with Atrous Separable Convolution](https://arxiv.org/abs/1802.02611)
- **Segmentation Models PyTorch**: https://github.com/qubvel/segmentation_models.pytorch
- **Albumentations**: https://albumentations.ai/

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

---

## 📝 Progress Tracking

### Week 1 Status ✅

**Completed:**
- ✅ Project setup and environment configuration
- ✅ Custom dataset class implementation for image segmentation
- ✅ Data loading and preprocessing pipeline
- ✅ DeepLabV3Plus model architecture setup with ResNet50 backbone
- ✅ Combined loss function (CrossEntropy + Dice Loss)
- ✅ Training loop with mixed precision training
- ✅ Evaluation metrics (IoU, Pixel Accuracy)
- ✅ Learning curve visualization
- ✅ Inference pipeline

**Next Steps (Week 2):**
- [ ] Experiment with different backbone architectures
- [ ] Implement data augmentation strategies
- [ ] Hyperparameter tuning and optimization
- [ ] Model deployment preparation

**Updates**: This README will be updated weekly to track project progress.

---

**Note**: This project is for educational purposes as part of the EduNet internship program. Make sure you have proper permissions and licenses for any datasets you use.

**Happy Coding! 🚀**

