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

This project is for semantic segmentation of rooftops in aerial/satellite images. I'm using DeepLabV3Plus with ResNet50 as the backbone, which I've pretrained on ImageNet. I've added some optimizations to make training faster and get better results.

**What's included:**
- Semantic segmentation (classifying each pixel)
- ResNet50 encoder pretrained on ImageNet
- Mixed precision training (makes it faster)
- Combined loss function (CrossEntropy + Dice Loss)
- Learning rate scheduling
- GPU support
- Metrics: IoU and Pixel Accuracy

## ✨ Features

### Model Features
- DeepLabV3Plus architecture
- ResNet50 pretrained on ImageNet
- Uses transfer learning

### Training Features
- Mixed precision training (FP16) - about 2x faster
- Loss function combines CrossEntropy (60%) and Dice Loss (40%)
- Learning rate automatically reduces when validation loss stops improving
- Gradient clipping to prevent training issues
- Early stopping after 10 epochs without improvement
- Saves best models based on IoU and loss

### Evaluation Features
- Pixel Accuracy - percentage of correct pixels
- mIoU (mean IoU) - better metric for segmentation tasks
- Plots learning curves
- Shows side-by-side comparison of predictions vs ground truth

## 📦 Requirements

- Python 3.7 or higher

### Required Libraries
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
- GPU with CUDA support recommended (much faster)
- Can run on CPU but will be slow
- At least 8GB RAM (16GB+ is better)
- Enough storage for your dataset and saved models

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

**Requirements:**
- Images in RGB format (.tif, .jpg, .png work fine)
- Labels as grayscale masks
- Label naming: `{image_name}_label.tif`
- Label values: 0 for background, 255 for rooftop
- Images will be resized to 256x256 automatically

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

When you run the training:
1. Loads and preprocesses the dataset
2. Splits into train/val/test sets (90/5/5 by default)
3. Sets up model with pretrained ResNet50 weights
4. Trains using mixed precision
5. Validates after each epoch
6. Saves best models (based on IoU and loss)
7. Shows progress and metrics

### Inference

After training:
1. Loads the best model
2. Runs predictions on test set
3. Shows side-by-side: Original image, Ground Truth, Prediction

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

Using DeepLabV3Plus which has:
- Encoder: ResNet50 for feature extraction
- ASPP: Multi-scale feature extraction
- Decoder: Upsamples to get full resolution segmentation

Input is 256x256 RGB images, output is 2 classes (background and rooftop).
Model has around 40M parameters with ResNet50.

## 🎓 Training Details

### Loss Function

Combined loss: 60% CrossEntropy + 40% Dice Loss
- CrossEntropy is the standard classification loss
- Dice Loss works better for segmentation, especially with imbalanced classes
- Combining both gives good results

### Optimizer

Using AdamW optimizer:
- Learning rate: 0.001
- Weight decay: 0.0001

### Learning Rate Scheduling

ReduceLROnPlateau scheduler:
- Watches validation loss
- If loss doesn't improve for 5 epochs, reduces LR by half
- Minimum LR is 1e-6

### Mixed Precision Training

Using FP16 (half precision):
- Makes training about 2x faster
- Uses less GPU memory
- Accuracy stays almost the same

### Training Optimizations

- Gradient clipping (max norm 1.0) to prevent gradient explosion
- Non-blocking GPU transfers for speed
- Early stopping after 10 epochs without improvement
- Saves best models based on IoU and loss

## 📊 Results

### Metrics Explained

1. **Pixel Accuracy (PA)**
   - Percentage of pixels classified correctly
   - Higher is better (0 to 1)

2. **Mean IoU (mIoU)**
   - Better metric for segmentation than pixel accuracy
   - Measures how much prediction and ground truth overlap
   - Higher is better (0 to 1)

3. **Loss**
   - Combined loss value
   - Lower is better

### Expected Performance

With default settings (GPU):
- Training time: around 2-3 hours
- mIoU: usually 0.70-0.85 (depends on your dataset)
- Pixel Accuracy: usually 0.90-0.95

The notebook plots learning curves and shows inference results visually.

## 🔑 Why These Choices?

**Pretrained Backbone**: ResNet50 already learned useful features from ImageNet, so I reuse those and just learn the segmentation part. Much faster and better results.

**Combined Loss**: CrossEntropy is good for classification, Dice Loss is better for segmentation especially with imbalanced classes. Using both works well.

**Mixed Precision**: Using FP16 instead of FP32 makes training 2x faster and uses half the GPU memory with almost no accuracy loss.

**IoU Metric**: Pixel accuracy can be misleading (like if most pixels are background). IoU measures actual overlap which is better for segmentation tasks.

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

If you have questions or run into issues, feel free to open an issue on GitHub.

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

**Note**: This is part of my EduNet internship project. Make sure you have the right permissions for any datasets you use.

Thanks for checking it out! 🚀

