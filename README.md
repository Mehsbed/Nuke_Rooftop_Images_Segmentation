# 🏠 Rooftop Segmentation with DeepLabV3Plus



---



<div align="center">



# 📅 **WEEK 3 PROGRESS**



### **Internship Project - EduNet**



**Final project submission week 3**



**Internship Duration**: [27th October 2025] to [27th November 2025]  

**Organization**: EduNet  

**Week**: 3 of 3



</div>



---



A complete deep learning project for semantic segmentation of rooftops in aerial/satellite images using DeepLabV3Plus architecture with PyTorch.



![Python](https://img.shields.io/badge/python-3.7+-blue.svg)

![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)

![Week](https://img.shields.io/badge/Week-3-green.svg)

![Status](https://img.shields.io/badge/Status-Final%20Submission-success.svg)



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

- [Project Improvements & Progress](#-project-improvements--progress)

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

- ✅ Complete end-to-end pipeline from data preprocessing to inference

- ✅ Automatic model checkpointing and best model selection

- ✅ Real-time training monitoring and visualization



## ✨ Features



### Model Features

- **DeepLabV3Plus Architecture**: State-of-the-art segmentation model

- **Pretrained Backbone**: ResNet50 encoder pretrained on ImageNet

- **Transfer Learning**: Leverages pretrained weights for better performance

- **2-Class Segmentation**: Background and Rooftop classification



### Training Features

- **Mixed Precision Training**: 2x faster training with FP16

- **Combined Loss Function**: CrossEntropy (60%) + Dice Loss (40%)

- **Learning Rate Scheduling**: Automatic LR reduction on plateau (ReduceLROnPlateau)

- **Gradient Clipping**: Prevents gradient explosion (max norm: 1.0)

- **Early Stopping**: Stops training when validation metrics plateau (10 epochs patience)

- **Model Checkpointing**: Saves best models based on IoU and Loss metrics

- **Non-blocking GPU Transfers**: Optimized data loading for faster training



### Evaluation Features

- **Pixel Accuracy (PA)**: Percentage of correctly classified pixels

- **Mean Intersection over Union (mIoU)**: Better metric for segmentation

- **Learning Curves Visualization**: Track training progress over epochs

- **Inference Visualization**: Side-by-side comparison of predictions vs ground truth

- **Real-time Metrics**: Display training and validation metrics during training



### Additional Features

- **Dataset Verification Utility**: Check dataset structure and format before training

- **Model Setup Testing**: Verify environment and model initialization

- **Data Visualization Tools**: Visualize dataset samples and augmentations

- **Complete Training Pipeline**: End-to-end workflow from data loading to inference

- **Automatic Best Model Loading**: Loads best model for inference automatically



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

jupyter>=1.0.0

ipykernel>=6.0.0

```



### Hardware

- **GPU**: NVIDIA GPU with CUDA support (recommended)

- **CPU**: Can run on CPU but significantly slower

- **Memory**: Minimum 8GB RAM, 16GB+ recommended

- **Storage**: Space for dataset and saved models



## 🚀 Installation



1. **Clone the repository**:

```bash

git clone <repository-url>

cd projects

```



2. **Create a virtual environment** (recommended):

```bash

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

```



3. **Install PyTorch** (with CUDA support for GPU):

```bash

# For CUDA 11.8

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118



# For CPU only

pip install torch torchvision torchaudio

```



4. **Install other dependencies**:

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

- Recommended image size: 256×256 pixels (will be resized automatically during preprocessing)



## 💻 Usage



### Running the Notebook



1. **Open Jupyter Notebook**:

```bash

jupyter notebook main.ipynb

```



2. **Run cells sequentially**:

   - **Section 1**: Imports and Setup

   - **Section 2**: Dataset Verification (optional - check your dataset first)

   - **Section 3**: Model Setup Test (optional - verify environment)

   - **Section 4**: Dataset Class and Data Loading

   - **Section 5**: Load Dataset

   - **Section 6**: Visualize Dataset (optional)

   - **Section 7**: Model Setup

   - **Section 8**: Loss Function and Metrics

   - **Section 9**: Training Function

   - **Section 10**: Start Training

   - **Section 11**: Visualize Learning Curves

   - **Section 12**: Inference on Test Set



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

num_classes = 2                           # Background + Rooftop



# Training Configuration

epochs = 50                               # Number of training epochs

learning_rate = 1e-3                      # Initial learning rate

weight_decay = 1e-4                       # L2 regularization

gradient_clip = 1.0                       # Gradient clipping threshold

```



### Training



The training process will:

1. Load and preprocess the dataset

2. Split into train/validation/test sets (90/5/5 by default)

3. Initialize the model with pretrained weights

4. Train with mixed precision (FP16)

5. Validate after each epoch

6. Save best models based on IoU and Loss

7. Display progress and metrics in real-time

8. Apply early stopping if no improvement for 10 epochs



### Inference



After training, the inference cell will:

1. Load the best trained model (best IoU or best Loss)

2. Run predictions on test set

3. Visualize results side-by-side (Original, Ground Truth, Prediction)



## 📂 Project Structure



```

projects/

├── main.ipynb                    # Main notebook with all code

├── README.md                     # This file

├── PPT_Content.md                # PowerPoint presentation content

├── requirements.txt              # Python dependencies

├── LICENSE                       # MIT License

├── .gitignore                    # Git ignore rules

├── .gitattributes                # Git attributes (Git LFS configuration)

├── dataset/                      # Dataset directory (not in git)

│   ├── images/

│   │   └── images/

│   │       ├── image_001.tif

│   │       └── ...

│   └── label/

│       └── label/

│           ├── image_001_label.tif

│           └── ...

└── saved_models/                 # Trained models (not in git)

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

- **ASPP**: Captures multi-scale context with atrous convolutions

- **Decoder**: Refines segmentation boundaries

- **Output**: 2-channel logits (background, rooftop)



**Total Parameters**: ~40M (with ResNet50)



## 🎓 Training Details



### Loss Function



**CombinedLoss = 0.6 × CrossEntropy + 0.4 × Dice Loss**



- **CrossEntropy Loss**: Standard classification loss, handles class probabilities

- **Dice Loss**: Better for segmentation, handles class imbalance effectively

- **Combined**: Gets benefits of both losses for robust training



### Optimizer



**AdamW** with:

- Learning Rate: 1e-3

- Weight Decay: 1e-4

- Adaptive learning rates per parameter

- Better generalization than standard Adam



### Learning Rate Scheduling



**ReduceLROnPlateau**:

- Monitors validation loss

- Reduces LR by factor of 0.5 when loss plateaus

- Patience: 5 epochs

- Minimum LR: 1e-6

- Helps fine-tune model when validation loss stops improving



### Mixed Precision Training



Uses **FP16** (half precision) for:

- 2x faster training

- 50% less GPU memory

- Minimal accuracy loss

- Automatic mixed precision (AMP) with PyTorch



### Training Optimizations



- **Gradient Clipping**: Prevents gradient explosion (max norm: 1.0)

- **Non-blocking Transfers**: Faster GPU data transfer with `pin_memory=True`

- **Early Stopping**: Stops if no improvement for 10 epochs

- **Model Checkpointing**: Saves best models based on IoU and Loss metrics

- **Windows Compatibility**: Automatic `num_workers=0` for Windows to avoid multiprocessing issues



## 📊 Results



### Metrics Explained



1. **Pixel Accuracy (PA)**

   - Percentage of correctly classified pixels

   - Formula: `Correct Pixels / Total Pixels`

   - Range: 0 to 1 (higher is better)

   - Can be misleading if classes are imbalanced



2. **Mean Intersection over Union (mIoU)**

   - Better metric for segmentation tasks

   - Measures overlap between prediction and ground truth

   - Formula: `IoU = Intersection / Union`

   - Range: 0 to 1 (higher is better)

   - More meaningful for imbalanced datasets



3. **Loss**

   - Combined loss value (CrossEntropy + Dice Loss)

   - Lower is better

   - Tracks training and validation loss separately



### Expected Performance



With the default configuration:

- **Training Time**: ~2-3 hours (on GPU, depends on dataset size)

- **mIoU**: 0.70-0.85 (depends on dataset quality)

- **Pixel Accuracy**: 0.90-0.95

- **Model Size**: ~40M parameters

- **Inference Speed**: Real-time on GPU



### Visualizing Results



The notebook includes:

- **Learning Curves**: Track training progress over epochs (loss, IoU, Pixel Accuracy)

- **Inference Results**: Visual comparison of predictions vs ground truth

- **Real-time Metrics**: Display metrics during training with progress bars



## 🔑 Key Features Explained



### Why Pretrained Backbone?

- ResNet50 learned useful features on ImageNet (edges, textures, shapes)

- Transfer learning: Reuse features, only learn segmentation-specific parts

- Faster convergence and better results

- Reduces training time significantly



### Why Combined Loss?

- CrossEntropy: Good for classification, handles probabilities well

- Dice Loss: Better for segmentation with class imbalance

- Combined: Gets benefits of both, more robust training

- Better boundary detection and class balance



### Why Mixed Precision?

- Uses 16-bit floats instead of 32-bit

- 2x faster training

- 50% less GPU memory

- Minimal accuracy loss (usually <1%)

- Industry standard for deep learning training



### Why IoU Metric?

- Pixel Accuracy can be misleading (e.g., most pixels are background)

- IoU measures overlap, better for segmentation evaluation

- Handles class imbalance better

- More meaningful for real-world applications



### Why Early Stopping?

- Prevents overfitting

- Saves training time

- Automatically stops when model stops improving

- Patience of 10 epochs ensures we don't stop too early



## 📝 Project Improvements & Progress



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



### Week 2 Status ✅



**Completed:**

- ✅ Hyperparameter tuning and optimization

- ✅ Model checkpointing system (best IoU and best Loss)

- ✅ Early stopping implementation

- ✅ Learning rate scheduling (ReduceLROnPlateau)

- ✅ Gradient clipping for training stability

- ✅ Enhanced visualization tools

- ✅ Dataset verification utilities

- ✅ Model setup testing utilities

- ✅ Windows compatibility improvements (num_workers handling)

- ✅ Code organization and documentation



### Week 3 Status ✅ (Final Submission)



**Completed:**

- ✅ Complete project cleanup and optimization

- ✅ Removed unnecessary files (.ipynb_checkpoints, temporary files)

- ✅ Cleared notebook outputs for cleaner git commits

- ✅ Updated .gitignore for comprehensive file exclusion

- ✅ Fixed .gitignore to protect .ipynb files (removed *.json exclusion)

- ✅ Created comprehensive README documentation

- ✅ Added PowerPoint presentation content (PPT_Content.md)

- ✅ Final code review and testing

- ✅ Project structure optimization

- ✅ Git repository preparation

- ✅ Final model training and evaluation

- ✅ Complete documentation of all improvements



**Project Summary:**

This project has evolved from a basic implementation in Week 1 to a complete, production-ready solution in Week 3. All code is consolidated in a single Jupyter notebook (`main.ipynb`) for easy execution. The project includes comprehensive documentation, proper git configuration, and all necessary utilities for dataset verification, model testing, and result visualization.



**Key Achievements:**

- 🎯 Complete end-to-end pipeline from data preprocessing to inference

- 🚀 Optimized training with mixed precision (2x faster)

- 📊 Comprehensive evaluation with multiple metrics

- 💾 Automatic model checkpointing and best model selection

- 📈 Real-time training monitoring and visualization

- 🧹 Clean, organized, and production-ready codebase

- 📚 Complete documentation and presentation materials



## 🤝 Contributing



Contributions are welcome! Please feel free to submit a Pull Request. Some areas for improvement:



- [ ] Add more backbone options (EfficientNet, Vision Transformer)

- [ ] Implement data augmentation strategies

- [ ] Add more evaluation metrics (F1-Score, Precision, Recall)

- [ ] Support for multi-class segmentation

- [ ] Export model to ONNX/TensorRT for deployment

- [ ] Add support for different image sizes

- [ ] Implement test-time augmentation

- [ ] Add model ensemble capabilities



## 📄 License



This project is licensed under the MIT License - see the LICENSE file for details.



## 📚 References



- **DeepLabV3Plus**: [Encoder-Decoder with Atrous Separable Convolution](https://arxiv.org/abs/1802.02611)

- **Segmentation Models PyTorch**: https://github.com/qubvel/segmentation_models.pytorch

- **Albumentations**: https://albumentations.ai/

- **PyTorch Mixed Precision Training**: https://pytorch.org/docs/stable/amp.html



## 📧 Contact



For questions or issues, please open an issue on GitHub.



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



### Week 2 Status ✅



**Completed:**

- ✅ Hyperparameter tuning and optimization

- ✅ Model checkpointing system (best IoU and best Loss)

- ✅ Early stopping implementation

- ✅ Learning rate scheduling (ReduceLROnPlateau)

- ✅ Gradient clipping for training stability

- ✅ Enhanced visualization tools

- ✅ Dataset verification utilities

- ✅ Model setup testing utilities

- ✅ Windows compatibility improvements

- ✅ Code organization and documentation



### Week 3 Status ✅ (Final Submission)



**Completed:**

- ✅ Complete project cleanup and optimization

- ✅ Removed unnecessary files and cleared notebook outputs

- ✅ Updated .gitignore and .gitattributes for proper version control

- ✅ Created comprehensive README documentation

- ✅ Added PowerPoint presentation content

- ✅ Final code review and testing

- ✅ Project structure optimization

- ✅ Git repository preparation

- ✅ Final model training and evaluation

- ✅ Complete documentation of all improvements



**Final Project Status**: ✅ **COMPLETE - READY FOR SUBMISSION**



---



**Note**: This project is for educational purposes as part of the EduNet internship program. Make sure you have proper permissions and licenses for any datasets you use.



**Happy Coding! 🚀**
