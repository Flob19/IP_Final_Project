# Super-Resolution GAN Project

A comprehensive implementation of Super-Resolution Generative Adversarial Networks (SRGAN) for image super-resolution, including baseline models and advanced variants.

## Project Status

This repository contains the complete source code, documentation, and configuration files for the Super-Resolution GAN project. The code is organized into modular components and ready for GitHub deployment.

## Overview

This project implements three different approaches to image super-resolution:

1. **SRCNN Baseline** - A PSNR-oriented convolutional neural network with 3 convolutional layers
2. **SRGAN Baseline** - SRResNet generator with BatchNorm, Binary Cross-Entropy loss, and VGG content loss
3. **Attentive ESRGAN** - Enhanced SRGAN with channel attention, no BatchNorm, and Relativistic Average Least Squares GAN (RaLSGAN) loss

The project focuses on 4x upscaling of low-resolution images using the DIV2K dataset.

## Prerequisites

### Coding Environment

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **GPU**: Recommended (CUDA-compatible GPU for faster training)

### Package Versions

Install the required packages using:

```bash
pip install -r requirements.txt
```

The project requires the following packages (see `requirements.txt` for exact versions):

- `tensorflow>=2.10.0` - Deep learning framework
- `numpy>=1.21.0` - Numerical computations
- `opencv-python>=4.5.0` - Image processing
- `matplotlib>=3.5.0` - Visualization
- `kagglehub>=0.1.0` - Optional: for automatic dataset downloading
- `protobuf==3.20.*` - Protocol buffers (specific version required)

## Project Structure

```
IP_Final_Project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration and hyperparameters
├── train.py                     # Command-line training script
├── evaluate.py                  # Command-line evaluation script
├── dip_final_project_srgan.py  # Original notebook code (reference)
├── .gitignore                   # Git ignore file
├── data/                        # Dataset directory (created by user)
│   └── DIV2K/                  # DIV2K dataset
├── models/                      # Trained models directory (auto-created)
│   ├── srgan_generator_epoch_*.keras
│   └── attentive_esrgan_epoch_*.keras
├── results/                     # Evaluation results (optional)
└── src/                         # Source code modules
    ├── __init__.py
    ├── data.py                  # Data loading and preprocessing
    ├── training.py              # Training functions
    ├── evaluation.py            # Evaluation functions (PSNR, SSIM)
    ├── visualization.py         # Visualization functions
    └── models/                  # Model modules
        ├── __init__.py
        ├── srcnn.py             # SRCNN model class
        ├── srgan.py             # SRGAN Generator and Discriminator classes
        └── attentive_esrgan.py  # Attentive ESRGAN Generator and Discriminator classes
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Flob19/IP_Final_Project.git
cd IP_Final_Project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download the DIV2K dataset from the official source:
- **Official website**: https://data.vision.ee.ethz.ch/cvl/DIV2K/
- Extract the dataset to a folder (e.g., `./data/DIV2K`)

Alternatively, you can use `kagglehub` for automatic download:
```bash
pip install kagglehub
```

Then edit `config.py` to set your dataset path:

```python
DIV2K_ROOT = "./data/DIV2K"  # Path to your DIV2K dataset
```

**Note**: The dataset structure should be:
```
DIV2K/
├── DIV2K_train_HR/
│   └── DIV2K_train_HR/
│       ├── 0001.png
│       ├── 0002.png
│       └── ...
└── DIV2K_valid_HR/
    └── DIV2K_valid_HR/
        ├── 0801.png
        ├── 0802.png
        └── ...
```

### 4. Training (Command-Line Interface)

The project supports command-line training with flexible options:

#### Train SRGAN Baseline

```bash
# Basic training with default parameters (30 epochs)
python train.py srgan

# Custom epochs and steps per epoch
python train.py srgan --epochs 50 --steps-per-epoch 100

# Save training plots to files
python train.py srgan --save-plots

# Quick test with 3 epochs
python train.py srgan --epochs 3 --steps-per-epoch 10
```

#### Train Attentive ESRGAN

```bash
# Basic training (30 epochs)
python train.py attentive-esrgan

# Custom training parameters
python train.py attentive-esrgan --epochs 40 --steps-per-epoch 80 --save-plots

# Quick test with 3 epochs
python train.py attentive-esrgan --epochs 3 --steps-per-epoch 10
```

#### Train SRCNN

```bash
python train.py srcnn
```

**Note**: Trained models are automatically saved to `./models/` directory.

**Optional**: For quick verification, you can use `test_training.py` to run both models with 3 epochs each:
```bash
python test_training.py
```

### 5. Evaluation (Command-Line Interface)

Evaluate trained models on test images:

#### Evaluate SRGAN Model

```bash
python evaluate.py srgan \
    --model-path models/srgan_generator_epoch_30.keras \
    --image path/to/test_image.jpg \
    --output-dir results/
```

#### Evaluate Attentive ESRGAN Model

```bash
python evaluate.py attentive-esrgan \
    --model-path models/attentive_esrgan_epoch_30.keras \
    --image path/to/test_image.jpg \
    --output-dir results/
```

#### Compare Two Models

```bash
python evaluate.py compare \
    --model-a models/srgan_generator_epoch_30.keras \
    --model-b models/attentive_esrgan_epoch_30.keras \
    --image path/to/test_image.jpg \
    --label-a "SRGAN Baseline" \
    --label-b "Attentive ESRGAN" \
    --output-dir results/
```

### 5. Programmatic Usage

You can also import and use individual components programmatically:

```python
from src.models.srgan import SRGANGenerator, SRGANDiscriminator
from src.models.attentive_esrgan import AttentiveESRGANGenerator, RelativisticDiscriminator
from src.training import train_srgan_baseline, train_attentive_esrgan
from src.data import SRGANDataGenerator
from src.evaluation import evaluate_gan_model
from src.visualization import plot_gan_results

# Build models
generator = SRGANGenerator(scale=4, num_res_blocks=16)
discriminator = SRGANDiscriminator(input_shape=(128, 128, 3))

# Prepare data
train_gen = SRGANDataGenerator(train_dir, batch_size=16, crop_size=128, scale_factor=4)

# Train
history = train_srgan_baseline(generator.model, discriminator.model, ...)

# Evaluate
results = evaluate_gan_model(generator.model, "path/to/image.jpg")
plot_gan_results(results, model_name="SRGAN")
```

## Hyperparameters

All hyperparameters are defined in `config.py`. Key settings include:

### General Training Parameters

- **BATCH_SIZE**: 16
- **HR_CROP_SIZE**: 128 (high-resolution patch size)
- **UPSCALE**: 4 (4x super-resolution)
- **LR_CROP_SIZE**: 32 (low-resolution patch size, calculated as HR_CROP_SIZE // UPSCALE)

### SRCNN Hyperparameters

- **Learning Rate**: 1e-3
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam

### SRGAN Baseline Hyperparameters

- **Generator Learning Rate**: 1e-4
- **Discriminator Learning Rate**: 1e-4
- **Adversarial Loss Weight**: 1e-3
- **Content Loss Weight**: 1.0
- **Number of Residual Blocks**: 16
- **Epochs**: 30
- **Steps per Epoch**: 50
- **Loss Function**: Binary Cross-Entropy (BCE) for discriminator, BCE + VGG MSE for generator

### Attentive ESRGAN Hyperparameters

- **Generator Learning Rate**: 1e-4
- **Discriminator Learning Rate**: 5e-5
- **Beta1**: 0.9
- **Beta2**: 0.999
- **Gradient Clipping**: 1.0
- **Content Loss Weight**: 0.006
- **Adversarial Loss Weight**: 5e-3
- **Pixel Loss Weight**: 1e-2
- **Number of Residual Blocks**: 16
- **Epochs**: 30
- **Steps per Epoch**: 50
- **Channel Attention Ratio**: 16
- **Loss Function**: RaLSGAN (Relativistic Average Least Squares GAN) + VGG perceptual loss + L1 pixel loss

## Experiment Results

### Model Architectures

#### SRCNN
- **Architecture**: 3 convolutional layers (64@9x9 → 32@1x1 → 3@5x5)
- **Purpose**: PSNR-oriented baseline
- **Input/Output**: RGB images in [0,1] range

#### SRGAN Baseline
- **Generator**: SRResNet with BatchNorm
  - Initial conv layer (64 filters, 9x9)
  - 16 residual blocks with BatchNorm
  - 2 upsampling blocks (PixelShuffle x2 each)
  - Output layer with tanh activation ([-1,1] range)
- **Discriminator**: 8-layer CNN with BatchNorm
- **Loss**: BCE adversarial loss + VGG19 perceptual loss

#### Attentive ESRGAN
- **Generator**: Enhanced SRResNet without BatchNorm
  - Channel attention mechanism in residual blocks
  - Residual scaling factor: 0.2
  - No BatchNorm layers
  - Same upsampling structure as SRGAN
- **Discriminator**: Relativistic discriminator (outputs logits)
- **Loss**: RaLSGAN + VGG19 perceptual loss + L1 pixel loss

### Training Process

1. **Data Preprocessing**:
   - Random cropping of HR patches (128x128)
   - Bicubic downsampling to create LR patches (32x32)
   - Normalization to [-1, 1] range

2. **Training Strategy**:
   - Alternating training of discriminator and generator
   - Validation metrics (PSNR, SSIM) computed every epoch
   - Model checkpoints saved after each epoch

3. **Evaluation Metrics**:
   - **PSNR** (Peak Signal-to-Noise Ratio): Higher is better
   - **SSIM** (Structural Similarity Index): Higher is better (range [0,1])

### Expected Performance

The models are evaluated on the DIV2K validation set. Typical results:

- **Bicubic Interpolation**: Baseline for comparison
- **SRGAN Baseline**: Improved PSNR and SSIM over bicubic
- **Attentive ESRGAN**: Further improvements with better perceptual quality

*Note: Actual results depend on training duration, dataset quality, and hardware capabilities.*

## Dataset

The project uses the **DIV2K** dataset:
- **Training Set**: 800 high-resolution images
- **Validation Set**: 100 high-resolution images
- **Resolution**: Various resolutions (minimum 128x128 patches)

### Download Instructions

1. **Manual Download** (Recommended):
   - Visit: https://data.vision.ee.ethz.ch/cvl/DIV2K/
   - Download the training and validation HR images
   - Extract to `./data/DIV2K/` or update `DIV2K_ROOT` in `config.py`

2. **Automatic Download** (Optional):
   - Install kagglehub: `pip install kagglehub`
   - The script will prompt to download automatically if dataset is not found

## Model Files

Trained models are saved in the `./models/` directory:
- `srgan_generator_epoch_{N}.keras` - SRGAN baseline checkpoints
- `attentive_esrgan_epoch_{N}.keras` - Attentive ESRGAN checkpoints

## Visualization

The project includes visualization utilities:
- Training loss curves (Discriminator vs Generator)
- Validation metrics plots (PSNR and SSIM over epochs)
- Side-by-side image comparisons (Bicubic vs Model vs Ground Truth)

## Notes

- Training requires significant computational resources (GPU recommended)
- Model warm-up weights are optional - training can start from scratch
- All paths in `config.py` are set for local development by default
- The dataset path must be configured before training (see Usage section)

## References

- SRCNN: "Image Super-Resolution Using Deep Convolutional Networks" (Dong et al., 2014)
- SRGAN: "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" (Ledig et al., 2017)
- ESRGAN: "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks" (Wang et al., 2018)
- RCAN: Zhang, Y., Li, K., Li, K., Wang, L., Zhong, B., and Fu, Y., "Image Super-Resolution Using Very Deep Residual Channel Attention Networks", arXiv e-prints, Art. no. arXiv:1807.02758, 2018. doi:10.48550/arXiv.1807.02758.

## Authors

A. B. Bahi, Felix Floberg, Simon Que, 黎裴方東

