"""
Configuration file for Super-Resolution GAN project.
Contains all hyperparameters and paths.
"""

import os

# ============================================================
# Dataset Paths
# ============================================================
# DIV2K root - Set this to your local dataset path
# You can download DIV2K from: https://data.vision.ee.ethz.ch/cvl/DIV2K/
# Or use kagglehub to download automatically (see train.py)
DIV2K_ROOT = "./data/DIV2K"

# Pre-trained model paths (optional - set to None if not available)
# These are optional warm-up weights that can improve training
SRCNN_PRETRAINED_PATH = None  # Set path if you have pre-trained SRCNN weights
SRRESNET_WARMUP_PATH = None   # Set path if you have SRResNet warm-up weights
ATTENTIVE_WARMUP_PATH = None # Set path if you have Attentive ESRGAN warm-up weights

# Test images directory (optional)
TEST_IMAGES_DIR = "./data/test_images"

# ============================================================
# Training Hyperparameters
# ============================================================
BATCH_SIZE = 16
HR_CROP_SIZE = 128
UPSCALE = 4
LR_CROP_SIZE = HR_CROP_SIZE // UPSCALE

# SRCNN hyperparameters
SRCNN_LEARNING_RATE = 1e-3

# SRGAN hyperparameters
SRGAN_GEN_LEARNING_RATE = 1e-4
SRGAN_DISC_LEARNING_RATE = 1e-4
SRGAN_ADV_LOSS_WEIGHT = 1e-3
SRGAN_CONTENT_LOSS_WEIGHT = 1.0
SRGAN_NUM_RES_BLOCKS = 16
SRGAN_EPOCHS = 30
SRGAN_STEPS_PER_EPOCH = 50

# Attentive ESRGAN hyperparameters
ESRGAN_GEN_LEARNING_RATE = 1e-4
ESRGAN_DISC_LEARNING_RATE = 5e-5
ESRGAN_BETA_1 = 0.9
ESRGAN_BETA_2 = 0.999
ESRGAN_CLIPNORM = 1.0
ESRGAN_CONTENT_LOSS_WEIGHT = 0.006
ESRGAN_ADV_LOSS_WEIGHT = 5e-3
ESRGAN_PIXEL_LOSS_WEIGHT = 1e-2
ESRGAN_NUM_RES_BLOCKS = 16
ESRGAN_EPOCHS = 30
ESRGAN_STEPS_PER_EPOCH = 50

# Channel attention ratio
CHANNEL_ATTENTION_RATIO = 16

# Validation settings
NUM_VAL_BATCHES = 3

# Model save settings
SAVE_MODELS = True
MODEL_SAVE_DIR = "./models"

