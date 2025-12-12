"""
Main training script for Super-Resolution GAN project.
Supports training SRCNN, SRGAN baseline, and Attentive ESRGAN models.
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras

# Optional: kagglehub for automatic dataset download
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    print("Note: kagglehub not available. Please download DIV2K dataset manually.")

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data import SRGANDataGenerator, resolve_div2k_paths
from src.models import (
    build_srcnn,
    build_srgan_generator,
    build_srgan_discriminator,
    build_vgg,
    build_srgan_combined,
    build_attentive_generator,
    build_relativistic_discriminator
)
from src.training import train_srgan_baseline, train_attentive_esrgan
from src.utils import (
    predict_srcnn_full_image,
    predict_srgan_full_image,
    predict_attentive_full_image,
    plot_gan_history,
    compare_two_gan_models
)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("TensorFlow version:", tf.__version__)


def download_dataset():
    """Download DIV2K dataset using kagglehub (if available)."""
    if not KAGGLEHUB_AVAILABLE:
        print("Error: kagglehub is not installed.")
        print("Please install it with: pip install kagglehub")
        print("Or download DIV2K manually from: https://data.vision.ee.ethz.ch/cvl/DIV2K/")
        print("Then set DIV2K_ROOT in config.py to point to your dataset location.")
        sys.exit(1)
    
    print("Downloading DIV2K dataset using kagglehub...")
    try:
        div2k_path = kagglehub.dataset_download('soumikrakshit/div2k-high-resolution-images')
        print(f"Dataset downloaded to: {div2k_path}")
        return div2k_path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please download DIV2K manually from: https://data.vision.ee.ethz.ch/cvl/DIV2K/")
        print("Then set DIV2K_ROOT in config.py to point to your dataset location.")
        sys.exit(1)


def main():
    """Main training pipeline."""
    
    # Check for dataset
    if not os.path.exists(config.DIV2K_ROOT):
        print(f"DIV2K_ROOT not found at {config.DIV2K_ROOT}")
        print("\nOptions:")
        print("1. Download manually from: https://data.vision.ee.ethz.ch/cvl/DIV2K/")
        print("   Then set DIV2K_ROOT in config.py to point to the dataset")
        print("2. Use kagglehub to download automatically (requires: pip install kagglehub)")
        
        if KAGGLEHUB_AVAILABLE:
            response = input("\nAttempt automatic download with kagglehub? (y/n): ")
            if response.lower() == 'y':
                div2k_path = download_dataset()
                config.DIV2K_ROOT = div2k_path
            else:
                print("Please download the dataset manually and update config.py")
                sys.exit(1)
        else:
            print("\nPlease download the dataset manually and update config.py")
            sys.exit(1)
    
    # Prepare data generators
    print("\n=== Preparing Data Generators ===")
    train_hr_dir, valid_hr_dir = resolve_div2k_paths(config.DIV2K_ROOT)
    train_gen = SRGANDataGenerator(
        train_hr_dir,
        batch_size=config.BATCH_SIZE,
        crop_size=config.HR_CROP_SIZE,
        scale_factor=config.UPSCALE
    )
    val_gen = SRGANDataGenerator(
        valid_hr_dir,
        batch_size=config.BATCH_SIZE,
        crop_size=config.HR_CROP_SIZE,
        scale_factor=config.UPSCALE
    )
    
    print(f"Training batches: {len(train_gen)}")
    print(f"Validation batches: {len(val_gen)}")
    
    # ============================================================
    # Train SRGAN Baseline
    # ============================================================
    print("\n=== Training SRGAN Baseline ===")
    srgan_gen = build_srgan_generator(
        scale=config.UPSCALE,
        num_res_blocks=config.SRGAN_NUM_RES_BLOCKS
    )
    
    # Load warm-up weights if available (optional)
    if config.SRRESNET_WARMUP_PATH and os.path.exists(config.SRRESNET_WARMUP_PATH):
        try:
            srgan_gen.load_weights(config.SRRESNET_WARMUP_PATH)
            print(f"Loaded SRResNet warm-up weights from: {config.SRRESNET_WARMUP_PATH}")
        except Exception as e:
            print(f"Could not load warm-up weights: {e}")
    else:
        print("Note: Starting SRGAN training from scratch (no warm-up weights)")
    
    srgan_disc = build_srgan_discriminator(
        input_shape=(config.HR_CROP_SIZE, config.HR_CROP_SIZE, 3)
    )
    vgg = build_vgg(hr_shape=(config.HR_CROP_SIZE, config.HR_CROP_SIZE, 3))
    
    srgan_disc.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=config.SRGAN_DISC_LEARNING_RATE),
        metrics=['accuracy'],
    )
    
    srgan_combined = build_srgan_combined(
        srgan_gen, srgan_disc, vgg,
        lr_shape=(config.LR_CROP_SIZE, config.LR_CROP_SIZE, 3)
    )
    
    srgan_history = train_srgan_baseline(
        generator=srgan_gen,
        discriminator=srgan_disc,
        srgan=srgan_combined,
        vgg=vgg,
        train_loader=train_gen,
        val_loader=val_gen,
        epochs=config.SRGAN_EPOCHS,
        steps_per_epoch=config.SRGAN_STEPS_PER_EPOCH,
    )
    
    # Plot training history
    plot_gan_history(srgan_history, title_prefix="SRGAN Baseline")
    
    # ============================================================
    # Train Attentive ESRGAN
    # ============================================================
    print("\n=== Training Attentive ESRGAN ===")
    att_gen = build_attentive_generator(
        scale=config.UPSCALE,
        num_res_blocks=config.ESRGAN_NUM_RES_BLOCKS
    )
    
    # Load warm-up weights if available (optional)
    if config.ATTENTIVE_WARMUP_PATH and os.path.exists(config.ATTENTIVE_WARMUP_PATH):
        try:
            att_gen.load_weights(config.ATTENTIVE_WARMUP_PATH)
            print(f"Loaded Attentive warm-up weights from: {config.ATTENTIVE_WARMUP_PATH}")
        except Exception as e:
            print(f"Could not load warm-up weights: {e}")
    else:
        print("Note: Starting Attentive ESRGAN training from scratch (no warm-up weights)")
    
    att_disc = build_relativistic_discriminator(
        input_shape=(config.HR_CROP_SIZE, config.HR_CROP_SIZE, 3)
    )
    vgg_esr = build_vgg(hr_shape=(config.HR_CROP_SIZE, config.HR_CROP_SIZE, 3))
    
    att_history = train_attentive_esrgan(
        generator=att_gen,
        discriminator=att_disc,
        vgg=vgg_esr,
        train_loader=train_gen,
        val_loader=val_gen,
        epochs=config.ESRGAN_EPOCHS,
        steps_per_epoch=config.ESRGAN_STEPS_PER_EPOCH,
    )
    
    # Plot training history
    plot_gan_history(att_history, title_prefix="Attentive ESRGAN")
    
    # ============================================================
    # Compare Models (if test images available)
    # ============================================================
    if os.path.exists(config.TEST_IMAGES_DIR):
        print("\n=== Comparing Models ===")
        test_files = sorted([
            f for f in os.listdir(config.TEST_IMAGES_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        
        if test_files:
            img_path = os.path.join(config.TEST_IMAGES_DIR, test_files[0])
            compare_two_gan_models(
                gen_a=srgan_gen,
                gen_b=att_gen,
                label_a="SRGAN Baseline",
                label_b="Attentive ESRGAN",
                image_path=img_path,
                scale_factor=config.UPSCALE,
            )
    
    print("\n=== Training Complete ===")
    print(f"Models saved to: {config.MODEL_SAVE_DIR}")


if __name__ == "__main__":
    main()

