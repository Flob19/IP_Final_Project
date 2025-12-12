#!/usr/bin/env python3
"""
Main training script with command-line interface.
Supports training SRCNN, SRGAN baseline, and Attentive ESRGAN models.
"""

import argparse
import os
import sys
import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data import SRGANDataGenerator, resolve_div2k_paths
from src.models.srcnn import SRCNN
from src.models.srgan import SRGANGenerator, SRGANDiscriminator, build_vgg, build_srgan_combined
from src.models.attentive_esrgan import AttentiveESRGANGenerator, RelativisticDiscriminator, build_vgg as build_vgg_esr
from src.training import train_srgan_baseline, train_attentive_esrgan
from src.visualization import plot_training_history

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("TensorFlow version:", tf.__version__)


def download_dataset():
    """Download DIV2K dataset using kagglehub (if available)."""
    try:
        import kagglehub
        print("Downloading DIV2K dataset using kagglehub...")
        div2k_path = kagglehub.dataset_download('soumikrakshit/div2k-high-resolution-images')
        print(f"Dataset downloaded to: {div2k_path}")
        return div2k_path
    except ImportError:
        print("Error: kagglehub is not installed.")
        print("Please install it with: pip install kagglehub")
        print("Or download DIV2K manually from: https://data.vision.ee.ethz.ch/cvl/DIV2K/")
        sys.exit(1)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please download DIV2K manually from: https://data.vision.ee.ethz.ch/cvl/DIV2K/")
        sys.exit(1)


def check_dataset():
    """Check if dataset exists, prompt for download if not."""
    if not os.path.exists(config.DIV2K_ROOT):
        print(f"DIV2K_ROOT not found at {config.DIV2K_ROOT}")
        print("\nOptions:")
        print("1. Download manually from: https://data.vision.ee.ethz.ch/cvl/DIV2K/")
        print("   Then set DIV2K_ROOT in config.py to point to the dataset")
        print("2. Use kagglehub to download automatically (requires: pip install kagglehub)")
        
        try:
            import kagglehub
            response = input("\nAttempt automatic download with kagglehub? (y/n): ")
            if response.lower() == 'y':
                div2k_path = download_dataset()
                config.DIV2K_ROOT = div2k_path
            else:
                print("Please download the dataset manually and update config.py")
                sys.exit(1)
        except ImportError:
            print("\nPlease download the dataset manually and update config.py")
            sys.exit(1)


def train_srcnn(args):
    """Train SRCNN model."""
    print("\n=== Training SRCNN Baseline ===")
    
    # Prepare data generators
    train_hr_dir, valid_hr_dir = resolve_div2k_paths(config.DIV2K_ROOT)
    # Note: SRCNN uses different preprocessing, would need separate generator
    # For now, this is a placeholder
    print("SRCNN training not fully implemented in this version.")
    print("Please use the original notebook code for SRCNN training.")


def train_srgan(args):
    """Train SRGAN baseline model."""
    print("\n=== Training SRGAN Baseline ===")
    
    # Prepare data generators
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
    
    # Build models
    srgan_gen = SRGANGenerator(
        scale=config.UPSCALE,
        num_res_blocks=config.SRGAN_NUM_RES_BLOCKS
    )
    
    # Load warm-up weights if available
    if config.SRRESNET_WARMUP_PATH and os.path.exists(config.SRRESNET_WARMUP_PATH):
        try:
            srgan_gen.load_weights(config.SRRESNET_WARMUP_PATH)
            print(f"Loaded SRResNet warm-up weights from: {config.SRRESNET_WARMUP_PATH}")
        except Exception as e:
            print(f"Could not load warm-up weights: {e}")
    else:
        print("Note: Starting SRGAN training from scratch (no warm-up weights)")
    
    srgan_disc = SRGANDiscriminator(
        input_shape=(config.HR_CROP_SIZE, config.HR_CROP_SIZE, 3)
    )
    srgan_disc.compile_model()
    
    vgg = build_vgg(hr_shape=(config.HR_CROP_SIZE, config.HR_CROP_SIZE, 3))
    
    srgan_combined = build_srgan_combined(
        srgan_gen, srgan_disc, vgg,
        lr_shape=(config.LR_CROP_SIZE, config.LR_CROP_SIZE, 3)
    )
    
    # Training
    epochs = args.epochs if args.epochs else config.SRGAN_EPOCHS
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch else config.SRGAN_STEPS_PER_EPOCH
    
    srgan_history = train_srgan_baseline(
        generator=srgan_gen.model,
        discriminator=srgan_disc.model,
        srgan=srgan_combined,
        vgg=vgg,
        train_loader=train_gen,
        val_loader=val_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )
    
    # Plot training history
    save_path = os.path.join(config.MODEL_SAVE_DIR, "srgan_history") if args.save_plots else None
    plot_training_history(srgan_history, title_prefix="SRGAN Baseline", save_path=save_path)
    
    print(f"\n=== Training Complete ===")
    print(f"Models saved to: {config.MODEL_SAVE_DIR}")


def train_attentive_esrgan(args):
    """Train Attentive ESRGAN model."""
    print("\n=== Training Attentive ESRGAN ===")
    
    # Prepare data generators
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
    
    # Build models
    att_gen = AttentiveESRGANGenerator(
        scale=config.UPSCALE,
        num_res_blocks=config.ESRGAN_NUM_RES_BLOCKS
    )
    
    # Load warm-up weights if available
    if config.ATTENTIVE_WARMUP_PATH and os.path.exists(config.ATTENTIVE_WARMUP_PATH):
        try:
            att_gen.load_weights(config.ATTENTIVE_WARMUP_PATH)
            print(f"Loaded Attentive warm-up weights from: {config.ATTENTIVE_WARMUP_PATH}")
        except Exception as e:
            print(f"Could not load warm-up weights: {e}")
    else:
        print("Note: Starting Attentive ESRGAN training from scratch (no warm-up weights)")
    
    att_disc = RelativisticDiscriminator(
        input_shape=(config.HR_CROP_SIZE, config.HR_CROP_SIZE, 3)
    )
    vgg_esr = build_vgg_esr(hr_shape=(config.HR_CROP_SIZE, config.HR_CROP_SIZE, 3))
    
    # Training
    epochs = args.epochs if args.epochs else config.ESRGAN_EPOCHS
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch else config.ESRGAN_STEPS_PER_EPOCH
    
    att_history = train_attentive_esrgan(
        generator=att_gen.model,
        discriminator=att_disc.model,
        vgg=vgg_esr,
        train_loader=train_gen,
        val_loader=val_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )
    
    # Plot training history
    save_path = os.path.join(config.MODEL_SAVE_DIR, "attentive_esrgan_history") if args.save_plots else None
    plot_training_history(att_history, title_prefix="Attentive ESRGAN", save_path=save_path)
    
    print(f"\n=== Training Complete ===")
    print(f"Models saved to: {config.MODEL_SAVE_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description='Train Super-Resolution GAN models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train SRGAN baseline
  python train.py srgan

  # Train Attentive ESRGAN with custom epochs
  python train.py attentive-esrgan --epochs 50

  # Train with custom steps per epoch
  python train.py srgan --epochs 30 --steps-per-epoch 100 --save-plots
        """
    )
    
    parser.add_argument(
        'model',
        choices=['srcnn', 'srgan', 'attentive-esrgan'],
        help='Model to train'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help=f'Number of epochs (default: from config.py)'
    )
    
    parser.add_argument(
        '--steps-per-epoch',
        type=int,
        help=f'Steps per epoch (default: from config.py)'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save training plots to files'
    )
    
    args = parser.parse_args()
    
    # Check dataset
    check_dataset()
    
    # Create model save directory
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    # Train selected model
    if args.model == 'srcnn':
        train_srcnn(args)
    elif args.model == 'srgan':
        train_srgan(args)
    elif args.model == 'attentive-esrgan':
        train_attentive_esrgan(args)


if __name__ == "__main__":
    main()
