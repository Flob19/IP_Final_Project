#!/usr/bin/env python3
"""
Quick test script to verify SRGAN and Attentive ESRGAN training works.
Runs only 3 epochs for each model to quickly verify functionality.
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data import SRGANDataGenerator, resolve_div2k_paths
from src.models.srgan import SRGANGenerator, SRGANDiscriminator, build_vgg, build_srgan_combined
from src.models.attentive_esrgan import AttentiveESRGANGenerator, RelativisticDiscriminator, build_vgg as build_vgg_esr
from src.training import train_srgan_baseline, train_attentive_esrgan

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("TensorFlow version:", tf.__version__)
print("\n" + "="*60)
print("QUICK TEST: SRGAN and Attentive ESRGAN (3 epochs each)")
print("="*60 + "\n")


def check_dataset():
    """Check if dataset exists."""
    if not os.path.exists(config.DIV2K_ROOT):
        print(f"ERROR: DIV2K_ROOT not found at {config.DIV2K_ROOT}")
        print("\nPlease:")
        print("1. Download DIV2K from: https://data.vision.ee.ethz.ch/cvl/DIV2K/")
        print("2. Extract to ./data/DIV2K/")
        print("3. Or update DIV2K_ROOT in config.py")
        return False
    return True


def test_srgan():
    """Test SRGAN training with 3 epochs."""
    print("\n" + "="*60)
    print("TESTING SRGAN BASELINE (3 epochs)")
    print("="*60)
    
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
    
    if len(train_gen) == 0:
        print("ERROR: No training data found!")
        return False
    
    # Build models
    print("\nBuilding SRGAN models...")
    srgan_gen = SRGANGenerator(
        scale=config.UPSCALE,
        num_res_blocks=config.SRGAN_NUM_RES_BLOCKS
    )
    print("✓ Generator built")
    
    srgan_disc = SRGANDiscriminator(
        input_shape=(config.HR_CROP_SIZE, config.HR_CROP_SIZE, 3)
    )
    srgan_disc.compile_model()
    print("✓ Discriminator built and compiled")
    
    vgg = build_vgg(hr_shape=(config.HR_CROP_SIZE, config.HR_CROP_SIZE, 3))
    print("✓ VGG feature extractor built")
    
    srgan_combined = build_srgan_combined(
        srgan_gen, srgan_disc, vgg,
        lr_shape=(config.LR_CROP_SIZE, config.LR_CROP_SIZE, 3)
    )
    print("✓ Combined model built")
    
    # Training with 3 epochs
    print("\nStarting training (3 epochs)...")
    try:
        srgan_history = train_srgan_baseline(
            generator=srgan_gen.model,
            discriminator=srgan_disc.model,
            srgan=srgan_combined,
            vgg=vgg,
            train_loader=train_gen,
            val_loader=val_gen,
            epochs=3,  # Only 3 epochs for testing
            steps_per_epoch=min(10, len(train_gen)),  # Limit steps for quick test
        )
        print("\n✓ SRGAN training completed successfully!")
        print(f"  Final D loss: {srgan_history['d_loss'][-1]:.4f}")
        print(f"  Final G loss: {srgan_history['g_loss'][-1]:.4f}")
        if 'val_psnr' in srgan_history:
            print(f"  Final PSNR: {srgan_history['val_psnr'][-1]:.2f} dB")
        return True
    except Exception as e:
        print(f"\n✗ SRGAN training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_attentive_esrgan():
    """Test Attentive ESRGAN training with 3 epochs."""
    print("\n" + "="*60)
    print("TESTING ATTENTIVE ESRGAN (3 epochs)")
    print("="*60)
    
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
    
    if len(train_gen) == 0:
        print("ERROR: No training data found!")
        return False
    
    # Build models
    print("\nBuilding Attentive ESRGAN models...")
    att_gen = AttentiveESRGANGenerator(
        scale=config.UPSCALE,
        num_res_blocks=config.ESRGAN_NUM_RES_BLOCKS
    )
    print("✓ Generator built")
    
    att_disc = RelativisticDiscriminator(
        input_shape=(config.HR_CROP_SIZE, config.HR_CROP_SIZE, 3)
    )
    print("✓ Discriminator built")
    
    vgg_esr = build_vgg_esr(hr_shape=(config.HR_CROP_SIZE, config.HR_CROP_SIZE, 3))
    print("✓ VGG feature extractor built")
    
    # Training with 3 epochs
    print("\nStarting training (3 epochs)...")
    try:
        att_history = train_attentive_esrgan(
            generator=att_gen.model,
            discriminator=att_disc.model,
            vgg=vgg_esr,
            train_loader=train_gen,
            val_loader=val_gen,
            epochs=3,  # Only 3 epochs for testing
            steps_per_epoch=min(10, len(train_gen)),  # Limit steps for quick test
        )
        print("\n✓ Attentive ESRGAN training completed successfully!")
        print(f"  Final D loss: {att_history['d_loss'][-1]:.4f}")
        print(f"  Final G loss: {att_history['g_loss'][-1]:.4f}")
        if 'val_psnr' in att_history:
            print(f"  Final PSNR: {att_history['val_psnr'][-1]:.2f} dB")
        return True
    except Exception as e:
        print(f"\n✗ Attentive ESRGAN training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests."""
    print("\nChecking dataset...")
    if not check_dataset():
        sys.exit(1)
    
    print("✓ Dataset found\n")
    
    # Create model save directory
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    # Run tests
    results = []
    
    print("\n" + "="*60)
    print("TEST 1: SRGAN Baseline")
    print("="*60)
    results.append(("SRGAN", test_srgan()))
    
    print("\n" + "="*60)
    print("TEST 2: Attentive ESRGAN")
    print("="*60)
    results.append(("Attentive ESRGAN", test_attentive_esrgan()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests passed! Code is working correctly.")
        print("You can now run full training with:")
        print("  python train.py srgan --epochs 30")
        print("  python train.py attentive-esrgan --epochs 30")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

