"""
Evaluation functions for Super-Resolution models.
Computes PSNR, SSIM, and other metrics.
"""

import cv2
import numpy as np
import tensorflow as tf
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_psnr(img1, img2, max_val=1.0):
    """Compute PSNR between two images."""
    img1_tf = tf.convert_to_tensor(img1, tf.float32)
    img2_tf = tf.convert_to_tensor(img2, tf.float32)
    return tf.image.psnr(img1_tf, img2_tf, max_val=max_val).numpy()


def compute_ssim(img1, img2, max_val=1.0):
    """Compute SSIM between two images."""
    img1_tf = tf.convert_to_tensor(img1, tf.float32)
    img2_tf = tf.convert_to_tensor(img2, tf.float32)
    return tf.image.ssim(img1_tf, img2_tf, max_val=max_val).numpy()


def evaluate_srcnn(model, image_path, scale_factor=4):
    """
    Evaluate SRCNN on a full-resolution image.
    Returns: (psnr_sr, psnr_bicubic, sr_image, bicubic_image, hr_image)
    """
    hr_img = cv2.imread(image_path)
    if hr_img is None:
        raise ValueError(f"[SRCNN] Could not load image: {image_path}")
    
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

    h, w, _ = hr_img.shape
    h, w = (h // scale_factor) * scale_factor, (w // scale_factor) * scale_factor
    hr_img = hr_img[:h, :w, :]

    lr_shape = (w // scale_factor, h // scale_factor)
    lr_img_small = cv2.resize(hr_img, lr_shape, interpolation=cv2.INTER_CUBIC)
    lr_up = cv2.resize(lr_img_small, (w, h), interpolation=cv2.INTER_CUBIC)

    # SRCNN input: [0,1]
    inp = lr_up.astype(np.float32) / 255.0
    inp_batch = np.expand_dims(inp, axis=0)

    sr_img = model.predict(inp_batch, verbose=0)[0]
    sr_img = np.clip(sr_img, 0.0, 1.0)

    hr_img_01 = hr_img.astype(np.float32) / 255.0
    bicubic_img = lr_up.astype(np.float32) / 255.0

    psnr_sr = compute_psnr(hr_img_01, sr_img, max_val=1.0)
    psnr_bic = compute_psnr(hr_img_01, bicubic_img, max_val=1.0)

    return {
        'psnr_sr': psnr_sr,
        'psnr_bicubic': psnr_bic,
        'sr_image': sr_img,
        'bicubic_image': bicubic_img,
        'hr_image': hr_img_01
    }


def evaluate_gan_model(generator, image_path, scale_factor=4):
    """
    Evaluate GAN-based model (SRGAN/ESRGAN) on a full image.
    Generator expects LR in [-1,1] and outputs SR in [-1,1].
    Returns: (psnr_sr, ssim_sr, psnr_bic, ssim_bic, sr_image, bicubic_image, hr_image)
    """
    hr_img = cv2.imread(image_path)
    if hr_img is None:
        raise ValueError(f"[GAN Model] Could not load image: {image_path}")
    
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

    h, w, _ = hr_img.shape
    h, w = (h // scale_factor) * scale_factor, (w // scale_factor) * scale_factor
    hr_img = hr_img[:h, :w, :]

    lr_shape = (w // scale_factor, h // scale_factor)
    lr_img_small = cv2.resize(hr_img, lr_shape, interpolation=cv2.INTER_CUBIC)

    # Normalize to [-1,1] for generator
    lr_input_01 = lr_img_small.astype(np.float32) / 255.0
    lr_input = lr_input_01 * 2.0 - 1.0
    inp_batch = np.expand_dims(lr_input, axis=0)

    sr = generator.predict(inp_batch, verbose=0)[0]  # [-1,1]
    sr_01 = (sr + 1.0) / 2.0  # [0,1]
    sr_01 = np.clip(sr_01, 0.0, 1.0)

    bicubic = cv2.resize(lr_img_small, (w, h), interpolation=cv2.INTER_CUBIC)
    bicubic = bicubic.astype(np.float32) / 255.0
    hr_01 = hr_img.astype(np.float32) / 255.0

    psnr_sr = compute_psnr(hr_01, sr_01, max_val=1.0)
    ssim_sr = compute_ssim(hr_01, sr_01, max_val=1.0)
    psnr_bic = compute_psnr(hr_01, bicubic, max_val=1.0)
    ssim_bic = compute_ssim(hr_01, bicubic, max_val=1.0)

    return {
        'psnr_sr': psnr_sr,
        'ssim_sr': ssim_sr,
        'psnr_bicubic': psnr_bic,
        'ssim_bicubic': ssim_bic,
        'sr_image': sr_01,
        'bicubic_image': bicubic,
        'hr_image': hr_01
    }


def evaluate_batch(generator, lr_images, hr_images, is_gan=True):
    """
    Evaluate model on a batch of images.
    
    Args:
        generator: Model generator
        lr_images: Low-resolution images (numpy array)
        hr_images: High-resolution images (numpy array)
        is_gan: Whether the model is GAN-based (uses [-1,1] range)
    
    Returns:
        Dictionary with average PSNR and SSIM
    """
    psnrs = []
    ssims = []
    
    for lr_img, hr_img in zip(lr_images, hr_images):
        if is_gan:
            # GAN models expect [-1,1]
            lr_input = (lr_img + 1.0) / 2.0 if lr_img.min() < 0 else lr_img
            lr_input = lr_input * 2.0 - 1.0
            inp_batch = np.expand_dims(lr_input, axis=0)
            
            sr = generator.predict(inp_batch, verbose=0)[0]
            sr_01 = np.clip((sr + 1.0) / 2.0, 0.0, 1.0)
            hr_01 = np.clip((hr_img + 1.0) / 2.0, 0.0, 1.0)
        else:
            # SRCNN expects [0,1]
            inp_batch = np.expand_dims(lr_img, axis=0)
            sr_01 = np.clip(generator.predict(inp_batch, verbose=0)[0], 0.0, 1.0)
            hr_01 = np.clip(hr_img, 0.0, 1.0)
        
        psnr = compute_psnr(hr_01, sr_01, max_val=1.0)
        ssim = compute_ssim(hr_01, sr_01, max_val=1.0)
        
        psnrs.append(psnr)
        ssims.append(ssim)
    
    return {
        'psnr_mean': float(np.mean(psnrs)),
        'psnr_std': float(np.std(psnrs)),
        'ssim_mean': float(np.mean(ssims)),
        'ssim_std': float(np.std(ssims))
    }

