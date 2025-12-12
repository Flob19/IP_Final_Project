"""
Utility functions for evaluation and visualization.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def predict_srcnn_full_image(model, image_path, scale_factor=4):
    """
    Evaluate SRCNN on a full-resolution image (baseline PSNR model).
    Steps:
        HR -> downscale (LR) -> bicubic upsample -> SRCNN -> compare to HR
    Everything is in [0,1] for SRCNN.
    """
    hr_img = cv2.imread(image_path)
    if hr_img is None:
        print(f"[SRCNN] Could not load image: {image_path}")
        return
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

    tf_hr = tf.convert_to_tensor(hr_img_01, tf.float32)
    tf_sr = tf.convert_to_tensor(sr_img, tf.float32)
    tf_bic = tf.convert_to_tensor(bicubic_img, tf.float32)

    psnr_sr = tf.image.psnr(tf_hr, tf_sr, max_val=1.0).numpy()
    psnr_bic = tf.image.psnr(tf_hr, tf_bic, max_val=1.0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    axes[0].imshow(bicubic_img)
    axes[0].set_title(f"Bicubic\nPSNR: {psnr_bic:.2f} dB")
    axes[0].axis("off")

    axes[1].imshow(sr_img)
    axes[1].set_title(
        f"SRCNN\nPSNR: {psnr_sr:.2f} dB",
        color="green" if psnr_sr > psnr_bic else "black",
        fontweight="bold",
    )
    axes[1].axis("off")

    axes[2].imshow(hr_img_01)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def predict_srgan_full_image(generator, image_path, scale_factor=4):
    """
    Evaluate SRGAN baseline on a full image.
    Generator expects LR in [-1,1] and outputs SR in [-1,1].
    """
    hr_img = cv2.imread(image_path)
    if hr_img is None:
        print(f"[SRGAN] Could not load image: {image_path}")
        return
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

    sr = generator.predict(inp_batch, verbose=0)[0]        # [-1,1]
    sr_01 = (sr + 1.0) / 2.0                               # [0,1]
    sr_01 = np.clip(sr_01, 0.0, 1.0)

    bicubic = cv2.resize(lr_img_small, (w, h), interpolation=cv2.INTER_CUBIC)
    bicubic = bicubic.astype(np.float32) / 255.0

    hr_01 = hr_img.astype(np.float32) / 255.0

    tf_hr = tf.convert_to_tensor(hr_01, tf.float32)
    tf_sr = tf.convert_to_tensor(sr_01, tf.float32)
    tf_bic = tf.convert_to_tensor(bicubic, tf.float32)

    psnr_sr = tf.image.psnr(tf_hr, tf_sr, max_val=1.0).numpy()
    ssim_sr = tf.image.ssim(tf_hr, tf_sr, max_val=1.0).numpy()
    psnr_bic = tf.image.psnr(tf_hr, tf_bic, max_val=1.0).numpy()
    ssim_bic = tf.image.ssim(tf_hr, tf_bic, max_val=1.0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    axes[0].imshow(bicubic)
    axes[0].set_title(f"Bicubic\nPSNR: {psnr_bic:.2f} dB | SSIM: {ssim_bic:.4f}")
    axes[0].axis("off")

    axes[1].imshow(sr_01)
    title_col = "green" if psnr_sr > psnr_bic else "black"
    axes[1].set_title(f"SRGAN\nPSNR: {psnr_sr:.2f} dB | SSIM: {ssim_sr:.4f}",
                      color=title_col, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(hr_01)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def predict_attentive_full_image(generator, image_path, scale_factor=4):
    """
    Evaluate Attentive ESRGAN on a full image.
    Same normalization as SRGAN baseline: [-1,1] in, tanh output.
    """
    hr_img = cv2.imread(image_path)
    if hr_img is None:
        print(f"[Attentive ESRGAN] Could not load image: {image_path}")
        return
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

    h, w, _ = hr_img.shape
    h, w = (h // scale_factor) * scale_factor, (w // scale_factor) * scale_factor
    hr_img = hr_img[:h, :w, :]

    lr_shape = (w // scale_factor, h // scale_factor)
    lr_img_small = cv2.resize(hr_img, lr_shape, interpolation=cv2.INTER_CUBIC)

    lr_01 = lr_img_small.astype(np.float32) / 255.0
    lr_in = lr_01 * 2.0 - 1.0
    inp_batch = np.expand_dims(lr_in, axis=0)

    sr = generator.predict(inp_batch, verbose=0)[0]   # [-1,1]
    sr_01 = (sr + 1.0) / 2.0
    sr_01 = np.clip(sr_01, 0.0, 1.0)

    bicubic = cv2.resize(lr_img_small, (w, h), interpolation=cv2.INTER_CUBIC)
    bicubic = bicubic.astype(np.float32) / 255.0
    hr_01 = hr_img.astype(np.float32) / 255.0

    tf_hr = tf.convert_to_tensor(hr_01, tf.float32)
    tf_sr = tf.convert_to_tensor(sr_01, tf.float32)
    tf_bic = tf.convert_to_tensor(bicubic, tf.float32)

    psnr_sr = tf.image.psnr(tf_hr, tf_sr, max_val=1.0).numpy()
    ssim_sr = tf.image.ssim(tf_hr, tf_sr, max_val=1.0).numpy()
    psnr_bic = tf.image.psnr(tf_hr, tf_bic, max_val=1.0).numpy()
    ssim_bic = tf.image.ssim(tf_hr, tf_bic, max_val=1.0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    axes[0].imshow(bicubic)
    axes[0].set_title(f"Bicubic\nPSNR: {psnr_bic:.2f} dB | SSIM: {ssim_bic:.4f}")
    axes[0].axis("off")

    axes[1].imshow(sr_01)
    title_col = "green" if psnr_sr > psnr_bic else "black"
    axes[1].set_title(
        f"Attentive ESRGAN\nPSNR: {psnr_sr:.2f} dB | SSIM: {ssim_sr:.4f}",
        color=title_col, fontweight="bold",
    )
    axes[1].axis("off")

    axes[2].imshow(hr_01)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def plot_gan_history(history, title_prefix="Model"):
    """
    Plot D/G loss and optional validation PSNR/SSIM vs epoch.
    history: dict returned by train_srgan_baseline / train_attentive_esrgan.
    """
    epochs = history.get("epoch", list(range(1, len(history.get("d_loss", [])) + 1)))

    # ---- 1) Loss curves ----
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["d_loss"], label="D loss")
    plt.plot(epochs, history["g_loss"], label="G loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---- 2) Validation metrics (if available) ----
    if "val_psnr" in history and "val_ssim" in history:
        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("PSNR (dB)", color="tab:blue")
        ax1.plot(epochs, history["val_psnr"], marker="o", label="PSNR", color="tab:blue")
        ax1.tick_params(axis='y', labelcolor="tab:blue")
        ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel("SSIM", color="tab:orange")
        ax2.plot(epochs, history["val_ssim"], marker="s", label="SSIM", color="tab:orange")
        ax2.tick_params(axis='y', labelcolor="tab:orange")

        plt.title(f"{title_prefix} Validation Metrics")
        fig.tight_layout()
        plt.show()


def compare_two_gan_models(gen_a,
                           gen_b,
                           label_a="Model A",
                           label_b="Model B",
                           image_path=None,
                           scale_factor=4):
    """
    Compare two GAN-based SR models (tanh output in [-1,1]) on the same image.
    Shows: Bicubic, Model A, Model B, Ground Truth
    and prints PSNR/SSIM for each model.
    """
    if image_path is None:
        print("Please provide image_path.")
        return

    # 1. Load HR image
    hr_img = cv2.imread(image_path)
    if hr_img is None:
        print(f"[Compare] Could not load image: {image_path}")
        return
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

    h, w, _ = hr_img.shape
    h, w = (h // scale_factor) * scale_factor, (w // scale_factor) * scale_factor
    hr_img = hr_img[:h, :w, :]

    # 2. Create LR
    lr_shape = (w // scale_factor, h // scale_factor)
    lr_img_small = cv2.resize(hr_img, lr_shape, interpolation=cv2.INTER_CUBIC)

    # 3. Prepare input [-1,1] for both models
    lr_01 = lr_img_small.astype(np.float32) / 255.0
    lr_in = lr_01 * 2.0 - 1.0
    inp_batch = np.expand_dims(lr_in, axis=0)

    # 4. Run both models
    sr_a = gen_a.predict(inp_batch, verbose=0)[0]    # [-1,1]
    sr_b = gen_b.predict(inp_batch, verbose=0)[0]    # [-1,1]

    sr_a_01 = np.clip((sr_a + 1.0) / 2.0, 0.0, 1.0)
    sr_b_01 = np.clip((sr_b + 1.0) / 2.0, 0.0, 1.0)

    bicubic = cv2.resize(lr_img_small, (w, h), interpolation=cv2.INTER_CUBIC)
    bicubic = bicubic.astype(np.float32) / 255.0
    hr_01 = hr_img.astype(np.float32) / 255.0

    # 5. Compute metrics
    def _metrics(sr_img_01):
        tf_hr = tf.convert_to_tensor(hr_01, tf.float32)
        tf_sr = tf.convert_to_tensor(sr_img_01, tf.float32)
        psnr = tf.image.psnr(tf_hr, tf_sr, max_val=1.0).numpy()
        ssim = tf.image.ssim(tf_hr, tf_sr, max_val=1.0).numpy()
        return psnr, ssim

    psnr_bic, ssim_bic = _metrics(bicubic)
    psnr_a, ssim_a = _metrics(sr_a_01)
    psnr_b, ssim_b = _metrics(sr_b_01)

    print(f"[Bicubic]   PSNR: {psnr_bic:.2f} dB | SSIM: {ssim_bic:.4f}")
    print(f"[{label_a}] PSNR: {psnr_a:.2f} dB | SSIM: {ssim_a:.4f}")
    print(f"[{label_b}] PSNR: {psnr_b:.2f} dB | SSIM: {ssim_b:.4f}")

    # 6. Plot
    fig, axes = plt.subplots(1, 4, figsize=(28, 8))

    axes[0].imshow(bicubic)
    axes[0].set_title(f"Bicubic\nPSNR: {psnr_bic:.2f} | SSIM: {ssim_bic:.4f}")
    axes[0].axis("off")

    axes[1].imshow(sr_a_01)
    axes[1].set_title(f"{label_a}\nPSNR: {psnr_a:.2f} | SSIM: {ssim_a:.4f}")
    axes[1].axis("off")

    axes[2].imshow(sr_b_01)
    axes[2].set_title(f"{label_b}\nPSNR: {psnr_b:.2f} | SSIM: {ssim_b:.4f}")
    axes[2].axis("off")

    axes[3].imshow(hr_01)
    axes[3].set_title("Ground Truth")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

