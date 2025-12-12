"""
Visualization functions for Super-Resolution models.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def plot_srcnn_results(eval_results, save_path=None):
    """
    Plot SRCNN evaluation results.
    
    Args:
        eval_results: Dictionary from evaluate_srcnn()
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    
    axes[0].imshow(eval_results['bicubic_image'])
    axes[0].set_title(f"Bicubic\nPSNR: {eval_results['psnr_bicubic']:.2f} dB")
    axes[0].axis("off")

    axes[1].imshow(eval_results['sr_image'])
    color = "green" if eval_results['psnr_sr'] > eval_results['psnr_bicubic'] else "black"
    axes[1].set_title(
        f"SRCNN\nPSNR: {eval_results['psnr_sr']:.2f} dB",
        color=color,
        fontweight="bold",
    )
    axes[1].axis("off")

    axes[2].imshow(eval_results['hr_image'])
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_gan_results(eval_results, model_name="GAN Model", save_path=None):
    """
    Plot GAN-based model evaluation results.
    
    Args:
        eval_results: Dictionary from evaluate_gan_model()
        model_name: Name of the model for title
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    
    axes[0].imshow(eval_results['bicubic_image'])
    axes[0].set_title(
        f"Bicubic\nPSNR: {eval_results['psnr_bicubic']:.2f} dB | "
        f"SSIM: {eval_results['ssim_bicubic']:.4f}"
    )
    axes[0].axis("off")

    axes[1].imshow(eval_results['sr_image'])
    color = "green" if eval_results['psnr_sr'] > eval_results['psnr_bicubic'] else "black"
    axes[1].set_title(
        f"{model_name}\nPSNR: {eval_results['psnr_sr']:.2f} dB | "
        f"SSIM: {eval_results['ssim_sr']:.4f}",
        color=color,
        fontweight="bold"
    )
    axes[1].axis("off")

    axes[2].imshow(eval_results['hr_image'])
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_comparison(gen_a, gen_b, eval_results_a, eval_results_b,
                   label_a="Model A", label_b="Model B", save_path=None):
    """
    Compare two GAN-based models side by side.
    
    Args:
        gen_a: First generator model
        gen_b: Second generator model
        eval_results_a: Evaluation results for model A
        eval_results_b: Evaluation results for model B
        label_a: Label for model A
        label_b: Label for model B
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(28, 8))

    axes[0].imshow(eval_results_a['bicubic_image'])
    axes[0].set_title(
        f"Bicubic\nPSNR: {eval_results_a['psnr_bicubic']:.2f} | "
        f"SSIM: {eval_results_a['ssim_bicubic']:.4f}"
    )
    axes[0].axis("off")

    axes[1].imshow(eval_results_a['sr_image'])
    axes[1].set_title(
        f"{label_a}\nPSNR: {eval_results_a['psnr_sr']:.2f} | "
        f"SSIM: {eval_results_a['ssim_sr']:.4f}"
    )
    axes[1].axis("off")

    axes[2].imshow(eval_results_b['sr_image'])
    axes[2].set_title(
        f"{label_b}\nPSNR: {eval_results_b['psnr_sr']:.2f} | "
        f"SSIM: {eval_results_b['ssim_sr']:.4f}"
    )
    axes[2].axis("off")

    axes[3].imshow(eval_results_a['hr_image'])
    axes[3].set_title("Ground Truth")
    axes[3].axis("off")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history, title_prefix="Model", save_path=None):
    """
    Plot training history (loss curves and validation metrics).
    
    Args:
        history: Dictionary with training history
        title_prefix: Prefix for plot titles
        save_path: Optional path to save the figure
    """
    epochs = history.get("epoch", list(range(1, len(history.get("d_loss", [])) + 1)))

    # Loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["d_loss"], label="D loss", marker='o')
    plt.plot(epochs, history["g_loss"], label="G loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(f"{save_path}_loss.png", dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

    # Validation metrics (if available)
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
        
        if save_path:
            plt.savefig(f"{save_path}_metrics.png", dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

