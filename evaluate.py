#!/usr/bin/env python3
"""
Evaluation script with command-line interface.
Evaluate trained models on test images.
"""

import argparse
import os
import sys
import tensorflow as tf

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.models.srcnn import SRCNN
from src.models.srgan import SRGANGenerator
from src.models.attentive_esrgan import AttentiveESRGANGenerator
from src.evaluation import evaluate_srcnn, evaluate_gan_model
from src.visualization import plot_srcnn_results, plot_gan_results, plot_comparison

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def evaluate_srcnn_model(args):
    """Evaluate SRCNN model."""
    print(f"\n=== Evaluating SRCNN ===")
    print(f"Model path: {args.model_path}")
    print(f"Image path: {args.image}")
    
    # Load model
    srcnn = SRCNN()
    if args.model_path:
        srcnn.load_weights(args.model_path)
    
    # Evaluate
    results = evaluate_srcnn(srcnn.get_model(), args.image, scale_factor=config.UPSCALE)
    
    print(f"\nResults:")
    print(f"PSNR (SRCNN): {results['psnr_sr']:.2f} dB")
    print(f"PSNR (Bicubic): {results['psnr_bicubic']:.2f} dB")
    
    # Visualize
    save_path = os.path.join(args.output_dir, "srcnn_result.png") if args.output_dir else None
    plot_srcnn_results(results, save_path=save_path)


def evaluate_srgan_model(args):
    """Evaluate SRGAN model."""
    print(f"\n=== Evaluating SRGAN ===")
    print(f"Model path: {args.model_path}")
    print(f"Image path: {args.image}")
    
    # Load model
    generator = SRGANGenerator(scale=config.UPSCALE, num_res_blocks=config.SRGAN_NUM_RES_BLOCKS)
    if args.model_path:
        generator.load_weights(args.model_path)
    
    # Evaluate
    results = evaluate_gan_model(generator.get_model(), args.image, scale_factor=config.UPSCALE)
    
    print(f"\nResults:")
    print(f"PSNR (SRGAN): {results['psnr_sr']:.2f} dB")
    print(f"SSIM (SRGAN): {results['ssim_sr']:.4f}")
    print(f"PSNR (Bicubic): {results['psnr_bicubic']:.2f} dB")
    print(f"SSIM (Bicubic): {results['ssim_bicubic']:.4f}")
    
    # Visualize
    save_path = os.path.join(args.output_dir, "srgan_result.png") if args.output_dir else None
    plot_gan_results(results, model_name="SRGAN", save_path=save_path)


def evaluate_attentive_esrgan_model(args):
    """Evaluate Attentive ESRGAN model."""
    print(f"\n=== Evaluating Attentive ESRGAN ===")
    print(f"Model path: {args.model_path}")
    print(f"Image path: {args.image}")
    
    # Load model
    generator = AttentiveESRGANGenerator(
        scale=config.UPSCALE,
        num_res_blocks=config.ESRGAN_NUM_RES_BLOCKS
    )
    if args.model_path:
        generator.load_weights(args.model_path)
    
    # Evaluate
    results = evaluate_gan_model(generator.get_model(), args.image, scale_factor=config.UPSCALE)
    
    print(f"\nResults:")
    print(f"PSNR (Attentive ESRGAN): {results['psnr_sr']:.2f} dB")
    print(f"SSIM (Attentive ESRGAN): {results['ssim_sr']:.4f}")
    print(f"PSNR (Bicubic): {results['psnr_bicubic']:.2f} dB")
    print(f"SSIM (Bicubic): {results['ssim_bicubic']:.4f}")
    
    # Visualize
    save_path = os.path.join(args.output_dir, "attentive_esrgan_result.png") if args.output_dir else None
    plot_gan_results(results, model_name="Attentive ESRGAN", save_path=save_path)


def compare_models(args):
    """Compare two GAN models."""
    print(f"\n=== Comparing Models ===")
    print(f"Model A path: {args.model_a}")
    print(f"Model B path: {args.model_b}")
    print(f"Image path: {args.image}")
    
    # Load models
    gen_a = SRGANGenerator(scale=config.UPSCALE, num_res_blocks=config.SRGAN_NUM_RES_BLOCKS)
    gen_b = AttentiveESRGANGenerator(scale=config.UPSCALE, num_res_blocks=config.ESRGAN_NUM_RES_BLOCKS)
    
    if args.model_a:
        gen_a.load_weights(args.model_a)
    if args.model_b:
        gen_b.load_weights(args.model_b)
    
    # Evaluate both
    results_a = evaluate_gan_model(gen_a.get_model(), args.image, scale_factor=config.UPSCALE)
    results_b = evaluate_gan_model(gen_b.get_model(), args.image, scale_factor=config.UPSCALE)
    
    print(f"\nResults:")
    print(f"{args.label_a}: PSNR={results_a['psnr_sr']:.2f} dB, SSIM={results_a['ssim_sr']:.4f}")
    print(f"{args.label_b}: PSNR={results_b['psnr_sr']:.2f} dB, SSIM={results_b['ssim_sr']:.4f}")
    
    # Visualize comparison
    save_path = os.path.join(args.output_dir, "comparison.png") if args.output_dir else None
    plot_comparison(
        gen_a.get_model(), gen_b.get_model(),
        results_a, results_b,
        label_a=args.label_a,
        label_b=args.label_b,
        save_path=save_path
    )


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Super-Resolution GAN models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate SRGAN model
  python evaluate.py srgan --model-path models/srgan_generator_epoch_30.keras --image test.jpg

  # Evaluate Attentive ESRGAN
  python evaluate.py attentive-esrgan --model-path models/attentive_esrgan_epoch_30.keras --image test.jpg --output-dir results/

  # Compare two models
  python evaluate.py compare --model-a models/srgan_generator_epoch_30.keras --model-b models/attentive_esrgan_epoch_30.keras --image test.jpg
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Evaluation command')
    
    # SRCNN evaluation
    parser_srcnn = subparsers.add_parser('srcnn', help='Evaluate SRCNN model')
    parser_srcnn.add_argument('--model-path', type=str, help='Path to SRCNN model weights')
    parser_srcnn.add_argument('--image', type=str, required=True, help='Path to test image')
    parser_srcnn.add_argument('--output-dir', type=str, help='Directory to save results')
    
    # SRGAN evaluation
    parser_srgan = subparsers.add_parser('srgan', help='Evaluate SRGAN model')
    parser_srgan.add_argument('--model-path', type=str, required=True, help='Path to SRGAN generator model')
    parser_srgan.add_argument('--image', type=str, required=True, help='Path to test image')
    parser_srgan.add_argument('--output-dir', type=str, help='Directory to save results')
    
    # Attentive ESRGAN evaluation
    parser_attentive = subparsers.add_parser('attentive-esrgan', help='Evaluate Attentive ESRGAN model')
    parser_attentive.add_argument('--model-path', type=str, required=True, help='Path to Attentive ESRGAN generator model')
    parser_attentive.add_argument('--image', type=str, required=True, help='Path to test image')
    parser_attentive.add_argument('--output-dir', type=str, help='Directory to save results')
    
    # Compare models
    parser_compare = subparsers.add_parser('compare', help='Compare two GAN models')
    parser_compare.add_argument('--model-a', type=str, required=True, help='Path to first model (SRGAN)')
    parser_compare.add_argument('--model-b', type=str, required=True, help='Path to second model (Attentive ESRGAN)')
    parser_compare.add_argument('--image', type=str, required=True, help='Path to test image')
    parser_compare.add_argument('--label-a', type=str, default='SRGAN Baseline', help='Label for model A')
    parser_compare.add_argument('--label-b', type=str, default='Attentive ESRGAN', help='Label for model B')
    parser_compare.add_argument('--output-dir', type=str, help='Directory to save results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    if args.command == 'srcnn':
        evaluate_srcnn_model(args)
    elif args.command == 'srgan':
        evaluate_srgan_model(args)
    elif args.command == 'attentive-esrgan':
        evaluate_attentive_esrgan_model(args)
    elif args.command == 'compare':
        compare_models(args)


if __name__ == "__main__":
    main()

