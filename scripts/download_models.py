"""
Download Pre-trained Models Script

This script downloads pre-trained models from GitHub Releases.
Use this if you don't want to train from scratch or if Git LFS is not available.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --model all
    python scripts/download_models.py --model mobilenetv2

"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODELS_DIR

# GitHub release URL (update after creating release)
GITHUB_REPO = "YOUR_USERNAME/waste_classifier_capstone"
RELEASE_TAG = "v1.0-models"
BASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

# Available models
MODELS = {
    'mobilenetv2': {
        'file': 'mobilenetv2_final.keras',
        'url': f'{BASE_URL}/mobilenetv2_final.keras',
        'size': '25.02 MB',
        'accuracy': '93.90%',
        'description': 'MobileNetV2 transfer learning model (RECOMMENDED)'
    },
    'mobilenetv2_fp32': {
        'file': 'mobilenetv2_fp32.tflite',
        'url': f'{BASE_URL}/mobilenetv2_fp32.tflite',
        'size': '9.84 MB',
        'accuracy': '93.90%',
        'description': 'TFLite FP32 optimized model (for deployment)'
    },
    'mobilenetv2_int8': {
        'file': 'mobilenetv2_int8.tflite',
        'url': f'{BASE_URL}/mobilenetv2_int8.tflite',
        'size': '2.94 MB',
        'accuracy': '93.20%',
        'description': 'TFLite INT8 quantized model (for edge devices)'
    },
    'baseline': {
        'file': 'baseline_final.keras',
        'url': f'{BASE_URL}/baseline_final.keras',
        'size': '5.3 MB',
        'accuracy': '79.59%',
        'description': 'Baseline CNN model (for comparison)'
    }
}

def download_file(url, dest_path):
    """
    Download file from URL with progress bar.

    Arguments:
    url -- str, URL to download from.
    dest_path -- Path, destination file path.

    Returns:
    bool -- True if successful, False otherwise.
    """
    try:
        print(f"[DOWNLOADING] {dest_path.name}")
        print(f"   URL: {url}")

        # Download with progress
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r   Progress: {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, reporthook)
        print(f"\n   [OK] Downloaded successfully")
        return True

    except Exception as e:
        print(f"\n   [ERROR] Download failed: {e}")
        return False

def main(args):
    """Main function for downloading models."""
    print("=" * 70)
    print("WASTE CLASSIFICATION - MODEL DOWNLOADER")
    print("=" * 70)

    # Create models directory if not exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which models to download
    if args.model == 'all':
        models_to_download = MODELS.keys()
    else:
        if args.model not in MODELS:
            print(f"\n[ERROR] Unknown model: {args.model}")
            print(f"\nAvailable models: {', '.join(MODELS.keys())}")
            return
        models_to_download = [args.model]

    # Display available models
    print("\n[INFO] Available Models:")
    print("=" * 70)
    for name, info in MODELS.items():
        marker = " <-- DOWNLOADING" if name in models_to_download else ""
        print(f"\n{name}{marker}")
        print(f"   File: {info['file']}")
        print(f"   Size: {info['size']}")
        print(f"   Accuracy: {info['accuracy']}")
        print(f"   Description: {info['description']}")

    # Download models
    print("\n" + "=" * 70)
    print("[START] Downloading Models")
    print("=" * 70)

    success_count = 0
    fail_count = 0

    for model_name in models_to_download:
        info = MODELS[model_name]
        dest_path = MODELS_DIR / info['file']

        # Check if already exists
        if dest_path.exists() and not args.force:
            print(f"\n[SKIP] {info['file']} already exists (use --force to re-download)")
            success_count += 1
            continue

        # Download
        print()
        if download_file(info['url'], dest_path):
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("[SUMMARY]")
    print("=" * 70)
    print(f"   Successfully downloaded: {success_count}")
    print(f"   Failed: {fail_count}")
    print(f"   Models saved to: {MODELS_DIR}")

    if success_count > 0:
        print("\n[OK] You can now use the models without training!")
        print("\n   Example usage:")
        print("   python scripts/08_image_detection.py --image test.jpg")
        print("   python scripts/99_evaluate_model.py --model mobilenetv2")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Pre-trained Models')
    parser.add_argument('--model', type=str, default='mobilenetv2',
                        choices=list(MODELS.keys()) + ['all'],
                        help='Model to download (default: mobilenetv2)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if file exists')
    args = parser.parse_args()

    main(args)
