import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

def get_dataset_stats(data_dir):
    """
    Calculates and returns statistics about the dataset.

    Arguments:
    data_dir -- Path, directory of the raw dataset.

    Returns:
    stats -- dict, a dictionary containing class names and image counts.
    """
    stats = {}
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if class_dir.exists():
            stats[class_name] = len(list(class_dir.glob('*.jpg')))
        else:
            stats[class_name] = 0
    return stats

def plot_class_distribution(stats, save_path=None):
    """
    Plots the class distribution as a bar chart.

    Arguments:
    stats -- dict, class statistics from get_dataset_stats.
    save_path -- Path or None, path to save the figure.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(stats.keys()), y=list(stats.values()))
    plt.title('Waste Classification Dataset - Class Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI)
        print(f"✅ Class distribution plot saved to {save_path}")
    else:
        plt.show()

def visualize_sample_images(data_dir, num_samples=5, save_path=None):
    """
    Visualizes sample images from each class.

    Arguments:
    data_dir -- Path, directory of the raw dataset.
    num_samples -- int, number of sample images per class.
    save_path -- Path or None, path to save the figure.
    """
    num_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(num_classes, num_samples, figsize=(15, 20))
    
    for i, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"⚠️ Warning: Class directory not found: {class_dir}")
            continue
            
        images = list(class_dir.glob('*.jpg'))[:num_samples]
        
        for j, img_path in enumerate(images):
            img = Image.open(img_path)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(class_name.upper(), fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI)
        print(f"✅ Sample images visualization saved to {save_path}")
    else:
        plt.show()

def main():
    """Main function to run data exploration."""
    print("=" * 70)
    print("WASTE CLASSIFICATION - DATA EXPLORATION")
    print("=" * 70)
    
    # Create output directories
    create_directories()
    
    # Check if raw data exists
    if not RAW_DATA_DIR.exists():
        print(f"\n❌ ERROR: Raw data directory not found: {RAW_DATA_DIR}")
        print("\n📥 Please download the dataset from:")
        print(f"   {DATASET_URL}")
        print(f"\n   And extract it to: {RAW_DATA_DIR}")
        return
    
    # Get dataset statistics
    print(f"\n📊 Analyzing dataset at: {RAW_DATA_DIR}")
    stats = get_dataset_stats(RAW_DATA_DIR)
    
    total_images = sum(stats.values())
    print(f"\n📈 Dataset Statistics:")
    print(f"   - Total classes: {len(CLASS_NAMES)}")
    print(f"   - Total images: {total_images}")
    print(f"\n   Class breakdown:")
    for class_name, count in stats.items():
        print(f"      {class_name:12s}: {count:5d} images")
    
    # Check for missing classes
    missing_classes = [c for c, count in stats.items() if count == 0]
    if missing_classes:
        print(f"\n⚠️ Warning: Missing data for classes: {missing_classes}")
        return
    
    # Plot class distribution
    print(f"\n📊 Generating class distribution plot...")
    dist_plot_path = REPORTS_DIR / "class_distribution.png"
    plot_class_distribution(stats, save_path=dist_plot_path)
    
    # Visualize sample images
    print(f"\n🖼️  Generating sample images visualization...")
    samples_plot_path = REPORTS_DIR / "sample_images.png"
    visualize_sample_images(RAW_DATA_DIR, num_samples=5, save_path=samples_plot_path)
    
    print("\n" + "=" * 70)
    print("✅ Data exploration complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()

