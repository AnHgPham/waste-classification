import os
import sys
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.data import split_data, create_data_generators

def main():
    """Main function to run data preprocessing."""
    print("=" * 70)
    print("WASTE CLASSIFICATION - DATA PREPROCESSING")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Create directories
    create_directories()
    
    # Check if raw data exists
    if not RAW_DATA_DIR.exists():
        print(f"\n❌ ERROR: Raw data directory not found: {RAW_DATA_DIR}")
        print("\n📥 Please download the dataset first.")
        print("   Run: python scripts/01_data_exploration.py")
        return
    
    # Split the data
    print(f"\n📂 Splitting data into train/val/test sets...")
    print(f"   - Train ratio: {TRAIN_RATIO:.0%}")
    print(f"   - Val ratio: {VAL_RATIO:.0%}")
    print(f"   - Test ratio: {TEST_RATIO:.0%}")
    print(f"   - Random seed: {RANDOM_SEED}")
    
    split_data(RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED)
    
    # Verify the split
    print(f"\n✅ Data split complete!")
    print(f"\n📊 Verifying split...")
    
    for split_name, split_dir in [("Train", TRAIN_DIR), ("Val", VAL_DIR), ("Test", TEST_DIR)]:
        total = 0
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob('*.jpg')))
                total += count
        print(f"   {split_name:5s}: {total:5d} images")
    
    # Create and test data generators
    print(f"\n🔄 Creating data generators...")
    train_ds, val_ds = create_data_generators(TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE, RANDOM_SEED)
    
    print(f"   ✅ Training dataset created")
    print(f"   ✅ Validation dataset created")
    
    # Test the generators
    print(f"\n🧪 Testing data generators...")
    for images, labels in train_ds.take(1):
        print(f"   - Batch shape: {images.shape}")
        print(f"   - Labels shape: {labels.shape}")
        print(f"   - Image value range: [{images.numpy().min():.2f}, {images.numpy().max():.2f}]")
    
    print("\n" + "=" * 70)
    print("✅ Data preprocessing complete!")
    print("=" * 70)
    print(f"\nProcessed data saved to: {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()

