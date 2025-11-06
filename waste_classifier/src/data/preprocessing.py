"""
Data preprocessing utilities.

This module provides functions for:
- Splitting raw data into train/val/test sets
- Creating TensorFlow data generators
- Applying data augmentation

"""

import os
import shutil
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..config import *


def split_data(raw_dir, processed_dir, train_ratio, val_ratio, test_ratio, seed):
    """
    Splits raw image data into train, validation, and test sets.

    Arguments:
    raw_dir -- Path, directory with raw images organized in class subfolders.
    processed_dir -- Path, directory to save the split datasets.
    train_ratio -- float, proportion of data for training.
    val_ratio -- float, proportion of data for validation.
    test_ratio -- float, proportion of data for testing.
    seed -- int, random seed for reproducibility.

    Returns:
    None
    """
    random.seed(seed)
    
    # Check if processed data already exists AND has images
    train_dir_check = processed_dir / 'train'
    if processed_dir.exists() and train_dir_check.exists():
        # Count images in train directory
        num_images = sum(1 for _ in train_dir_check.glob('*/*.jpg'))
        if num_images > 0:
            print(f"⚠️  Processed data directory already exists with {num_images} training images. Skipping split.")
            return
        else:
            print(f"⚠️  Processed directory exists but is empty. Re-splitting data...")
    
    print(f"Splitting data from {raw_dir} to {processed_dir}...")

    for class_name in CLASS_NAMES:
        class_dir = raw_dir / class_name
        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)

        num_images = len(images)
        # Data split calculation:
        # num_train = floor(N × train_ratio) = floor(N × 0.8)
        # num_val = floor(N × val_ratio) = floor(N × 0.1)
        # num_test = N - num_train - num_val (ensures all images used)
        # Example: if N=100 images:
        #   num_train = floor(100 × 0.8) = 80
        #   num_val = floor(100 × 0.1) = 10
        #   num_test = 100 - 80 - 10 = 10
        num_train = int(num_images * train_ratio)
        num_val = int(num_images * val_ratio)

        # Array slicing: images[start:end]
        # train_images: indices [0, num_train)
        # val_images: indices [num_train, num_train + num_val)
        # test_images: indices [num_train + num_val, end)
        train_images = images[:num_train]
        val_images = images[num_train : num_train + num_val]
        test_images = images[num_train + num_val:]

        # Create directories
        (processed_dir / 'train' / class_name).mkdir(parents=True, exist_ok=True)
        (processed_dir / 'val' / class_name).mkdir(parents=True, exist_ok=True)
        (processed_dir / 'test' / class_name).mkdir(parents=True, exist_ok=True)

        # Copy files
        for img in train_images:
            shutil.copy(img, processed_dir / 'train' / class_name / img.name)
        for img in val_images:
            shutil.copy(img, processed_dir / 'val' / class_name / img.name)
        for img in test_images:
            shutil.copy(img, processed_dir / 'test' / class_name / img.name)

    print("✅ Data splitting complete.")


def create_data_generators(train_dir, val_dir, img_size, batch_size, seed):
    """
    Creates training and validation data generators with augmentation.

    Arguments:
    train_dir -- Path, directory for training data.
    val_dir -- Path, directory for validation data.
    img_size -- tuple, target image size (height, width).
    batch_size -- int, number of samples per batch.
    seed -- int, random seed for reproducibility.

    Returns:
    train_ds -- tf.data.Dataset, training dataset.
    val_ds -- tf.data.Dataset, validation dataset.
    
    Note:
    - Images are kept in range [0, 255] for MobileNetV2 preprocessing
    - DO NOT apply Rescaling(1./255) - MobileNetV2 will handle normalization
    """
    # Data augmentation pipeline - operates on [0, 255] range
    # More aggressive augmentation for better generalization
    #
    # AUGMENTATION FORMULAS:
    #
    # 1. RandomFlip("horizontal"):
    #    new_pixel[i,j] = original_pixel[i, W-1-j] with 50% probability
    #
    # 2. RandomRotation(rotation_factor=0.2):
    #    angle = 0.2 × 2π × random(-1, 1) = ±0.4π radians = ±72°
    #    Applies rotation matrix: R(θ) = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
    #
    # 3. RandomZoom(zoom_factor=0.2):
    #    scale = 1 + 0.2 × random(-1, 1) ∈ [0.8, 1.2]
    #    new_coords = original_coords × scale
    #    zoom_in: scale > 1 (crops center), zoom_out: scale < 1 (adds padding)
    #
    # 4. RandomContrast(contrast_factor=0.2):
    #    factor = 1 + 0.2 × random(-1, 1) ∈ [0.8, 1.2]
    #    mean_pixel = mean(image)
    #    new_pixel = clip((pixel - mean_pixel) × factor + mean_pixel, 0, 255)
    #    Higher factor increases contrast, lower factor decreases contrast
    augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(AUGMENTATION_CONFIG['rotation_factor']),
        layers.RandomZoom(AUGMENTATION_CONFIG['zoom_factor']),
        layers.RandomContrast(AUGMENTATION_CONFIG['contrast_factor']),
    ]
    
    # Add optional augmentations if enabled in config
    if AUGMENTATION_CONFIG.get('brightness_factor', 0) > 0:
        # RandomBrightness(brightness_factor=0.1):
        #   delta = 0.1 × 255 × random(-1, 1) ∈ [-25.5, 25.5]
        #   new_pixel = clip(pixel + delta, 0, 255)
        #   Positive delta brightens, negative delta darkens
        augmentation_layers.append(
            layers.RandomBrightness(AUGMENTATION_CONFIG['brightness_factor'])
        )

    if AUGMENTATION_CONFIG.get('width_shift_factor', 0) > 0 or AUGMENTATION_CONFIG.get('height_shift_factor', 0) > 0:
        # RandomTranslation(height_factor=0.1, width_factor=0.1):
        #   For image of size (224, 224):
        #   shift_x = 224 × 0.1 × random(-1, 1) ∈ [-22.4, 22.4] pixels
        #   shift_y = 224 × 0.1 × random(-1, 1) ∈ [-22.4, 22.4] pixels
        #   new_pixel[i,j] = original_pixel[i+shift_y, j+shift_x]
        #   Simulates camera movement or object position variation
        augmentation_layers.append(
            layers.RandomTranslation(
                height_factor=AUGMENTATION_CONFIG.get('height_shift_factor', 0),
                width_factor=AUGMENTATION_CONFIG.get('width_shift_factor', 0)
            )
        )
    
    data_augmentation = keras.Sequential(augmentation_layers, name="data_augmentation")

    # Create datasets
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True,
        seed=seed
    )

    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False,
        seed=seed
    )

    # Apply augmentation to training data only
    # IMPORTANT: Do NOT apply Rescaling here! 
    # MobileNetV2's preprocess_input expects [0, 255] and will normalize to [-1, 1]
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y), 
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Prefetch for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds

