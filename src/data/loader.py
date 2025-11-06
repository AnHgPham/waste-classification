"""
Data loading utilities.

"""

import tensorflow as tf
from tensorflow import keras

from ..config import *


def load_dataset(data_dir, img_size, batch_size, shuffle=True, seed=None, normalize=True):
    """
    Load dataset from directory.

    Arguments:
    data_dir -- Path, directory containing class subfolders.
    img_size -- tuple, target image size (height, width).
    batch_size -- int, number of samples per batch.
    shuffle -- bool, whether to shuffle the data.
    seed -- int or None, random seed for reproducibility.
    normalize -- bool, whether to apply [0,1] normalization. 
                 Set to False when using transfer learning models with their own preprocessing.

    Returns:
    dataset -- tf.data.Dataset, loaded dataset.
    
    Note:
    - When using MobileNetV2 or other pretrained models, set normalize=False
    - MobileNetV2's preprocess_input expects [0,255] and normalizes to [-1,1]
    """
    dataset = keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=shuffle,
        seed=seed
    )
    
    # Apply normalization only if requested (for baseline models)
    # Skip normalization for transfer learning (let preprocess_input handle it)
    if normalize:
        normalization_layer = keras.layers.Rescaling(1./255)
        dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
    
    return dataset

