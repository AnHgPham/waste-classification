"""
Baseline CNN model.

This module provides functions to build the baseline CNN architecture.

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..config import *


def build_baseline_model(input_shape, num_classes):
    """
    Builds a baseline Convolutional Neural Network model.

    The architecture consists of several convolutional blocks followed by a
    dense classifier head. Batch Normalization and Dropout are used for
    regularization.

    Arguments:
    input_shape -- tuple, shape of the input images (height, width, channels).
    num_classes -- int, number of output classes.

    Returns:
    model -- tf.keras.Model, the compiled CNN model.
    """
    model = keras.Sequential(name="Baseline_CNN")
    model.add(layers.Input(shape=input_shape))

    # CRITICAL: Rescale pixel values from [0, 255] to [0, 1]
    # This is essential for baseline CNN training stability
    # Formula: normalized_pixel = pixel / 255
    # Example: pixel=127 → normalized=127/255 ≈ 0.498
    # Range transformation: [0, 255] → [0, 1]
    model.add(layers.Rescaling(1./255))

    # Convolutional Blocks
    # Each block: Conv -> Conv -> Batch Norm -> Max Pooling
    # BASELINE_FILTERS = [32, 64, 128, 256] → creates 4 blocks
    #
    # Convolutional operation formula:
    #   Output[i,j,k] = ReLU(Σ(Input[i+m, j+n, c] × Kernel[m,n,c,k]) + Bias[k])
    # where:
    #   - (i,j) = spatial position, k = filter index, c = channel index
    #   - ReLU(x) = max(0, x)
    #   - padding='same' maintains spatial dimensions
    #
    # Batch Normalization formula:
    #   BN(x) = γ × ((x - μ) / √(σ² + ε)) + β
    # where:
    #   - μ = batch mean, σ² = batch variance
    #   - γ = learnable scale, β = learnable shift
    #   - ε = 1e-7 (numerical stability)
    #
    # Max Pooling (2x2):
    #   Output[i,j,k] = max(Input[2i:2i+2, 2j:2j+2, k])
    # Effect: reduces spatial dimensions by 2x (224→112→56→28→14)
    for filters in BASELINE_FILTERS:
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Classifier Head
    # Global Average Pooling formula:
    #   GAP(x)[channel] = (1 / (H × W)) × Σ(x[h,w,channel]) for all h,w
    # Reduces spatial dimensions to single value per channel
    # Example: (14, 14, 256) → (256,)
    model.add(layers.GlobalAveragePooling2D())

    # Dense layer formula:
    #   Output = ReLU(Weights @ Input + Bias)
    # Shape: (256,) → (128,)
    model.add(layers.Dense(BASELINE_DENSE_UNITS, activation='relu'))

    # Dropout regularization:
    #   During training: output = input × mask / (1 - dropout_rate)
    #   During inference: output = input (no dropout)
    # where mask is randomly 0 or 1 with probability = dropout_rate
    # BASELINE_DROPOUT_RATE = 0.5 (50% of neurons randomly dropped)
    model.add(layers.Dropout(BASELINE_DROPOUT_RATE))

    # Output layer with Softmax activation:
    #   Softmax(z_i) = exp(z_i) / Σ(exp(z_j)) for all j
    # Converts logits to probability distribution: Σ(output) = 1
    # Example: logits=[2.0, 1.0, 0.1] → probs=[0.659, 0.242, 0.099]
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

