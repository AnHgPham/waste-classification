"""
Transfer learning model using MobileNetV2.

This module provides functions for:
- Building a model using a pretrained base (MobileNetV2)
- Adding a custom classification head
- Freezing and unfreezing layers for fine-tuning

"""

import tensorflow as tf
from tensorflow import keras
from keras import layers

from ..config import *


def build_transfer_model(input_shape, num_classes, freeze_base=True):
    """
    Builds a transfer learning model using MobileNetV2 as the base.

    Arguments:
    input_shape -- tuple, shape of the input images.
    num_classes -- int, number of output classes.
    freeze_base -- bool, whether to freeze the base model layers.

    Returns:
    model -- tf.keras.Model, the compiled transfer learning model.
    """
    # 1. Load the pretrained base model
    # Get model class from keras.applications dynamically
    model_class = getattr(keras.applications, TRANSFER_BASE_MODEL)
    base_model = model_class(
        input_shape=input_shape,
        include_top=False,  # Do not include the original classifier
        weights=TRANSFER_WEIGHTS
    )

    # 2. Freeze the base model layers
    base_model.trainable = not freeze_base

    # 3. Create the new model
    inputs = keras.Input(shape=input_shape)
    
    # Preprocessing layer for MobileNetV2
    # Input should be in range [0, 255], will be normalized to [-1, 1]
    # Formula: normalized = (pixel / 127.5) - 1
    # Example: pixel=255 → (255/127.5)-1 = 1.0
    #          pixel=0   → (0/127.5)-1   = -1.0
    #          pixel=127 → (127/127.5)-1 ≈ -0.004
    # Range transformation: [0, 255] → [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Base model - use training mode appropriately for BatchNorm
    # When freeze_base=True, use training=False for frozen BN stats
    # When freeze_base=False (fine-tuning), use training=True to update BN
    x = base_model(x, training=not freeze_base)
    
    # 4. Add the classification head
    # Using a deeper classification head for better feature learning
    # Global Average Pooling:
    #   GAP(x)[channel] = mean(x[:,:,channel])
    # Reduces MobileNetV2 output (7, 7, 1280) → (1280,)
    x = layers.GlobalAveragePooling2D(name="GlobalAvgPool")(x)

    # First dense layer
    # Formula: Output = ReLU(W @ Input + b)
    # Shape: (1280,) → (256,) where TRANSFER_DENSE_UNITS=256
    x = layers.Dense(TRANSFER_DENSE_UNITS, activation='relu', name="Dense_1")(x)
    # Batch Normalization: BN(x) = γ × ((x - μ) / √(σ² + ε)) + β
    x = layers.BatchNormalization(name="BatchNorm_1")(x)
    # Dropout: randomly set TRANSFER_DROPOUT_RATE (30%) of activations to zero
    # Formula: output = input × mask / (1 - 0.3) during training
    x = layers.Dropout(TRANSFER_DROPOUT_RATE, name="Dropout_1")(x)

    # Second dense layer for more capacity
    # Integer division: TRANSFER_DENSE_UNITS // 2 = 256 // 2 = 128
    # Shape: (256,) → (128,)
    x = layers.Dense(TRANSFER_DENSE_UNITS // 2, activation='relu', name="Dense_2")(x)
    x = layers.BatchNormalization(name="BatchNorm_2")(x)
    x = layers.Dropout(TRANSFER_DROPOUT_RATE, name="Dropout_2")(x)

    # Output layer with Softmax:
    #   Softmax(z_i) = exp(z_i) / Σ(exp(z_j)) for all j in num_classes
    # Converts (128,) logits → (num_classes,) probabilities
    # Property: Σ(outputs) = 1.0, all values in [0, 1]
    outputs = layers.Dense(num_classes, activation='softmax', name="Classifier")(x)
    
    # 5. Build the final model
    model = keras.Model(inputs, outputs, name="MobileNetV2_Transfer_Learning")
    
    return model


def unfreeze_layers(model, num_layers_to_unfreeze):
    """
    Unfreezes the top N layers of the base model for fine-tuning.

    Fine-tuning mathematics:
    - Frozen layers: gradients are not computed, weights unchanged
    - Unfrozen layers: gradients flow, weights updated via:
      θ_new = θ_old - α × ∇L(θ)
    - Use low learning rate (α) to preserve pretrained features

    Strategy: Unfreeze top layers (closest to output) first as they contain
    task-specific features, while bottom layers contain general features

    Arguments:
    model -- tf.keras.Model, the model to modify.
    num_layers_to_unfreeze -- int, the number of layers to unfreeze from the top.

    Returns:
    model -- tf.keras.Model, the modified model.
    """
    # Get base model (name is 'mobilenetv2_1.00_224' not 'MobileNetV2')
    base_model = model.get_layer('mobilenetv2_1.00_224')
    base_model.trainable = True

    # Freeze all layers first
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze the top N layers
    # Python slicing: layers[-N:] gets last N elements
    # Example: if N=20, unfreezes layers at indices [-20, -19, ..., -2, -1]
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True

    print(f"Unfroze {num_layers_to_unfreeze} layers from the base model.")
    return model

