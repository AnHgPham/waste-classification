"""
Model optimization utilities for deployment.

This module provides functions for:
- Converting Keras models to TensorFlow Lite
- Applying quantization (INT8) for edge devices
- Evaluating TFLite model performance

"""

import tensorflow as tf
import numpy as np
from pathlib import Path

from ..config import *


def convert_to_tflite(model, output_path):
    """
    Converts a Keras model to TensorFlow Lite format (no quantization).

    Arguments:
    model -- tf.keras.Model or str/Path. If string/Path, loads the model from that path.
              If Model object, uses it directly.
    output_path -- str or Path, path to save the TFLite model (.tflite).

    Returns:
    None
    """
    # Handle both model object and path
    if isinstance(model, (str, Path)):
        model = tf.keras.models.load_model(model)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"[OK] TFLite model saved to {output_path}")


def representative_dataset_generator(data_source, num_samples=100):
    """
    Generator function to provide representative data for INT8 quantization.

    Arguments:
    data_source -- Path or tf.data.Dataset. If Path, loads images from directory.
                   If Dataset, uses it directly.
    num_samples -- int, number of samples to use.

    Yields:
    List of input tensors for calibration.
    """
    from tensorflow import keras
    
    # Handle both directory path and dataset object
    if isinstance(data_source, (str, Path)):
        # Load a few images from directory
        dataset = keras.utils.image_dataset_from_directory(
            data_source,
            image_size=(IMG_SIZE[0], IMG_SIZE[1]),
            batch_size=1,
            shuffle=True
        )
    else:
        # Use provided dataset
        dataset = data_source.unbatch().batch(1)
    
    count = 0
    for images, _ in dataset:
        if count >= num_samples:
            break
        # Images should already be in [0, 255] range for MobileNetV2
        # Do NOT normalize here - let the model's preprocessing handle it
        yield [images]
        count += 1


def quantize_model(model, output_path, data_source=None):
    """
    Converts and quantizes a Keras model to TensorFlow Lite with INT8 quantization.

    INT8 QUANTIZATION MATHEMATICS:

    Quantization maps floating-point values to 8-bit integers:

    1. Find range of float values during calibration:
       r_min, r_max = min(weights), max(weights)

    2. Calculate quantization parameters:
       scale = (r_max - r_min) / (q_max - q_min)
             = (r_max - r_min) / 255  (for INT8: q_min=-128, q_max=127)
       zero_point = round(-r_min / scale)

    3. Quantization (float → int8):
       q = clip(round(r / scale) + zero_point, -128, 127)

    4. Dequantization (int8 → float) during inference:
       r = (q - zero_point) × scale

    Example:
       If r_min=-1.0, r_max=1.0:
         scale = (1.0 - (-1.0)) / 255 = 2.0 / 255 ≈ 0.00784
         zero_point = round(-(-1.0) / 0.00784) ≈ 128
       For r=0.5: q = round(0.5 / 0.00784) + 128 ≈ 192
       For r=-0.5: q = round(-0.5 / 0.00784) + 128 ≈ 64

    Benefits:
       - Model size reduction: ~75% (32-bit float → 8-bit int)
       - Faster inference on edge devices (INT8 operations)
       - Minimal accuracy loss (typically <1-2%)

    Arguments:
    model -- tf.keras.Model or str/Path. If string/Path, loads the model from that path.
             If Model object, uses it directly.
    output_path -- str or Path, path to save the quantized TFLite model.
    data_source -- Path, tf.data.Dataset, or None.
                   If Path: directory with training data for calibration.
                   If Dataset: uses it directly for calibration.
                   If None: uses TRAIN_DIR from config.

    Returns:
    None
    """
    if data_source is None:
        data_source = TRAIN_DIR
    
    # Handle both model object and path
    if isinstance(model, (str, Path)):
        model = tf.keras.models.load_model(model)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization flag
    # tf.lite.Optimize.DEFAULT enables weight quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Provide representative dataset for full integer quantization
    # The representative dataset is used to calibrate quantization parameters:
    #   - Run inference on ~100 sample images
    #   - Record min/max activation values at each layer
    #   - Calculate scale and zero_point for each layer
    # This ensures quantization preserves model accuracy
    converter.representative_dataset = lambda: representative_dataset_generator(data_source)

    # Ensure input/output are also quantized (full INT8 quantization)
    # TFLITE_BUILTINS_INT8: all operations use INT8 (no fallback to float)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Input type: uint8 (0-255) instead of float32
    # Output type: uint8 (needs dequantization to get probabilities)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_quant_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_quant_model)

    print(f"[OK] Quantized TFLite model saved to {output_path}")


def evaluate_tflite_model(tflite_model_path, test_dir, img_size, num_classes):
    """
    Evaluates a TFLite model on the test set.

    Arguments:
    tflite_model_path -- str or Path, path to the TFLite model.
    test_dir -- Path, directory containing test data.
    img_size -- tuple, (height, width) of input images.
    num_classes -- int, number of classes.

    Returns:
    accuracy -- float, test accuracy.
    """
    from tensorflow import keras
    import numpy as np
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check if model uses quantization
    input_dtype = input_details[0]['dtype']
    is_quantized = (input_dtype == np.uint8)

    # Load test data - keep in [0, 255] range
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=1,
        label_mode='categorical',
        shuffle=False
    )

    correct = 0
    total = 0

    for images, labels in test_ds:
        # Preprocess
        image = images[0].numpy()
        
        if is_quantized:
            # For quantized models: keep as uint8 in [0, 255]
            image = image.astype(np.uint8)
        else:
            # For float models: keep in [0, 255] range for MobileNetV2
            # The model has built-in preprocessing that will normalize to [-1, 1]
            image = image.astype(np.float32)
        
        image = np.expand_dims(image, axis=0)

        # Set input
        interpreter.set_tensor(input_details[0]['index'], image)

        # Run inference
        interpreter.invoke()

        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])

        # Dequantize if needed
        if is_quantized:
            # Dequantization formula: r = (q - zero_point) × scale
            # where:
            #   q = quantized output (uint8 or int8)
            #   r = real (float) value
            #   scale, zero_point = quantization parameters from calibration
            # Example: if q=192, scale=0.00392, zero_point=0:
            #          r = (192 - 0) × 0.00392 ≈ 0.753
            scale, zero_point = output_details[0]['quantization']
            output = scale * (output.astype(np.float32) - zero_point)

        # ArgMax: finds index of maximum value
        # prediction = argmax(output) = argmax([p0, p1, ..., p9])
        # Example: output=[0.1, 0.7, 0.2] → prediction=1
        prediction = np.argmax(output)
        true_label = np.argmax(labels[0].numpy())

        if prediction == true_label:
            correct += 1
        total += 1

    # Accuracy calculation:
    # accuracy = (number of correct predictions) / (total predictions)
    #          = TP / (TP + TN + FP + FN)
    # Example: if correct=950, total=1000:
    #          accuracy = 950 / 1000 = 0.95 = 95%
    accuracy = correct / total

    return accuracy

