import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.deployment import convert_to_tflite, quantize_model, evaluate_tflite_model
from src.data import create_data_generators

def get_file_size(file_path):
    """Get file size in MB."""
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def create_representative_dataset(train_ds):
    """
    Create a representative dataset for quantization calibration.

    Arguments:
    train_ds -- tf.data.Dataset, training dataset.

    Returns:
    representative_ds -- tf.data.Dataset, representative dataset.
    """
    # Take a subset of training data for calibration
    representative_ds = train_ds.unbatch().take(100)
    return representative_ds

def main(args):
    """Main function for model optimization."""
    print("=" * 70)
    print("WASTE CLASSIFICATION - MODEL OPTIMIZATION")
    print("=" * 70)
    
    # Load model
    model_name = args.model
    model_path = get_model_path(model_name, 'final')
    
    if not model_path.exists():
        print(f"\n[ERROR] ERROR: Model not found: {model_path}")
        print(f"\n   Please train the {model_name} model first:")
        if model_name == 'baseline':
            print("   python scripts/week1_baseline_training.py")
        else:
            print("   python scripts/week2_transfer_learning.py")
        return
    
    print(f"\n[LOADING] Loading Keras model: {model_name}")
    model = tf.keras.models.load_model(model_path)
    print(f"   [OK] Model loaded from {model_path}")
    
    keras_size = get_file_size(model_path)
    print(f"   [INFO] Original Keras model size: {keras_size:.2f} MB")
    
    # =========================================================================
    # STEP 1: Convert to TensorFlow Lite (FP32)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Converting to TensorFlow Lite (FP32)")
    print("=" * 70)
    
    tflite_path = MODELS_DIR / f"{model_name}_fp32.tflite"
    print(f"\n[CONVERTING] Converting model to TFLite...")
    convert_to_tflite(model, tflite_path)
    
    tflite_size = get_file_size(tflite_path)
    print(f"   [INFO] TFLite FP32 model size: {tflite_size:.2f} MB")
    print(f"   [SIZE] Size reduction: {((keras_size - tflite_size) / keras_size * 100):.1f}%")
    
    # =========================================================================
    # STEP 2: Apply INT8 Quantization
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Applying INT8 Quantization")
    print("=" * 70)
    
    # Load training data for calibration
    print(f"\n[LOADING] Loading training data for quantization calibration...")
    train_ds, _ = create_data_generators(TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE, RANDOM_SEED)
    print(f"   [OK] Training dataset loaded (will use 100 samples for calibration)")
    
    tflite_quant_path = MODELS_DIR / f"{model_name}_int8.tflite"
    print(f"\n[CONVERTING] Quantizing model to INT8...")
    # Pass training dataset directly to quantize_model
    quantize_model(model, tflite_quant_path, train_ds)
    
    tflite_quant_size = get_file_size(tflite_quant_path)
    print(f"   [INFO] TFLite INT8 model size: {tflite_quant_size:.2f} MB")
    print(f"   [SIZE] Size reduction from FP32: {((tflite_size - tflite_quant_size) / tflite_size * 100):.1f}%")
    print(f"   [SIZE] Total size reduction from Keras: {((keras_size - tflite_quant_size) / keras_size * 100):.1f}%")
    
    # =========================================================================
    # STEP 3: Evaluate Models
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Evaluating Models")
    print("=" * 70)
    
    # Load test data
    print(f"\n[LOADING] Loading test dataset...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )
    
    # Apply preprocessing based on model type
    # MobileNetV2: Keep [0,255] range, preprocessing built into model
    # Baseline: Apply Rescaling(1./255)
    if model_name == 'baseline':
        print(f"   Applying baseline preprocessing (Rescaling 0-1)...")
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    else:
        print(f"   Using model's built-in preprocessing (MobileNetV2)...")
        # Images remain in [0, 255] range
    
    # Evaluate original Keras model
    print(f"\n[TESTING] Evaluating original Keras model...")
    keras_loss, keras_acc, keras_top5 = model.evaluate(test_ds, verbose=0)
    print(f"   [OK] Keras Model:")
    print(f"      - Accuracy: {keras_acc:.4f}")
    print(f"      - Top-5 Accuracy: {keras_top5:.4f}")
    print(f"      - Loss: {keras_loss:.4f}")
    
    # Evaluate TFLite FP32 model
    print(f"\n[TESTING] Evaluating TFLite FP32 model...")
    tflite_acc = evaluate_tflite_model(tflite_path, TEST_DIR, IMG_SIZE, NUM_CLASSES)
    print(f"   [OK] TFLite FP32 Model:")
    print(f"      - Accuracy: {tflite_acc:.4f}")
    print(f"      - Accuracy drop: {(keras_acc - tflite_acc):.4f}")
    
    # Evaluate TFLite INT8 model
    print(f"\n[TESTING] Evaluating TFLite INT8 quantized model...")
    tflite_quant_acc = evaluate_tflite_model(tflite_quant_path, TEST_DIR, IMG_SIZE, NUM_CLASSES)
    print(f"   [OK] TFLite INT8 Model:")
    print(f"      - Accuracy: {tflite_quant_acc:.4f}")
    print(f"      - Accuracy drop from Keras: {(keras_acc - tflite_quant_acc):.4f}")
    print(f"      - Accuracy drop from FP32: {(tflite_acc - tflite_quant_acc):.4f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("[OK] Model Optimization Complete!")
    print("=" * 70)
    
    print(f"\n[INFO] Optimization Summary:")
    print(f"\n   {'Model Type':<25} {'Size (MB)':<12} {'Accuracy':<10} {'Size Reduction'}")
    print(f"   {'-' * 70}")
    print(f"   {'Original Keras':<25} {keras_size:>8.2f}    {keras_acc:>8.4f}   {'baseline'}")
    print(f"   {'TFLite FP32':<25} {tflite_size:>8.2f}    {tflite_acc:>8.4f}   {((keras_size - tflite_size) / keras_size * 100):>5.1f}%")
    print(f"   {'TFLite INT8':<25} {tflite_quant_size:>8.2f}    {tflite_quant_acc:>8.4f}   {((keras_size - tflite_quant_size) / keras_size * 100):>5.1f}%")
    
    print(f"\n[SAVED] Optimized models saved to:")
    print(f"   - TFLite FP32: {tflite_path}")
    print(f"   - TFLite INT8: {tflite_quant_path}")
    
    print(f"\n[RECOMMEND] Recommendation:")
    if tflite_quant_acc >= keras_acc - 0.03:  # Less than 3% accuracy drop
        print(f"   [OK] Use INT8 quantized model for deployment")
        print(f"      - {((keras_size - tflite_quant_size) / keras_size * 100):.0f}% smaller with minimal accuracy loss")
        print(f"      - Ideal for edge devices (Raspberry Pi, mobile)")
    else:
        print(f"   [WARNING] INT8 quantization causes significant accuracy drop")
        print(f"      - Consider using FP32 TFLite model instead")
        print(f"      - Or collect more representative data for calibration")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize Model for Deployment')
    parser.add_argument('--model', type=str, default='mobilenetv2',
                        choices=['baseline', 'mobilenetv2'],
                        help='Model to optimize (default: mobilenetv2)')
    args = parser.parse_args()
    
    main(args)

