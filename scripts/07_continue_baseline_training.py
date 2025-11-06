"""
Continue Training Baseline CNN

This script allows you to resume training from a previously trained baseline model.
Useful when:
- Initial training accuracy is low
- Model hasn't converged yet
- You want to train longer without starting from scratch

Features:
- Load existing trained model
- Optionally adjust learning rate (usually lower)
- Train additional epochs
- Save with version number (v2, v3, etc.)
- Compare before/after performance

Usage:
    # Continue with default settings (20 more epochs, LR * 0.1)
    python scripts/07_continue_baseline_training.py

    # Custom epochs and learning rate
    python scripts/07_continue_baseline_training.py --epochs 30 --lr 0.0001

    # Keep original learning rate
    python scripts/07_continue_baseline_training.py --epochs 20 --keep-lr
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.data import create_data_generators

def find_latest_baseline_version():
    """
    Find the latest version of baseline model.

    Returns:
    latest_path -- Path, path to latest baseline model
    version -- int, version number (0 for 'final', 1+ for v1, v2, etc.)
    """
    # Check for versioned models first
    versions = []
    for i in range(1, 100):  # Check up to v99
        path = MODELS_DIR / f'baseline_v{i}.keras'
        if path.exists():
            versions.append((i, path))

    if versions:
        # Return highest version
        version, path = max(versions)
        return path, version

    # Check for original 'final' model
    final_path = get_model_path('baseline', 'final')
    if final_path.exists():
        return final_path, 0

    return None, None

def get_model_info(model):
    """
    Get information about the model.

    Arguments:
    model -- tf.keras.Model

    Returns:
    info -- dict, model information
    """
    info = {
        'total_params': model.count_params(),
        'trainable_params': sum([tf.size(w).numpy() for w in model.trainable_weights]),
        'layers': len(model.layers),
        'optimizer': model.optimizer.get_config()['name'] if hasattr(model, 'optimizer') else 'None',
        'learning_rate': float(model.optimizer.learning_rate) if hasattr(model, 'optimizer') else None
    }
    return info

def plot_training_comparison(history, old_history_path, save_path):
    """
    Plot training history comparing old and new training.

    Arguments:
    history -- History object from model.fit()
    old_history_path -- Path, path to old training history (if exists)
    save_path -- Path, path to save the comparison plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Load old history if exists
    old_history = None
    if old_history_path and old_history_path.exists():
        import pickle
        try:
            with open(old_history_path, 'rb') as f:
                old_history = pickle.load(f)
        except:
            pass

    # Plot accuracy
    epochs_new = range(1, len(history.history['accuracy']) + 1)
    ax1.plot(epochs_new, history.history['accuracy'], 'b-', linewidth=2, label='Train (New)')
    ax1.plot(epochs_new, history.history['val_accuracy'], 'r-', linewidth=2, label='Val (New)')

    if old_history:
        epochs_old = range(1, len(old_history['accuracy']) + 1)
        ax1.plot(epochs_old, old_history['accuracy'], 'b--', linewidth=1, alpha=0.5, label='Train (Old)')
        ax1.plot(epochs_old, old_history['val_accuracy'], 'r--', linewidth=1, alpha=0.5, label='Val (Old)')

    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot loss
    ax2.plot(epochs_new, history.history['loss'], 'b-', linewidth=2, label='Train (New)')
    ax2.plot(epochs_new, history.history['val_loss'], 'r-', linewidth=2, label='Val (New)')

    if old_history:
        ax2.plot(epochs_old, old_history['loss'], 'b--', linewidth=1, alpha=0.5, label='Train (Old)')
        ax2.plot(epochs_old, old_history['val_loss'], 'r--', linewidth=1, alpha=0.5, label='Val (Old)')

    ax2.set_title('Loss Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot accuracy improvement per epoch
    ax3.plot(epochs_new, history.history['val_accuracy'], 'g-', linewidth=2, marker='o')
    ax3.set_title('Validation Accuracy (Continued Training)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Val Accuracy')
    ax3.grid(True, alpha=0.3)

    # Summary statistics
    summary_text = f"""
    TRAINING SUMMARY

    New Training:
    - Epochs: {len(history.history['accuracy'])}
    - Final Train Acc: {history.history['accuracy'][-1]:.4f}
    - Final Val Acc: {history.history['val_accuracy'][-1]:.4f}
    - Best Val Acc: {max(history.history['val_accuracy']):.4f}

    Improvement:
    - Start Val Acc: {history.history['val_accuracy'][0]:.4f}
    - End Val Acc: {history.history['val_accuracy'][-1]:.4f}
    - Gain: {history.history['val_accuracy'][-1] - history.history['val_accuracy'][0]:.4f}
    """

    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   [OK] Comparison plot saved to {save_path}")
    plt.close()

def save_training_history(history, save_path):
    """
    Save training history to pickle file.

    Arguments:
    history -- History object from model.fit()
    save_path -- Path, path to save history
    """
    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"   [OK] Training history saved to {save_path}")

def main(args):
    """Main function for continuing baseline training."""
    print("=" * 70)
    print("CONTINUE TRAINING BASELINE CNN")
    print("=" * 70)

    # Set random seeds
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Create directories
    create_directories()

    # =========================================================================
    # STEP 1: LOAD EXISTING MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: LOADING EXISTING MODEL")
    print("=" * 70)

    # Find latest baseline model
    model_path, current_version = find_latest_baseline_version()

    if model_path is None:
        print(f"\n[ERROR] No trained baseline model found!")
        print("\nPlease train the baseline model first:")
        print("  python scripts/03_baseline_training.py")
        print("  OR")
        print("  python main.py --train-baseline")
        return

    print(f"\n[FOUND] Latest baseline model:")
    print(f"  Path: {model_path}")
    print(f"  Version: {'final' if current_version == 0 else f'v{current_version}'}")
    print(f"  Size: {model_path.stat().st_size / (1024*1024):.2f} MB")

    # Load model
    print(f"\n[LOADING] Model...")
    model = keras.models.load_model(model_path)
    print(f"  [OK] Model loaded successfully")

    # Get model info
    info = get_model_info(model)
    print(f"\n[MODEL INFO]")
    print(f"  Total parameters: {info['total_params']:,}")
    print(f"  Trainable parameters: {info['trainable_params']:,}")
    print(f"  Layers: {info['layers']}")
    print(f"  Current optimizer: {info['optimizer']}")
    if info['learning_rate']:
        print(f"  Current learning rate: {info['learning_rate']:.6f}")

    # Evaluate current performance
    print(f"\n[EVALUATING] Current model performance...")
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        print(f"  [WARNING] Processed data not found. Skipping evaluation.")
        initial_val_acc = None
    else:
        _, val_ds = create_data_generators(TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE, RANDOM_SEED)
        results = model.evaluate(val_ds, verbose=0)
        initial_val_acc = results[1]  # Assuming accuracy is second metric
        print(f"  [CURRENT] Validation Accuracy: {initial_val_acc:.4f} ({initial_val_acc*100:.2f}%)")
        print(f"  [CURRENT] Validation Loss: {results[0]:.4f}")

    # =========================================================================
    # STEP 2: CONFIGURE CONTINUED TRAINING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: CONFIGURING CONTINUED TRAINING")
    print("=" * 70)

    # Determine new learning rate
    if args.keep_lr:
        new_lr = info['learning_rate'] if info['learning_rate'] else LEARNING_RATE_BASELINE
        print(f"\n[LEARNING RATE] Keeping original: {new_lr:.6f}")
    elif args.lr:
        new_lr = args.lr
        print(f"\n[LEARNING RATE] Using custom: {new_lr:.6f}")
    else:
        # Default: reduce by 10x
        old_lr = info['learning_rate'] if info['learning_rate'] else LEARNING_RATE_BASELINE
        new_lr = old_lr * 0.1
        print(f"\n[LEARNING RATE] Reducing 10x: {old_lr:.6f} -> {new_lr:.6f}")

    # Recompile model with new learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=new_lr),
        loss=LOSS_FUNCTION,
        metrics=METRICS
    )
    print(f"  [OK] Model recompiled with new learning rate")

    # Training configuration
    additional_epochs = args.epochs
    print(f"\n[TRAINING CONFIG]")
    print(f"  Additional epochs: {additional_epochs}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {new_lr:.6f}")

    # =========================================================================
    # STEP 3: LOAD DATA
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: LOADING TRAINING DATA")
    print("=" * 70)

    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        print(f"\n[ERROR] Processed data not found!")
        print("  Please run preprocessing first:")
        print("  python scripts/02_preprocessing.py")
        return

    print(f"\n[LOADING] Datasets...")
    train_ds, val_ds = create_data_generators(TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE, RANDOM_SEED)
    print(f"  [OK] Datasets loaded")

    # =========================================================================
    # STEP 4: CONTINUE TRAINING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: CONTINUING TRAINING")
    print("=" * 70)

    # Determine next version number
    next_version = current_version + 1
    next_model_path = MODELS_DIR / f'baseline_v{next_version}.keras'

    print(f"\n[TRAINING] Starting continued training...")
    print(f"  Progress will be saved to: {next_model_path}")

    # Setup callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=EARLY_STOPPING_MONITOR,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=EARLY_STOPPING_RESTORE_BEST,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=REDUCE_LR_MONITOR,
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN_LR,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            next_model_path,
            monitor=CHECKPOINT_MONITOR,
            save_best_only=CHECKPOINT_SAVE_BEST_ONLY,
            verbose=1
        )
    ]

    # Train
    history = model.fit(
        train_ds,
        epochs=additional_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    # =========================================================================
    # STEP 5: EVALUATE IMPROVEMENT
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: EVALUATING IMPROVEMENT")
    print("=" * 70)

    # Evaluate final model
    print(f"\n[EVALUATING] Final model performance...")
    final_results = model.evaluate(val_ds, verbose=0)
    final_val_acc = final_results[1]
    final_val_loss = final_results[0]

    print(f"\n[RESULTS]")
    print(f"  Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(f"  Final Validation Loss: {final_val_loss:.4f}")

    if initial_val_acc:
        improvement = final_val_acc - initial_val_acc
        improvement_pct = (improvement / initial_val_acc) * 100
        print(f"\n[IMPROVEMENT]")
        print(f"  Before: {initial_val_acc:.4f} ({initial_val_acc*100:.2f}%)")
        print(f"  After:  {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        print(f"  Gain:   {improvement:+.4f} ({improvement_pct:+.2f}%)")

        if improvement > 0:
            print(f"\n  [SUCCESS] Model improved! Accuracy increased by {improvement:.4f}")
        elif improvement < -0.01:
            print(f"\n  [WARNING] Model got worse! Consider:")
            print(f"    - Using lower learning rate")
            print(f"    - Training for fewer epochs")
            print(f"    - Checking for overfitting")
        else:
            print(f"\n  [INFO] Minimal change. Model may have converged.")

    # =========================================================================
    # STEP 6: SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: SAVING RESULTS")
    print("=" * 70)

    # Save model
    print(f"\n[SAVING] Model as version {next_version}...")
    print(f"  Path: {next_model_path}")

    # Save training history
    history_path = REPORTS_DIR / f'baseline_v{next_version}_history.pkl'
    save_training_history(history, history_path)

    # Create comparison plot
    old_history_path = REPORTS_DIR / f'baseline_v{current_version}_history.pkl' if current_version > 0 else None
    plot_path = REPORTS_DIR / f'baseline_v{next_version}_comparison.png'
    plot_training_comparison(history, old_history_path, plot_path)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"\n[SUMMARY]")
    print(f"  Original model: {model_path.name}")
    print(f"  New model: {next_model_path.name}")
    print(f"  Additional epochs trained: {additional_epochs}")
    print(f"  Learning rate used: {new_lr:.6f}")
    print(f"  Final validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")

    if initial_val_acc and improvement > 0:
        print(f"  Improvement: +{improvement:.4f} ({improvement_pct:+.2f}%)")

    print(f"\n[FILES SAVED]")
    print(f"  Model: {next_model_path}")
    print(f"  History: {history_path}")
    print(f"  Plot: {plot_path}")

    print(f"\n[NEXT STEPS]")
    if final_val_acc < 0.90:
        print(f"  - Accuracy still below 90%. Consider:")
        print(f"    - Continue training more: python scripts/07_continue_baseline_training.py --epochs 30")
        print(f"    - Try transfer learning: python main.py --train-transfer")
    else:
        print(f"  - Good accuracy! Consider:")
        print(f"    - Evaluate on test set: python main.py --evaluate --model baseline")
        print(f"    - Try transfer learning for even better results")

    print("\n" + "=" * 70)
    print("[DONE]")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Continue training baseline CNN model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Continue with default settings (20 epochs, LR * 0.1)
  python scripts/07_continue_baseline_training.py

  # Train 30 more epochs
  python scripts/07_continue_baseline_training.py --epochs 30

  # Use custom learning rate
  python scripts/07_continue_baseline_training.py --epochs 20 --lr 0.0001

  # Keep original learning rate
  python scripts/07_continue_baseline_training.py --epochs 20 --keep-lr
        """
    )

    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of additional epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Custom learning rate (default: current_lr * 0.1)')
    parser.add_argument('--keep-lr', action='store_true',
                        help='Keep the original learning rate instead of reducing it')

    args = parser.parse_args()

    main(args)
