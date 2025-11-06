import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.data import create_data_generators
from src.models import build_baseline_model
from src.training import plot_training_history

def main(args):
    """Main function to train baseline model."""
    print("=" * 70)
    print("WASTE CLASSIFICATION - BASELINE CNN TRAINING")
    print("=" * 70)
    
    # Set random seeds
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Create directories
    create_directories()
    
    # Check if processed data exists
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        print(f"\n❌ ERROR: Processed data not found!")
        print("   Please run preprocessing first:")
        print("   python scripts/week1_preprocessing.py")
        return
    
    # Load datasets
    print(f"\n📂 Loading datasets...")
    train_ds, val_ds = create_data_generators(TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE, RANDOM_SEED)
    print(f"   ✅ Datasets loaded")
    
    # Build model
    print(f"\n🏗️  Building baseline CNN model...")
    model = build_baseline_model(INPUT_SHAPE, NUM_CLASSES)
    
    print(f"\n📊 Model Architecture:")
    model.summary()
    
    total_params = model.count_params()
    print(f"\n   Total parameters: {total_params:,}")
    
    # Compile model
    print(f"\n⚙️  Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_BASELINE),
        loss=LOSS_FUNCTION,
        metrics=METRICS
    )
    print(f"   - Optimizer: Adam (LR={LEARNING_RATE_BASELINE})")
    print(f"   - Loss: {LOSS_FUNCTION}")
    print(f"   - Metrics: {METRICS}")
    
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
            get_model_path('baseline', 'final'),
            monitor=CHECKPOINT_MONITOR,
            save_best_only=CHECKPOINT_SAVE_BEST_ONLY,
            verbose=1
        )
    ]
    
    # Train model
    epochs = args.epochs if args.epochs else EPOCHS_BASELINE
    print(f"\n🎓 Training model for {epochs} epochs...")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"   - Learning rate reduction patience: {REDUCE_LR_PATIENCE}")
    
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = get_model_path('baseline', 'final')
    print(f"\n💾 Saving final model to {final_model_path}")
    
    # Plot training history
    print(f"\n📊 Generating training history plots...")
    history_plot_path = REPORTS_DIR / "baseline_training_history.png"
    plot_training_history(history, save_path=history_plot_path)
    
    # Final evaluation
    print(f"\n📈 Final Training Results:")
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"   - Training Accuracy: {final_train_acc:.4f}")
    print(f"   - Validation Accuracy: {final_val_acc:.4f}")
    print(f"   - Training Loss: {final_train_loss:.4f}")
    print(f"   - Validation Loss: {final_val_loss:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Baseline CNN training complete!")
    print("=" * 70)
    print(f"\nModel saved to: {final_model_path}")
    print(f"Training history plot: {history_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Baseline CNN Model')
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'Number of training epochs (default: {EPOCHS_BASELINE})')
    args = parser.parse_args()
    
    main(args)

