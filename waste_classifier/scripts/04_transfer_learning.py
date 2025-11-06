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
from src.models import build_transfer_model, unfreeze_layers
from src.training import plot_training_history

def main(args):
    """Main function for transfer learning."""
    print("=" * 70)
    print("WASTE CLASSIFICATION - TRANSFER LEARNING (MobileNetV2)")
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
    
    # =========================================================================
    # PHASE 1: FEATURE EXTRACTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: FEATURE EXTRACTION (Frozen Base)")
    print("=" * 70)
    
    print(f"\n🏗️  Building transfer learning model...")
    model = build_transfer_model(INPUT_SHAPE, NUM_CLASSES, freeze_base=True)
    
    print(f"\n📊 Model Architecture:")
    model.summary()
    
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_params = model.count_params()
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    
    # Compile for phase 1
    print(f"\n⚙️  Compiling model for Phase 1...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_TRANSFER_PHASE1),
        loss=LOSS_FUNCTION,
        metrics=METRICS
    )
    print(f"   - Learning rate: {LEARNING_RATE_TRANSFER_PHASE1}")
    
    # Setup callbacks for phase 1
    callbacks_phase1 = [
        keras.callbacks.EarlyStopping(
            monitor=EARLY_STOPPING_MONITOR,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=EARLY_STOPPING_RESTORE_BEST,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            get_model_path('mobilenetv2', 'phase1'),
            monitor=CHECKPOINT_MONITOR,
            save_best_only=CHECKPOINT_SAVE_BEST_ONLY,
            verbose=1
        )
    ]
    
    # Train phase 1
    phase1_epochs = args.phase1_epochs if args.phase1_epochs else EPOCHS_TRANSFER_PHASE1
    print(f"\n🎓 Training Phase 1 for {phase1_epochs} epochs...")
    
    history_phase1 = model.fit(
        train_ds,
        epochs=phase1_epochs,
        validation_data=val_ds,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    # Save phase 1 results
    phase1_plot_path = REPORTS_DIR / "mobilenetv2_phase1_history.png"
    plot_training_history(history_phase1, "Phase 1: Feature Extraction", save_path=phase1_plot_path)
    
    print(f"\n📈 Phase 1 Results:")
    print(f"   - Training Accuracy: {history_phase1.history['accuracy'][-1]:.4f}")
    print(f"   - Validation Accuracy: {history_phase1.history['val_accuracy'][-1]:.4f}")
    
    # =========================================================================
    # PHASE 2: FINE-TUNING
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: FINE-TUNING (Unfrozen Top Layers)")
    print("=" * 70)
    
    # Unfreeze top layers
    num_layers_to_unfreeze = args.unfreeze_layers if args.unfreeze_layers else 30
    print(f"\n🔓 Unfreezing top {num_layers_to_unfreeze} layers...")
    model = unfreeze_layers(model, num_layers_to_unfreeze)
    
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_params = model.count_params()
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    
    # Compile for phase 2 with lower learning rate
    print(f"\n⚙️  Compiling model for Phase 2...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_TRANSFER_PHASE2),
        loss=LOSS_FUNCTION,
        metrics=METRICS
    )
    print(f"   - Learning rate: {LEARNING_RATE_TRANSFER_PHASE2} (lower for fine-tuning)")
    
    # Setup callbacks for phase 2
    callbacks_phase2 = [
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
            get_model_path('mobilenetv2', 'final'),
            monitor=CHECKPOINT_MONITOR,
            save_best_only=CHECKPOINT_SAVE_BEST_ONLY,
            verbose=1
        )
    ]
    
    # Train phase 2
    phase2_epochs = args.phase2_epochs if args.phase2_epochs else EPOCHS_TRANSFER_PHASE2
    print(f"\n🎓 Training Phase 2 for {phase2_epochs} epochs...")
    
    history_phase2 = model.fit(
        train_ds,
        epochs=phase2_epochs,
        validation_data=val_ds,
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    # Save phase 2 results
    phase2_plot_path = REPORTS_DIR / "mobilenetv2_phase2_history.png"
    plot_training_history(history_phase2, "Phase 2: Fine-Tuning", save_path=phase2_plot_path)
    
    print(f"\n📈 Phase 2 Results:")
    print(f"   - Training Accuracy: {history_phase2.history['accuracy'][-1]:.4f}")
    print(f"   - Validation Accuracy: {history_phase2.history['val_accuracy'][-1]:.4f}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ Transfer Learning Complete!")
    print("=" * 70)
    print(f"\n📊 Training Summary:")
    print(f"   Phase 1 (Feature Extraction):")
    print(f"      - Best Val Accuracy: {max(history_phase1.history['val_accuracy']):.4f}")
    print(f"   Phase 2 (Fine-Tuning):")
    print(f"      - Best Val Accuracy: {max(history_phase2.history['val_accuracy']):.4f}")
    
    final_model_path = get_model_path('mobilenetv2', 'final')
    print(f"\n💾 Final model saved to: {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning with MobileNetV2')
    parser.add_argument('--phase1-epochs', type=int, default=None,
                        help=f'Phase 1 epochs (default: {EPOCHS_TRANSFER_PHASE1})')
    parser.add_argument('--phase2-epochs', type=int, default=None,
                        help=f'Phase 2 epochs (default: {EPOCHS_TRANSFER_PHASE2})')
    parser.add_argument('--unfreeze-layers', type=int, default=30,
                        help='Number of layers to unfreeze for fine-tuning (default: 30)')
    args = parser.parse_args()
    
    main(args)

