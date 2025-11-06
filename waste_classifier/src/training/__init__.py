"""
Training utilities for waste classification models.

This module provides helper functions for model training including:
- Training history visualization
"""

import matplotlib.pyplot as plt
from ..config import *


def plot_training_history(history, phase_name=None, save_path=None):
    """
    Plots training history (loss and accuracy).

    Arguments:
    history -- History object from model.fit()
    phase_name -- str or None, name of the training phase (e.g., 'Phase 1')
    save_path -- Path or None, path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=HISTORY_FIGSIZE)

    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    title = f'{phase_name} - Model Accuracy' if phase_name else 'Model Accuracy'
    ax1.set_title(title, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    title = f'{phase_name} - Model Loss' if phase_name else 'Model Loss'
    ax2.set_title(title, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI)
        print(f"✅ Training history plot saved to {save_path}")
    else:
        plt.show()


__all__ = ['plot_training_history']
