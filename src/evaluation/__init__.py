"""
Evaluation utilities for waste classification models.

This module provides helper functions for model evaluation including:
- Confusion matrix visualization
- Classification metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ..config import *


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plots a confusion matrix.

    Arguments:
    cm -- np.ndarray, confusion matrix.
    class_names -- list, list of class names.
    save_path -- Path or None, path to save the figure.
    """
    plt.figure(figsize=CM_FIGSIZE)

    # Normalize confusion matrix if specified
    # Normalization formula:
    #   cm_normalized[i,j] = cm[i,j] / sum(cm[i,:])
    # Converts counts to proportions (row-wise normalization)
    # Example row: [80, 10, 5, 5] → [0.8, 0.1, 0.05, 0.05]
    # This shows: 80% correct, 10% confused with class j, etc.
    if CM_NORMALIZE:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm, annot=True, fmt='.2f' if CM_NORMALIZE else 'd',
                cmap=CM_CMAP, xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion' if CM_NORMALIZE else 'Count'})

    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI)
        print(f"[OK] Confusion matrix saved to {save_path}")
    else:
        plt.show()


__all__ = ['plot_confusion_matrix']
