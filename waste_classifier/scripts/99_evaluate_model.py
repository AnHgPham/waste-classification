"""
Model Evaluation Script

This script evaluates a trained model on the test set and generates:
- Confusion matrix
- Classification report
- Per-class accuracy

Usage:
    python scripts/evaluate_model.py --model mobilenetv2
    python scripts/evaluate_model.py --model baseline

"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.evaluation import plot_confusion_matrix

def main(args):
    """Main function for model evaluation."""
    print("=" * 70)
    print("WASTE CLASSIFICATION - MODEL EVALUATION")
    print("=" * 70)
    
    # Load model
    model_name = args.model
    model_path = get_model_path(model_name, 'final')
    
    if not model_path.exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        print(f"\n   Please train the {model_name} model first.")
        return
    
    print(f"\n[LOADING] Model: {model_name}")
    model = tf.keras.models.load_model(model_path)
    print(f"   [OK] Model loaded from {model_path}")

    # Load test dataset
    print(f"\n[LOADING] Test dataset...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )
    
    # Apply appropriate preprocessing based on model type
    # IMPORTANT: Both baseline and MobileNetV2 have preprocessing built-in!
    # Baseline: Has Rescaling(1./255) as first layer
    # MobileNetV2: Has preprocess_input built-in
    # Therefore, NO additional preprocessing needed for either model
    print(f"   Using model's built-in preprocessing...")
    
    print(f"   [OK] Test dataset loaded")

    # Evaluate model
    print(f"\n[EVALUATING] Model on test set...")
    # model.evaluate computes:
    # 1. Loss = Categorical Cross-Entropy:
    #    Loss = -(1/N) × Σ Σ y_true[i,j] × log(y_pred[i,j])
    #    where i = sample index, j = class index
    #
    # 2. Accuracy (Top-1):
    #    Accuracy = (correct predictions) / (total predictions)
    #    correct = 1 if argmax(y_pred) == argmax(y_true), else 0
    #
    # 3. Top-5 Accuracy:
    #    Top5_Acc = 1 if true_class in top_5_predictions, else 0
    #    Measures if true class is among 5 highest predicted probabilities
    test_loss, test_acc, test_top5 = model.evaluate(test_ds, verbose=1)

    print(f"\n[RESULTS] Test Results:")
    print(f"   - Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   - Top-5 Accuracy: {test_top5:.4f} ({test_top5*100:.2f}%)")
    print(f"   - Test Loss: {test_loss:.4f}")

    # Generate predictions for detailed analysis
    print(f"\n[GENERATING] Predictions...")
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        # ArgMax extracts class index from one-hot or probability distribution
        # Example: labels = [[0, 1, 0, ...]] → argmax = 1
        #          predictions = [[0.1, 0.7, 0.2, ...]] → argmax = 1
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(f"   [OK] {len(y_true)} predictions generated")

    # Generate confusion matrix
    # Confusion Matrix formula:
    #   CM[i,j] = count of samples with true_label=i and predicted_label=j
    # Structure:
    #                   Predicted
    #            | class0 | class1 | class2 | ...
    #   ---------|--------|--------|--------|----
    #   True  0  |   TP   |   FP   |   FP   | ...  (row sums to total class 0)
    #         1  |   FN   |   TP   |   FP   | ...
    #         2  |   FN   |   FN   |   TP   | ...
    # Diagonal = correct predictions (TP)
    # Off-diagonal = misclassifications
    print(f"\n[GENERATING] Confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    cm_path = get_report_path(model_name, 'confusion_matrix')
    plot_confusion_matrix(cm, CLASS_NAMES, save_path=cm_path)

    # Generate classification report
    # Classification metrics for each class:
    #
    # 1. Precision (Positive Predictive Value):
    #    Precision = TP / (TP + FP)
    #    "Of all predictions for this class, how many were correct?"
    #    High precision = few false positives
    #
    # 2. Recall (Sensitivity, True Positive Rate):
    #    Recall = TP / (TP + FN)
    #    "Of all actual instances of this class, how many did we find?"
    #    High recall = few false negatives
    #
    # 3. F1-Score (Harmonic Mean of Precision and Recall):
    #    F1 = 2 × (Precision × Recall) / (Precision + Recall)
    #    Balances precision and recall
    #    F1 = 1 (perfect), F1 = 0 (worst)
    #
    # 4. Support:
    #    Number of actual occurrences of this class in test set
    print(f"\n[GENERATING] Classification report...")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(report)
    
    # Save classification report
    report_path = get_report_path(model_name, 'classification_report')
    with open(report_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(report)
    print(f"\n[OK] Classification report saved to {report_path}")

    # Per-class accuracy
    # Class-specific accuracy formula:
    #   Accuracy_class_i = (correct predictions for class i) / (total instances of class i)
    #                    = TP_i / (TP_i + FN_i)
    #                    = Recall for class i
    # Example: if class "battery" has 100 instances and 95 correct predictions:
    #          Accuracy_battery = 95 / 100 = 0.95 = 95%
    print(f"\n[RESULTS] Per-Class Accuracy:")
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = (y_true == i)
        class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
        print(f"   {class_name:12s}: {class_acc:.4f} ({class_acc*100:.2f}%)")

    # Find most confused classes
    # Confusion analysis:
    #   cm_normalized[i,j] = proportion of class i predicted as class j
    #   Higher values (off-diagonal) = more confusion between classes
    # Example: cm_normalized[battery, metal] = 0.15 means:
    #          15% of battery samples were misclassified as metal
    print(f"\n[ANALYSIS] Most Confused Classes (Top 5):")
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.fill_diagonal(cm_normalized, 0)  # Ignore diagonal (correct predictions)

    confused_pairs = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j:
                confused_pairs.append((CLASS_NAMES[i], CLASS_NAMES[j], cm_normalized[i, j]))

    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    for i, (true_class, pred_class, confusion_rate) in enumerate(confused_pairs[:5], 1):
        print(f"   {i}. {true_class} -> {pred_class}: {confusion_rate:.2%}")

    print("\n" + "=" * 70)
    print("[COMPLETE] Model evaluation complete!")
    print("=" * 70)
    print(f"\n[SAVED] Reports saved to:")
    print(f"   - Confusion matrix: {cm_path}")
    print(f"   - Classification report: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Trained Model')
    parser.add_argument('--model', type=str, default='mobilenetv2',
                        choices=['baseline', 'mobilenetv2'],
                        help='Model to evaluate (default: mobilenetv2)')
    args = parser.parse_args()
    
    main(args)

