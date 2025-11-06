"""
Image-based Waste Detection and Classification

This script performs detection and classification on static images:
- YOLOv8 for object detection
- Trained classifier for waste type classification

Usage:
    python scripts/08_image_detection.py --image path/to/image.jpg
    python scripts/08_image_detection.py --image path/to/image.jpg --model mobilenetv2
    python scripts/08_image_detection.py --image path/to/image.jpg --output result.jpg

"""

import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.detection import load_yolo_model, detect_objects, crop_objects, classify_images, draw_results

def process_image(image_path, yolo_model, classifier_model, output_path=None):
    """
    Process a single image: detect objects and classify them.

    Arguments:
    image_path -- str or Path, path to input image.
    yolo_model -- YOLO, the YOLOv8 model.
    classifier_model -- tf.keras.Model, the waste classifier.
    output_path -- str or Path or None, path to save annotated image.

    Returns:
    annotated_image -- np.ndarray, the image with annotations.
    results -- list of tuples (bbox, class_name, confidence).
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    print(f"\n[INFO] Image size: {image.shape[1]}x{image.shape[0]}")

    # Detect objects
    print(f"[DETECTING] Detecting objects with YOLOv8...")
    boxes = detect_objects(yolo_model, image, confidence_threshold=YOLO_CONFIDENCE)

    if len(boxes) == 0:
        print("[INFO] No objects detected in image")
        return image, []

    print(f"[OK] Found {len(boxes)} object(s)")

    # Crop detected objects
    cropped_images = crop_objects(image, boxes)

    # Classify cropped images
    print(f"[CLASSIFYING] Classifying detected objects...")
    predictions = classify_images(classifier_model, cropped_images)

    # Draw results on image
    annotated_image = draw_results(image, boxes, predictions)

    # Prepare results
    results = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, class_id = box
        class_name, classification_conf = predictions[i]
        results.append({
            'bbox': (int(x1), int(y1), int(x2), int(y2)),
            'detection_confidence': float(conf),
            'waste_class': class_name,
            'classification_confidence': float(classification_conf)
        })

    # Save output if specified
    if output_path:
        cv2.imwrite(str(output_path), annotated_image)
        print(f"[SAVED] Annotated image saved to: {output_path}")

    return annotated_image, results

def main(args):
    """Main function for image-based detection."""
    print("=" * 70)
    print("WASTE CLASSIFICATION - IMAGE DETECTION")
    print("=" * 70)

    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"\n[ERROR] Image not found: {image_path}")
        return

    # Load classifier model
    model_name = args.model
    model_path = get_model_path(model_name, 'final')

    if not model_path.exists():
        print(f"\n[ERROR] Classifier model not found: {model_path}")
        print(f"\n   Please train the {model_name} model first:")
        if model_name == 'baseline':
            print("   python scripts/03_baseline_training.py")
        else:
            print("   python scripts/04_transfer_learning.py")
        return

    print(f"\n[LOADING] Loading classifier model: {model_name}")
    classifier_model = tf.keras.models.load_model(model_path)
    print(f"   [OK] Classifier loaded from {model_path}")

    # Load YOLO model
    print(f"\n[LOADING] Loading YOLOv8 detection model...")
    yolo_model = load_yolo_model(YOLO_MODEL)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate output filename
        output_path = SCREENSHOTS_DIR / f"detected_{image_path.name}"
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Process image
    print(f"\n[PROCESSING] Processing image: {image_path}")
    print("=" * 70)

    try:
        annotated_image, results = process_image(
            image_path,
            yolo_model,
            classifier_model,
            output_path
        )

        # Print detection results
        print("\n" + "=" * 70)
        print("[RESULTS] Detection Results")
        print("=" * 70)

        if not results:
            print("\n   No waste objects detected in the image")
        else:
            for i, result in enumerate(results, 1):
                print(f"\n   Object {i}:")
                print(f"      - Waste Class: {result['waste_class'].upper()}")
                print(f"      - Classification Confidence: {result['classification_confidence']:.2%}")
                print(f"      - Detection Confidence: {result['detection_confidence']:.2%}")
                print(f"      - Bounding Box: {result['bbox']}")

        print("\n" + "=" * 70)
        print("[SUMMARY]")
        print("=" * 70)
        print(f"   - Total objects detected: {len(results)}")
        print(f"   - Input image: {image_path}")
        print(f"   - Output image: {output_path}")

        # Display image if requested
        if args.display:
            print("\n[DISPLAY] Showing result (press any key to close)...")
            cv2.imshow('Waste Detection Result', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print("\n[OK] Image detection complete!")

    except Exception as e:
        print(f"\n[ERROR] Error processing image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image-based Waste Detection')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, default='mobilenetv2',
                        choices=['baseline', 'mobilenetv2'],
                        help='Classifier model to use (default: mobilenetv2)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save annotated image (default: auto-generated)')
    parser.add_argument('--display', action='store_true',
                        help='Display result image in a window')
    args = parser.parse_args()

    main(args)
