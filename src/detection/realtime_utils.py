"""
Real-time detection and visualization utilities.

This module provides functions for:
- Classifying cropped images
- Drawing bounding boxes and labels on frames
- Managing real-time detection workflow

"""

import cv2
import numpy as np
import tensorflow as tf

from ..config import *


def classify_images(classifier_model, images):
    """
    Classifies a batch of images using the provided classifier model.

    Arguments:
    classifier_model -- tf.keras.Model, the trained waste classifier.
    images -- list, a list of image arrays (np.ndarray) to classify.

    Returns:
    predictions -- list, a list of tuples (class_name, confidence).
    """
    if not images:
        return []

    # Preprocess images for the classifier
    processed_images = []
    for img in images:
        img_resized = tf.image.resize(img, IMG_SIZE)
        processed_images.append(img_resized)
    
    batch = tf.stack(processed_images)
    
    # Make predictions
    preds = classifier_model.predict(batch, verbose=False)
    
    # Decode predictions
    results = []
    for pred in preds:
        class_index = np.argmax(pred)
        confidence = pred[class_index]
        class_name = CLASS_NAMES[class_index]
        results.append((class_name, confidence))
        
    return results


def draw_results(frame, boxes, predictions):
    """
    Draws bounding boxes and classification results on the frame.

    Arguments:
    frame -- np.ndarray, the camera frame to draw on.
    boxes -- list, the list of bounding boxes.
    predictions -- list, the list of predictions from `classify_images`.

    Returns:
    frame -- np.ndarray, the frame with visualizations.
    """
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        class_name, confidence = predictions[i]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), BOX_THICKNESS)
        
        # Create label text
        label = f"{class_name.upper()}: {confidence:.2f}"
        
        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)
        
    return frame

