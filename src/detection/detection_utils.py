"""
Object detection utilities using YOLOv8.

This module provides functions for:
- Loading the YOLOv8 model
- Performing object detection on images
- Cropping detected objects from images

"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from ..config import *


def load_yolo_model(model_path=YOLO_MODEL):
    """
    Loads the YOLOv8 model from the specified path.

    Arguments:
    model_path -- str, path to the YOLOv8 model file (e.g., 'yolov8n.pt').

    Returns:
    model -- YOLO, the loaded YOLOv8 model object.
    """
    print(f"Loading YOLOv8 model from {model_path}...")
    
    # For PyTorch 2.6+, we need to handle the weights_only security feature
    # YOLOv8 models are trusted, so we'll temporarily allow unsafe loading
    try:
        # Save the original torch.load function
        original_load = torch.load
        
        # Create a wrapper that uses weights_only=False for trusted YOLOv8 models
        def safe_yolo_load(*args, **kwargs):
            # Only override if weights_only is not explicitly set
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        # Temporarily replace torch.load
        torch.load = safe_yolo_load
        
        try:
            model = YOLO(model_path)
        finally:
            # Restore the original torch.load
            torch.load = original_load
            
    except Exception as e:
        # Fallback: try with the original approach
        print(f"   Note: Using fallback loading method")
        model = YOLO(model_path)

    print("[OK] YOLOv8 model loaded.")
    return model


def detect_objects(model, image, confidence_threshold=YOLO_CONFIDENCE):
    """
    Performs object detection on a single image using the YOLOv8 model.

    Arguments:
    model -- YOLO, the loaded YOLOv8 model.
    image -- np.ndarray, the input image in BGR format.
    confidence_threshold -- float, the minimum confidence for a detection.

    Returns:
    boxes -- list, a list of bounding boxes for detected objects.
             Each box is in the format [x1, y1, x2, y2, confidence, class_id].
    """
    results = model(image, conf=confidence_threshold, verbose=False)
    return results[0].boxes.data.cpu().numpy()


def crop_objects(image, boxes):
    """
    Crops the detected objects from the image based on bounding boxes.

    Arguments:
    image -- np.ndarray, the original image.
    boxes -- list, a list of bounding boxes from `detect_objects`.

    Returns:
    cropped_images -- list, a list of cropped image arrays (np.ndarray).
    """
    cropped_images = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cropped_img = image[y1:y2, x1:x2]
        cropped_images.append(cropped_img)
    return cropped_images

