"""YOLO model processor for object detection."""
import cv2
import numpy as np
from typing import Tuple
from ultralytics import YOLO
from app.config import settings
import os


class YOLOProcessor:
    """Service for YOLO model inference."""
    
    def __init__(self):
        """Initialize YOLO model."""
        if not os.path.exists(settings.YOLO_MODEL_PATH):
            raise FileNotFoundError(
                f"YOLO model not found at {settings.YOLO_MODEL_PATH}"
            )
        
        self.model = YOLO(settings.YOLO_MODEL_PATH)
    
    def detect_objects(self, image_bytes: bytes) -> Tuple[bytes, int]:
        """
        Detect objects in an image using YOLO model.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Tuple of (processed_image_bytes, detected_objects_count)
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Run YOLO inference
        results = self.model(image)
        
        # Draw bounding boxes on image
        annotated_image = results[0].plot()
        
        # Count detected objects
        detected_count = len(results[0].boxes)
        
        # Convert annotated image back to bytes
        _, encoded_image = cv2.imencode('.jpg', annotated_image)
        processed_bytes = encoded_image.tobytes()
        
        return processed_bytes, detected_count


# Global instance (lazy initialization)
_yolo_processor: YOLOProcessor = None


def get_yolo_processor() -> YOLOProcessor:
    """Get or create YOLO processor instance."""
    global _yolo_processor
    if _yolo_processor is None:
        _yolo_processor = YOLOProcessor()
    return _yolo_processor






