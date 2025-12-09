"""YOLO model processor for object detection."""
import cv2
import numpy as np
from typing import Tuple
from ultralytics import YOLO
from app.config import settings
import os
import httpx


class YOLOProcessor:
    """Service for YOLO model inference."""
    
    def __init__(self):
        """Initialize YOLO model."""
        if not os.path.exists(settings.YOLO_MODEL_PATH):
            if settings.YOLO_MODEL_URL:
                print(f"Model not found at {settings.YOLO_MODEL_PATH}. Downloading from {settings.YOLO_MODEL_URL}...")
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(settings.YOLO_MODEL_PATH), exist_ok=True)
                
                try:
                    with httpx.Client(follow_redirects=True) as client:
                        with client.stream("GET", settings.YOLO_MODEL_URL) as response:
                            response.raise_for_status()
                            with open(settings.YOLO_MODEL_PATH, "wb") as f:
                                for chunk in response.iter_bytes():
                                    f.write(chunk)
                    print(f"Model downloaded successfully to {settings.YOLO_MODEL_PATH}")
                except Exception as e:
                    # Clean up partial download if it failed
                    if os.path.exists(settings.YOLO_MODEL_PATH):
                        os.remove(settings.YOLO_MODEL_PATH)
                    raise RuntimeError(f"Failed to download model from {settings.YOLO_MODEL_URL}: {str(e)}")
            else:
                raise FileNotFoundError(
                    f"YOLO model not found at {settings.YOLO_MODEL_PATH}"
                )
        
        self.model = YOLO(settings.YOLO_MODEL_PATH)
    
    def detect_objects(self, image_bytes: bytes, conf_threshold: float = 0.25) -> Tuple[bytes, int]:
        """
        Detect objects in an image using YOLO model.
        
        Args:
            image_bytes: Image data as bytes
            conf_threshold: Confidence threshold for detections (default: 0.25)
            
        Returns:
            Tuple of (processed_image_bytes, detected_objects_count)
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Resize image so smaller side is 640px
        h, w = image.shape[:2]
        if min(h, w) != 640:
            scale = 640 / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # Run YOLO inference
        results = self.model(image, conf=conf_threshold)
        result = results[0]
        
        # Red color in BGR format: (0, 0, 255)
        color = (0, 0, 255)
        line_width = 5
        
        annotated_image = image.copy()
        
        # Draw bounding boxes with custom color
        if len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, line_width)
        
        # Count detected objects
        detected_count = len(result.boxes)
        
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
