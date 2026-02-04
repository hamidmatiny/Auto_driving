"""
Camera-based object detection using YOLOv8
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
import os

from .base_detector import BaseDetector, Detection

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class CameraDetector(BaseDetector):
    """2D object detection from camera images using YOLOv8"""
    
    def __init__(self, model_name: str = "yolov8n", conf_threshold: float = 0.5):
        """
        Initialize camera detector
        
        Args:
            model_name: YOLOv8 model size (n, s, m, l, x)
            conf_threshold: Confidence threshold for detections
        """
        super().__init__()
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package not installed. Install with: pip install ultralytics")
        
        try:
            self.model = YOLO(f"{model_name}.pt")
        except Exception as e:
            print(f"Warning: Could not load {model_name}.pt. Using mock detector for demo.")
            self.model = None
    
    def detect(self, image_path: str) -> List[Detection]:
        """
        Detect objects in an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of Detection objects with 2D bounding boxes
        """
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")
        else:
            image = image_path
        
        self.image = image
        self.detections = []
        
        # If model not available, generate mock detections for demo
        if self.model is None:
            return self._generate_mock_detections(image)
        
        # Run inference
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        
        # Parse results
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    # Create detection object
                    detection = Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox_2d=[x1, y1, x2, y2],
                        class_id=class_id,
                        bbox_center=[(x1 + x2) // 2, (y1 + y2) // 2],
                        bbox_width=x2 - x1,
                        bbox_height=y2 - y1
                    )
                    self.detections.append(detection)
        
        return self.detections
    
    def _generate_mock_detections(self, image: np.ndarray) -> List[Detection]:
        """Generate mock detections for demo purposes"""
        h, w = image.shape[:2]
        
        # Create some example detections
        mock_detections = [
            {"class": "car", "conf": 0.95, "bbox": [50, 100, 250, 300]},
            {"class": "car", "conf": 0.88, "bbox": [300, 150, 500, 350]},
            {"class": "pedestrian", "conf": 0.92, "bbox": [200, 50, 280, 200]},
            {"class": "truck", "conf": 0.87, "bbox": [600, 200, 800, 400]},
        ]
        
        detections = []
        for mock in mock_detections:
            x1, y1, x2, y2 = mock["bbox"]
            # Make sure bbox is within image bounds
            if x2 < w and y2 < h:
                detection = Detection(
                    class_name=mock["class"],
                    confidence=mock["conf"],
                    bbox_2d=[x1, y1, x2, y2],
                    bbox_center=[(x1 + x2) // 2, (y1 + y2) // 2],
                    bbox_width=x2 - x1,
                    bbox_height=y2 - y1
                )
                detections.append(detection)
        
        return detections
    
    def visualize_results(self, detections: List[Detection] = None, 
                         save_path: str = None):
        """
        Visualize detections on image
        
        Args:
            detections: List of Detection objects (uses last if None)
            save_path: Path to save annotated image
        """
        if detections is None:
            detections = self.detections
        
        if not hasattr(self, 'image'):
            print("No image loaded. Run detect() first.")
            return
        
        image = self.image.copy()
        h, w = image.shape[:2]
        
        # Define color mapping
        color_map = {
            "car": (0, 255, 0),           # Green
            "truck": (255, 0, 0),         # Blue
            "bus": (0, 165, 255),         # Orange
            "pedestrian": (255, 0, 255),  # Magenta
            "bicycle": (0, 255, 255),     # Cyan
            "motorcycle": (128, 0, 128)   # Purple
        }
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection.attributes["bbox_2d"]
            conf = detection.confidence
            class_name = detection.class_name
            
            # Select color based on confidence
            if conf > 0.7:
                color = color_map.get(class_name, (0, 255, 0))
            elif conf > 0.5:
                color = (255, 255, 0)  # Yellow
            else:
                color = (255, 0, 0)  # Red
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - text_size[1] - 4), 
                         (x1 + text_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, image)
            print(f"Annotated image saved to: {save_path}")
        
        # Display
        cv2.imshow("Camera Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def export_detections(self, format: str = "json") -> Dict[str, Any]:
        """
        Export detections in specified format
        
        Args:
            format: Export format (json, xml, yolo)
            
        Returns:
            Dictionary with exported detections
        """
        export_data = {
            "model": self.model_name,
            "confidence_threshold": self.conf_threshold,
            "num_detections": len(self.detections),
            "detections": [d.to_dict() for d in self.detections]
        }
        
        return export_data
