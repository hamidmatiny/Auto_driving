"""
Base detector class for all sensor modalities
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import numpy as np


class Detection:
    """Represents a single object detection"""
    
    def __init__(self, class_name: str, confidence: float, **kwargs):
        self.class_name = class_name
        self.confidence = confidence
        self.attributes = kwargs
    
    def __repr__(self):
        return f"Detection(class={self.class_name}, confidence={self.confidence:.2f})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary"""
        result = {
            "class": self.class_name,
            "confidence": float(self.confidence)
        }
        result.update(self.attributes)
        return result


class BaseDetector(ABC):
    """Abstract base class for all detectors"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.detections = []
    
    @abstractmethod
    def detect(self, data_input: Any) -> List[Detection]:
        """
        Perform object detection on input data
        
        Args:
            data_input: Input data (image, point cloud, radar data, etc.)
            
        Returns:
            List of Detection objects
        """
        pass
    
    @abstractmethod
    def visualize_results(self, detections: List[Detection], data_input: Any = None):
        """
        Visualize detection results
        
        Args:
            detections: List of Detection objects
            data_input: Original input data for context
        """
        pass
    
    def get_detections(self) -> List[Detection]:
        """Get last detection results"""
        return self.detections
    
    def filter_by_confidence(self, detections: List[Detection], 
                            threshold: float) -> List[Detection]:
        """Filter detections by confidence threshold"""
        return [d for d in detections if d.confidence >= threshold]
    
    def filter_by_class(self, detections: List[Detection], 
                       classes: List[str]) -> List[Detection]:
        """Filter detections by class names"""
        return [d for d in detections if d.class_name in classes]
    
    def print_detections(self, detections: List[Detection] = None):
        """Print detection results in human-readable format"""
        if detections is None:
            detections = self.detections
        
        print(f"\n{'='*60}")
        print(f"Detection Results - {len(detections)} objects found")
        print(f"{'='*60}")
        
        for i, detection in enumerate(detections, 1):
            print(f"\n[{i}] {detection.class_name.upper()}")
            print(f"    Confidence: {detection.confidence:.2%}")
            for key, value in detection.attributes.items():
                if isinstance(value, (float, np.floating)):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
        
        print(f"\n{'='*60}\n")
