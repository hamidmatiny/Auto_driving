"""
Multi-Modal Object Detection for Autonomous Driving
Package for sensor fusion and object detection
"""

from .base_detector import BaseDetector, Detection
from .camera_detector import CameraDetector
from .lidar_detector import LiDARDetector
from .radar_detector import RadarDetector
from .fusion_engine import FusionEngine

__version__ = "1.0.0"
__author__ = "Autonomous Driving Team"

__all__ = [
    "BaseDetector",
    "Detection",
    "CameraDetector",
    "LiDARDetector",
    "RadarDetector",
    "FusionEngine"
]
