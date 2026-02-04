"""
Radar-based object detection from range-doppler data
"""

import numpy as np
import json
from typing import List, Dict, Any
from pathlib import Path
import os

from .base_detector import BaseDetector, Detection


class RadarDetector(BaseDetector):
    """Object detection from radar range-doppler data"""
    
    def __init__(self, range_threshold: float = 100.0, 
                 min_doppler: float = 0.5):
        """
        Initialize radar detector
        
        Args:
            range_threshold: Maximum detection range (meters)
            min_doppler: Minimum doppler velocity (m/s)
        """
        super().__init__()
        self.range_threshold = range_threshold
        self.min_doppler = min_doppler
        self.radar_data = None
    
    def detect(self, radar_data_path: str) -> List[Detection]:
        """
        Detect moving objects from radar data
        
        Args:
            radar_data_path: Path to radar data file (JSON or text)
            
        Returns:
            List of Detection objects with range, angle, and velocity
        """
        # Load radar data
        self.radar_data = self._load_radar_data(radar_data_path)
        if self.radar_data is None:
            print("Warning: Could not load radar data")
            return []
        
        self.detections = []
        
        # Parse radar detections
        if isinstance(self.radar_data, dict) and 'detections' in self.radar_data:
            detections_data = self.radar_data['detections']
        elif isinstance(self.radar_data, list):
            detections_data = self.radar_data
        else:
            print("Unexpected radar data format")
            return []
        
        # Process each detection
        for detection_data in detections_data:
            detection = self._parse_radar_detection(detection_data)
            if detection:
                self.detections.append(detection)
        
        return self.detections
    
    def _load_radar_data(self, path: str) -> Dict[str, Any]:
        """Load radar data from file"""
        try:
            if path.endswith('.json'):
                with open(path, 'r') as f:
                    return json.load(f)
            elif path.endswith('.npy'):
                return np.load(path, allow_pickle=True).item()
            else:
                # Try JSON fallback
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading radar data: {e}")
            return None
    
    def _parse_radar_detection(self, detection_data: Dict[str, Any]) -> Detection:
        """Parse radar detection from data dictionary"""
        try:
            # Extract radar measurements
            range_m = detection_data.get('range', 0)
            azimuth_deg = detection_data.get('azimuth', 0)
            doppler_velocity = detection_data.get('doppler_velocity', 0)
            rcs = detection_data.get('rcs', -50)  # Radar Cross Section in dBsm
            snr = detection_data.get('snr', 10)   # Signal-to-Noise Ratio in dB
            
            # Filter by range
            if range_m > self.range_threshold:
                return None
            
            # Calculate confidence based on SNR
            confidence = min(0.95, 0.5 + snr / 50.0)  # SNR to confidence mapping
            confidence = max(0.3, confidence)
            
            # Classify object based on RCS and velocity
            rcs_linear = 10 ** (rcs / 10)  # Convert dBsm to linear scale
            
            if abs(doppler_velocity) < self.min_doppler:
                class_name = "stationary_object"
            elif abs(doppler_velocity) > 20:  # High velocity
                class_name = "vehicle"
            elif abs(doppler_velocity) > 5:  # Medium velocity
                class_name = "motorcycle"
            else:
                class_name = "pedestrian"
            
            # Convert to Cartesian coordinates
            azimuth_rad = np.radians(azimuth_deg)
            x = range_m * np.cos(azimuth_rad)
            y = range_m * np.sin(azimuth_rad)
            z = detection_data.get('elevation', 0) * range_m  # Approximation
            
            detection = Detection(
                class_name=class_name,
                confidence=confidence,
                range=float(range_m),
                azimuth=float(azimuth_deg),
                doppler_velocity=float(doppler_velocity),
                rcs=float(rcs),
                snr=float(snr),
                position_cartesian=[float(x), float(y), float(z)],
                position_polar=[float(range_m), float(azimuth_deg)],
                object_id=detection_data.get('id', -1)
            )
            
            return detection
        
        except Exception as e:
            print(f"Error parsing radar detection: {e}")
            return None
    
    def visualize_results(self, detections: List[Detection] = None):
        """
        Print radar detection results
        
        Args:
            detections: List of Detection objects
        """
        if detections is None:
            detections = self.detections
        
        print(f"\n{'='*80}")
        print(f"RADAR DETECTION RESULTS - {len(detections)} objects detected")
        print(f"{'='*80}")
        
        # Print header
        print(f"{'ID':<4} {'Class':<15} {'Range':<10} {'Azimuth':<10} {'Velocity':<12} {'RCS':<10} {'SNR':<8} {'Conf':<8}")
        print(f"{'-'*80}")
        
        # Print detections
        for i, detection in enumerate(detections, 1):
            range_m = detection.attributes.get('range', 0)
            azimuth = detection.attributes.get('azimuth', 0)
            velocity = detection.attributes.get('doppler_velocity', 0)
            rcs = detection.attributes.get('rcs', 0)
            snr = detection.attributes.get('snr', 0)
            
            print(f"{i:<4} {detection.class_name:<15} {range_m:<10.2f} {azimuth:<10.2f} {velocity:<12.2f} {rcs:<10.2f} {snr:<8.2f} {detection.confidence:<8.2%}")
        
        print(f"{'='*80}\n")
    
    def get_moving_objects(self, detections: List[Detection] = None) -> List[Detection]:
        """Get only moving objects (non-zero doppler velocity)"""
        if detections is None:
            detections = self.detections
        
        return [d for d in detections if abs(d.attributes.get('doppler_velocity', 0)) >= self.min_doppler]
    
    def get_stationary_objects(self, detections: List[Detection] = None) -> List[Detection]:
        """Get only stationary objects"""
        if detections is None:
            detections = self.detections
        
        return [d for d in detections if abs(d.attributes.get('doppler_velocity', 0)) < self.min_doppler]
    
    def export_detections(self, format: str = "json") -> Dict[str, Any]:
        """
        Export detections in specified format
        
        Args:
            format: Export format (json, csv, etc.)
            
        Returns:
            Dictionary with exported detections
        """
        export_data = {
            "sensor": "radar",
            "num_detections": len(self.detections),
            "moving_objects": len(self.get_moving_objects()),
            "stationary_objects": len(self.get_stationary_objects()),
            "detections": []
        }
        
        for det in self.detections:
            det_dict = det.to_dict()
            export_data["detections"].append(det_dict)
        
        return export_data
    
    def generate_range_doppler_map(self, num_range_bins: int = 256, 
                                  num_doppler_bins: int = 128) -> np.ndarray:
        """
        Generate a 2D range-doppler map from detections
        
        Args:
            num_range_bins: Number of range bins
            num_doppler_bins: Number of doppler bins
            
        Returns:
            2D numpy array representing range-doppler map
        """
        rdm = np.zeros((num_range_bins, num_doppler_bins))
        
        for detection in self.detections:
            range_m = detection.attributes.get('range', 0)
            velocity = detection.attributes.get('doppler_velocity', 0)
            snr = detection.attributes.get('snr', 0)
            
            # Map to bins
            range_bin = int(range_m / self.range_threshold * num_range_bins)
            vel_bin = int((velocity + 50) / 100 * num_doppler_bins)  # Assuming ±50 m/s range
            
            if 0 <= range_bin < num_range_bins and 0 <= vel_bin < num_doppler_bins:
                rdm[range_bin, vel_bin] = snr
        
        return rdm
