"""
Sensor fusion engine combining camera, LiDAR, and radar detections
"""

import numpy as np
import json
import cv2
from typing import List, Dict, Any, Tuple
from pathlib import Path
import os
from datetime import datetime

from .camera_detector import CameraDetector
from .lidar_detector import LiDARDetector
from .radar_detector import RadarDetector
from .base_detector import Detection


class FusionEngine:
    """Multi-modal sensor fusion for robust object detection"""
    
    def __init__(self, camera_weight: float = 0.4, 
                 lidar_weight: float = 0.4,
                 radar_weight: float = 0.2):
        """
        Initialize fusion engine
        
        Args:
            camera_weight: Weight for camera detections
            lidar_weight: Weight for LiDAR detections
            radar_weight: Weight for radar detections
        """
        self.camera_detector = CameraDetector()
        self.lidar_detector = LiDARDetector()
        self.radar_detector = RadarDetector()
        
        self.camera_weight = camera_weight
        self.lidar_weight = lidar_weight
        self.radar_weight = radar_weight
        
        self.fused_detections = []
        self.camera_detections = []
        self.lidar_detections = []
        self.radar_detections = []
    
    def process(self, camera_data: str = None, 
                lidar_data: str = None,
                radar_data: str = None) -> Dict[str, Any]:
        """
        Process multi-modal sensor data and fuse detections
        
        Args:
            camera_data: Path to camera image or image data
            lidar_data: Path to LiDAR point cloud
            radar_data: Path to radar data
            
        Returns:
            Dictionary with fused detections and metadata
        """
        # Run individual detectors
        if camera_data:
            try:
                self.camera_detections = self.camera_detector.detect(camera_data)
                print(f"✓ Camera: {len(self.camera_detections)} detections")
            except Exception as e:
                print(f"✗ Camera detection failed: {e}")
                self.camera_detections = []
        
        if lidar_data:
            try:
                self.lidar_detections = self.lidar_detector.detect(lidar_data)
                print(f"✓ LiDAR: {len(self.lidar_detections)} detections")
            except Exception as e:
                print(f"✗ LiDAR detection failed: {e}")
                self.lidar_detections = []
        
        if radar_data:
            try:
                self.radar_detections = self.radar_detector.detect(radar_data)
                print(f"✓ Radar: {len(self.radar_detections)} detections")
            except Exception as e:
                print(f"✗ Radar detection failed: {e}")
                self.radar_detections = []
        
        # Fuse detections
        self.fused_detections = self._fuse_detections()
        print(f"✓ Fusion: {len(self.fused_detections)} fused objects\n")
        
        return self._generate_output()
    
    def _fuse_detections(self) -> List[Dict[str, Any]]:
        """
        Fuse detections from multiple sensors
        
        Returns:
            List of fused detection objects
        """
        fused_list = []
        used_camera = set()
        used_lidar = set()
        used_radar = set()
        object_id = 0
        
        # Match LiDAR and camera detections (3D to 2D)
        for lidar_idx, lidar_det in enumerate(self.lidar_detections):
            best_camera_match = None
            best_camera_score = 0
            
            for cam_idx, cam_det in enumerate(self.camera_detections):
                if cam_idx in used_camera:
                    continue
                
                # Simple matching: check if 3D point projects to 2D bbox
                score = self._match_3d_to_2d(lidar_det, cam_det)
                if score > best_camera_score:
                    best_camera_score = score
                    best_camera_match = cam_idx
            
            # Try to match with radar
            best_radar_match = None
            best_radar_score = 0
            
            for radar_idx, radar_det in enumerate(self.radar_detections):
                if radar_idx in used_radar:
                    continue
                
                score = self._match_3d_to_radar(lidar_det, radar_det)
                if score > best_radar_score:
                    best_radar_score = score
                    best_radar_match = radar_idx
            
            # Create fused detection
            fused_det = {
                "id": object_id,
                "class": self._fuse_class(
                    lidar_det.class_name if lidar_det else "unknown",
                    self.camera_detections[best_camera_match].class_name if best_camera_match is not None else "unknown",
                    self.radar_detections[best_radar_match].class_name if best_radar_match is not None else "unknown"
                ),
                "position_3d": lidar_det.attributes.get("bbox_3d", {}).get("center", [0, 0, 0]),
                "size_3d": lidar_det.attributes.get("bbox_3d", {}).get("size", [1, 1, 1]),
                "confidence": 0,
                "sensor_contributions": {}
            }
            
            # Add sensor contributions
            if lidar_idx not in used_lidar:
                fused_det["sensor_contributions"]["lidar"] = {
                    "confidence": lidar_det.confidence,
                    "distance": lidar_det.attributes.get("distance", 0)
                }
                used_lidar.add(lidar_idx)
            
            if best_camera_match is not None:
                cam_det = self.camera_detections[best_camera_match]
                fused_det["sensor_contributions"]["camera"] = {
                    "confidence": cam_det.confidence,
                    "bbox_2d": cam_det.attributes.get("bbox_2d", [])
                }
                used_camera.add(best_camera_match)
            
            if best_radar_match is not None:
                rad_det = self.radar_detections[best_radar_match]
                fused_det["sensor_contributions"]["radar"] = {
                    "confidence": rad_det.confidence,
                    "velocity": rad_det.attributes.get("doppler_velocity", 0),
                    "range": rad_det.attributes.get("range", 0)
                }
                used_radar.add(best_radar_match)
            
            # Calculate fused confidence
            fused_det["confidence"] = self._fuse_confidence(fused_det["sensor_contributions"])
            
            fused_list.append(fused_det)
            object_id += 1
        
        # Add unmatched camera detections
        for cam_idx, cam_det in enumerate(self.camera_detections):
            if cam_idx not in used_camera:
                fused_det = {
                    "id": object_id,
                    "class": cam_det.class_name,
                    "position_3d": [0, 0, 0],
                    "bbox_2d": cam_det.attributes.get("bbox_2d", []),
                    "confidence": cam_det.confidence,
                    "sensor_contributions": {
                        "camera": {
                            "confidence": cam_det.confidence,
                            "bbox_2d": cam_det.attributes.get("bbox_2d", [])
                        }
                    }
                }
                fused_list.append(fused_det)
                object_id += 1
        
        # Add unmatched radar detections
        for rad_idx, rad_det in enumerate(self.radar_detections):
            if rad_idx not in used_radar:
                fused_det = {
                    "id": object_id,
                    "class": rad_det.class_name,
                    "position_3d": rad_det.attributes.get("position_cartesian", [0, 0, 0]),
                    "confidence": rad_det.confidence,
                    "sensor_contributions": {
                        "radar": {
                            "confidence": rad_det.confidence,
                            "velocity": rad_det.attributes.get("doppler_velocity", 0),
                            "range": rad_det.attributes.get("range", 0)
                        }
                    }
                }
                fused_list.append(fused_det)
                object_id += 1
        
        return fused_list
    
    def _match_3d_to_2d(self, lidar_det: Detection, camera_det: Detection) -> float:
        """Calculate matching score between 3D and 2D detection"""
        # Simple heuristic: check if class names are similar
        lidar_class = lidar_det.class_name.lower()
        camera_class = camera_det.class_name.lower()
        
        # Map class names for comparison
        class_map = {
            "car": "car",
            "truck": "truck",
            "bus": "bus",
            "motorcycle": "motorcycle",
            "bicycle": "bicycle",
            "pedestrian": "pedestrian",
            "person": "pedestrian",
            "auto": "car"
        }
        
        lidar_class_mapped = class_map.get(lidar_class, lidar_class)
        camera_class_mapped = class_map.get(camera_class, camera_class)
        
        if lidar_class_mapped == camera_class_mapped:
            return 0.9
        elif (lidar_class_mapped in ["car", "truck", "bus"] and 
              camera_class_mapped in ["car", "truck", "bus"]):
            return 0.5
        else:
            return 0.1
    
    def _match_3d_to_radar(self, lidar_det: Detection, radar_det: Detection) -> float:
        """Calculate matching score between 3D LiDAR and radar"""
        # Compare position and class
        lidar_pos = lidar_det.attributes.get("bbox_3d", {}).get("center", [0, 0, 0])
        radar_pos = radar_det.attributes.get("position_cartesian", [0, 0, 0])
        
        # Calculate distance
        distance = np.linalg.norm(np.array(lidar_pos) - np.array(radar_pos))
        
        # Matching score based on distance
        if distance < 1.0:
            return 0.9
        elif distance < 3.0:
            return 0.6
        else:
            return 0.2
    
    def _fuse_class(self, lidar_class: str, camera_class: str, radar_class: str) -> str:
        """Fuse class predictions from multiple sensors"""
        classes = [lidar_class, camera_class, radar_class]
        
        # Vote-based fusion
        class_votes = {}
        for cls in classes:
            if cls and cls != "unknown":
                class_votes[cls] = class_votes.get(cls, 0) + 1
        
        if class_votes:
            return max(class_votes, key=class_votes.get)
        else:
            return "unknown"
    
    def _fuse_confidence(self, sensor_contributions: Dict[str, Any]) -> float:
        """Calculate fused confidence score"""
        weights_sum = 0
        confidence_sum = 0
        
        if "camera" in sensor_contributions:
            conf = sensor_contributions["camera"]["confidence"]
            confidence_sum += conf * self.camera_weight
            weights_sum += self.camera_weight
        
        if "lidar" in sensor_contributions:
            conf = sensor_contributions["lidar"]["confidence"]
            confidence_sum += conf * self.lidar_weight
            weights_sum += self.lidar_weight
        
        if "radar" in sensor_contributions:
            conf = sensor_contributions["radar"]["confidence"]
            confidence_sum += conf * self.radar_weight
            weights_sum += self.radar_weight
        
        if weights_sum > 0:
            return confidence_sum / weights_sum
        else:
            return 0.0
    
    def _generate_output(self) -> Dict[str, Any]:
        """Generate final output dictionary"""
        return {
            "timestamp": datetime.now().isoformat(),
            "num_fused_objects": len(self.fused_detections),
            "sensor_detections": {
                "camera": len(self.camera_detections),
                "lidar": len(self.lidar_detections),
                "radar": len(self.radar_detections)
            },
            "fused_detections": self.fused_detections
        }
    
    def visualize_fusion_results(self, results: Dict[str, Any] = None, 
                                save_path: str = None):
        """Visualize fusion results"""
        if results is None:
            results = self._generate_output()
        
        print(f"\n{'='*80}")
        print(f"SENSOR FUSION RESULTS")
        print(f"{'='*80}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Total Fused Objects: {results['num_fused_objects']}")
        print(f"\nSensor Contributions:")
        print(f"  - Camera:  {results['sensor_detections']['camera']:3d} detections")
        print(f"  - LiDAR:   {results['sensor_detections']['lidar']:3d} detections")
        print(f"  - Radar:   {results['sensor_detections']['radar']:3d} detections")
        print(f"\n{'ID':<4} {'Class':<15} {'Position (3D)':<25} {'Conf':<8} {'Sensors':<20}")
        print(f"{'-'*80}")
        
        for det in results['fused_detections']:
            pos = det['position_3d']
            pos_str = f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
            sensors = ", ".join(det['sensor_contributions'].keys())
            
            print(f"{det['id']:<4} {det['class']:<15} {pos_str:<25} {det['confidence']:<8.2%} {sensors:<20}")
        
        print(f"{'='*80}\n")
    
    def export_results(self, format: str = "json", output_path: str = None) -> str:
        """
        Export fusion results
        
        Args:
            format: Export format (json, xml, yolo)
            output_path: Path to save results
            
        Returns:
            Exported data as string
        """
        results = self._generate_output()
        
        if format == "json":
            export_str = json.dumps(results, indent=2)
        else:
            export_str = str(results)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(export_str)
            print(f"Results exported to: {output_path}")
        
        return export_str
