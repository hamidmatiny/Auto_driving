"""
LiDAR-based 3D object detection from point clouds
"""

import numpy as np
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
import os

from .base_detector import BaseDetector, Detection


class LiDARDetector(BaseDetector):
    """3D object detection from LiDAR point clouds"""
    
    def __init__(self, voxel_size: float = 0.05, min_points: int = 5):
        """
        Initialize LiDAR detector
        
        Args:
            voxel_size: Voxel size for grid-based clustering (meters)
            min_points: Minimum points to form a cluster
        """
        super().__init__()
        self.voxel_size = voxel_size
        self.min_points = min_points
        self.pointcloud = None
    
    def detect(self, pointcloud_path: str) -> List[Detection]:
        """
        Detect objects in a point cloud
        
        Args:
            pointcloud_path: Path to point cloud file (PCD, PLY, or JSON)
            
        Returns:
            List of Detection objects with 3D bounding boxes
        """
        # Load point cloud
        self.pointcloud = self._load_pointcloud(pointcloud_path)
        if self.pointcloud is None or len(self.pointcloud) == 0:
            print("Warning: Empty or invalid point cloud")
            return []
        
        self.detections = []
        
        # If using JSON (mock data), parse directly
        if pointcloud_path.endswith('.json'):
            return self._parse_json_detections(pointcloud_path)
        
        # Perform clustering
        clusters = self._cluster_points(self.pointcloud)
        
        if not clusters:
            print("No clusters found in point cloud")
            return []
        
        # Create detections from clusters
        for i, cluster in enumerate(clusters):
            if len(cluster) < self.min_points:
                continue
            
            detection = self._cluster_to_detection(cluster, i)
            if detection:
                self.detections.append(detection)
        
        return self.detections
    
    def _load_pointcloud(self, path: str) -> np.ndarray:
        """Load point cloud from various formats"""
        try:
            if path.endswith('.json'):
                with open(path, 'r') as f:
                    data = json.load(f)
                if 'points' in data:
                    return np.array(data['points'])
                return None
            
            elif path.endswith('.npy'):
                return np.load(path)
            
            else:
                # Try to load as generic text format
                return np.loadtxt(path)
        except Exception as e:
            print(f"Error loading point cloud: {e}")
            return None
    
    def _parse_json_detections(self, path: str) -> List[Detection]:
        """Parse detections from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        detections = []
        if 'detections' in data:
            for det_data in data['detections']:
                detection = Detection(
                    class_name=det_data.get('class', 'unknown'),
                    confidence=det_data.get('confidence', 0.8),
                    bbox_3d=det_data.get('bbox_3d', {}),
                    center=det_data.get('center', [0, 0, 0]),
                    size=det_data.get('size', [1, 1, 1]),
                    distance=det_data.get('distance', 0)
                )
                detections.append(detection)
        
        return detections
    
    def _cluster_points(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Cluster points using simple grid-based clustering
        
        Args:
            points: Nx3 array of point coordinates
            
        Returns:
            List of point clusters
        """
        if len(points) == 0:
            return []
        
        # Create voxel grid
        grid_size = 1.0 / self.voxel_size
        voxel_indices = (points / self.voxel_size).astype(np.int32)
        
        # Group points by voxel
        voxel_dict = {}
        for i, voxel_idx in enumerate(voxel_indices):
            key = tuple(voxel_idx)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(i)
        
        # Create clusters from adjacent voxels
        clusters = []
        visited = set()
        
        for voxel_key in voxel_dict.keys():
            if voxel_key in visited:
                continue
            
            # BFS to find connected voxels
            cluster_indices = set()
            queue = [voxel_key]
            
            while queue:
                current_key = queue.pop(0)
                if current_key in visited:
                    continue
                
                visited.add(current_key)
                cluster_indices.update(voxel_dict[current_key])
                
                # Check neighbors
                cx, cy, cz = current_key
                for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                    neighbor_key = (cx + dx, cy + dy, cz + dz)
                    if neighbor_key in voxel_dict and neighbor_key not in visited:
                        queue.append(neighbor_key)
            
            if len(cluster_indices) >= self.min_points:
                clusters.append(points[list(cluster_indices)])
        
        return clusters
    
    def _cluster_to_detection(self, cluster: np.ndarray, cluster_id: int) -> Detection:
        """Convert a point cluster to a Detection object with 3D bbox"""
        if len(cluster) == 0:
            return None
        
        # Calculate bounding box
        min_point = cluster.min(axis=0)
        max_point = cluster.max(axis=0)
        center = (min_point + max_point) / 2
        size = max_point - min_point
        
        # Calculate metrics
        distance = np.linalg.norm(center)
        density = len(cluster) / (size[0] * size[1] * size[2] + 1e-6)
        
        # Classify based on size heuristics
        if size[0] > 1.5 and size[1] > 1.5:  # Large in x-y
            class_name = "car"
            confidence = 0.9 if density > 100 else 0.7
        elif size[2] > 1.5:  # Tall object
            class_name = "pedestrian"
            confidence = 0.85
        elif size[0] > 2.5:  # Very long object
            class_name = "truck"
            confidence = 0.88
        else:
            class_name = "object"
            confidence = 0.6
        
        detection = Detection(
            class_name=class_name,
            confidence=confidence,
            bbox_3d={
                "center": center.tolist(),
                "size": size.tolist(),
                "min": min_point.tolist(),
                "max": max_point.tolist()
            },
            distance=float(distance),
            num_points=len(cluster),
            density=float(density),
            cluster_id=cluster_id
        )
        
        return detection
    
    def visualize_results(self, detections: List[Detection] = None, 
                         save_path: str = None):
        """
        Visualize 3D detections (text representation)
        
        Args:
            detections: List of Detection objects
            save_path: Path to save visualization
        """
        if detections is None:
            detections = self.detections
        
        if not self.pointcloud is not None:
            print("No point cloud loaded. Run detect() first.")
            return
        
        try:
            import open3d as o3d
            
            # Create visualization
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            
            # Add point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.pointcloud)
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
            vis.add_geometry(pcd)
            
            # Add bounding boxes
            for detection in detections:
                bbox_3d = detection.attributes.get('bbox_3d', {})
                if bbox_3d and 'center' in bbox_3d:
                    center = bbox_3d['center']
                    size = bbox_3d['size']
                    
                    # Create bbox
                    bbox = o3d.geometry.AxisAlignedBoundingBox(
                        min_bound=np.array(bbox_3d['min']),
                        max_bound=np.array(bbox_3d['max'])
                    )
                    bbox.color = (0, 1, 0)  # Green
                    vis.add_geometry(bbox)
            
            vis.run()
            vis.destroy_window()
            
        except ImportError:
            print("Open3D not installed. Falling back to text visualization.")
            self.print_detections(detections)
    
    def export_detections(self, format: str = "json") -> Dict[str, Any]:
        """
        Export detections in specified format
        
        Args:
            format: Export format (json, xml, etc.)
            
        Returns:
            Dictionary with exported detections
        """
        export_data = {
            "sensor": "lidar",
            "num_detections": len(self.detections),
            "voxel_size": self.voxel_size,
            "detections": []
        }
        
        for det in self.detections:
            det_dict = det.to_dict()
            export_data["detections"].append(det_dict)
        
        return export_data
