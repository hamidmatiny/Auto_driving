"""
Example: LiDAR 3D object detection
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lidar_detector import LiDARDetector
import json
import numpy as np


def create_sample_pointcloud():
    """Create a sample point cloud with simulated objects"""
    points = []
    
    # Object 1: Car (large cuboid)
    car_center = np.array([10, 0, 0])
    car_size = np.array([4, 2, 1.5])
    car_points = _create_cuboid_points(car_center, car_size, 500)
    points.extend(car_points)
    
    # Object 2: Pedestrian (tall thin cylinder)
    ped_center = np.array([5, 3, 0])
    ped_size = np.array([0.5, 0.5, 1.8])
    ped_points = _create_cuboid_points(ped_center, ped_size, 200)
    points.extend(ped_points)
    
    # Object 3: Truck (large cuboid)
    truck_center = np.array([15, -2, 0])
    truck_size = np.array([6, 2.5, 2])
    truck_points = _create_cuboid_points(truck_center, truck_size, 800)
    points.extend(truck_points)
    
    # Background noise
    noise_points = np.random.randn(1000, 3) * 5 + np.array([10, 0, 0])
    points.extend(noise_points)
    
    points = np.array(points)
    return points


def _create_cuboid_points(center, size, num_points):
    """Generate random points within a cuboid"""
    points = []
    half_size = size / 2
    
    for _ in range(num_points):
        point = center + np.random.uniform(-half_size, half_size, 3)
        points.append(point)
    
    return points


def create_sample_lidar_json():
    """Create a sample LiDAR detection data in JSON format"""
    pointcloud = create_sample_pointcloud()
    
    data = {
        "points": pointcloud.tolist(),
        "num_points": len(pointcloud),
        "sensor": "velodyne_64",
        "detections": [
            {
                "id": 1,
                "class": "car",
                "center": [10, 0, 0],
                "size": [4, 2, 1.5],
                "confidence": 0.95
            },
            {
                "id": 2,
                "class": "pedestrian",
                "center": [5, 3, 1],
                "size": [0.5, 0.5, 1.8],
                "confidence": 0.92
            },
            {
                "id": 3,
                "class": "truck",
                "center": [15, -2, 0.5],
                "size": [6, 2.5, 2],
                "confidence": 0.88
            }
        ]
    }
    
    sample_path = "examples/sample_data/sample_lidar_data.json"
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    
    with open(sample_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return sample_path


def main():
    """Run LiDAR detection example"""
    print("\n" + "="*80)
    print("LIDAR OBJECT DETECTION EXAMPLE")
    print("="*80 + "\n")
    
    # Create sample data
    print("Creating sample LiDAR data...")
    lidar_path = create_sample_lidar_json()
    print(f"✓ Sample LiDAR data created at: {lidar_path}\n")
    
    # Initialize detector
    print("Initializing LiDAR Detector...")
    detector = LiDARDetector(voxel_size=0.05, min_points=5)
    print("✓ Detector initialized\n")
    
    # Run detection
    print("Running 3D object detection on point cloud...")
    detections = detector.detect(lidar_path)
    print(f"✓ Detection complete: {len(detections)} objects found\n")
    
    # Print results
    detector.print_detections(detections)
    
    # Export results
    print("Exporting detection results...")
    export_data = detector.export_detections(format="json")
    
    output_file = "output/lidar_detections.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"✓ Results saved to: {output_file}\n")
    
    # Show structure
    print("Detection Structure Example:")
    if detections:
        print(json.dumps(detections[0].to_dict(), indent=2))
    
    print("\n" + "="*80)
    print("LiDAR Detection Statistics:")
    print("="*80)
    print(f"Objects detected: {len(detections)}")
    
    for det in detections:
        print(f"\n[{det.attributes.get('cluster_id')}] {det.class_name.upper()}")
        print(f"  Confidence: {det.confidence:.2%}")
        bbox_3d = det.attributes.get('bbox_3d', {})
        if 'center' in bbox_3d:
            center = bbox_3d['center']
            size = bbox_3d['size']
            print(f"  Position: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
            print(f"  Size: ({size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f})")
            print(f"  Distance: {det.attributes.get('distance', 0):.2f}m")
            print(f"  Points: {det.attributes.get('num_points', 0)}")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
