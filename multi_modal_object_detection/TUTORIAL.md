# Complete Tutorial & Practical Guide

## Getting Started Tutorial

### Part 1: Environment Setup (5 minutes)

#### Step 1: Install Python 3.8+

Check your Python version:
```bash
python3 --version
```

#### Step 2: Create Virtual Environment

```bash
cd ..

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

#### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all packages
pip install -r requirements.txt
```

This will install:
- `torch` & `torchvision` - Deep learning frameworks
- `ultralytics` - YOLOv8 implementation
- `opencv-python` - Image processing
- `open3d` - 3D point cloud processing
- And other supporting libraries

---

### Part 2: Running Your First Detection (10 minutes)

#### Camera Detection Example

```bash
# Run the camera detection example
python examples/example_camera.py
```

**What happens:**
1. Creates a sample image
2. Initializes YOLOv8 detector
3. Runs detection on the image
4. Displays results with confidence scores
5. Saves annotated image to `output/annotated_camera_image.jpg`

**Expected output:**
```
============================================================
Detection Results - 4 objects found
============================================================

[1] CAR
    Confidence: 95.00%
    bbox_2d: [50, 100, 250, 300]
    ...
```

---

### Part 3: Understanding the Code

#### Understanding Detection Objects

Create a Python script `my_first_detection.py`:

```python
#!/usr/bin/env python3
"""My first multi-modal detection script"""

from src.camera_detector import CameraDetector
from src.lidar_detector import LiDARDetector
from src.radar_detector import RadarDetector
import json

def main():
    print("Multi-Modal Object Detection Tutorial\n")
    
    # 1. CAMERA DETECTION
    print("="*50)
    print("1. CAMERA DETECTION")
    print("="*50)
    
    camera_detector = CameraDetector(
        model_name="yolov8n",      # Use small model
        conf_threshold=0.5          # 50% confidence minimum
    )
    
    # In real usage: camera_detector.detect("path/to/real_image.jpg")
    # For demo, we'll use the example
    from examples.example_camera import create_sample_image
    image_path = create_sample_image()
    
    camera_results = camera_detector.detect(image_path)
    print(f"Found {len(camera_results)} objects\n")
    
    for i, det in enumerate(camera_results, 1):
        print(f"Object {i}:")
        print(f"  Class: {det.class_name}")
        print(f"  Confidence: {det.confidence:.1%}")
        print(f"  Bounding box: {det.attributes['bbox_2d']}\n")
    
    # 2. LIDAR DETECTION
    print("="*50)
    print("2. LIDAR DETECTION (3D)")
    print("="*50)
    
    lidar_detector = LiDARDetector(
        voxel_size=0.05,           # 5cm voxels
        min_points=5               # Minimum 5 points per cluster
    )
    
    from examples.example_lidar import create_sample_lidar_json
    lidar_path = create_sample_lidar_json()
    
    lidar_results = lidar_detector.detect(lidar_path)
    print(f"Found {len(lidar_results)} objects\n")
    
    for i, det in enumerate(lidar_results, 1):
        center = det.attributes['bbox_3d']['center']
        size = det.attributes['bbox_3d']['size']
        print(f"Object {i}:")
        print(f"  Class: {det.class_name}")
        print(f"  Position: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
        print(f"  Size: {size}")
        print(f"  Distance: {det.attributes['distance']:.1f}m\n")
    
    # 3. RADAR DETECTION
    print("="*50)
    print("3. RADAR DETECTION (Velocity)")
    print("="*50)
    
    radar_detector = RadarDetector(
        range_threshold=100.0,      # Up to 100m
        min_doppler=0.5             # Minimum 0.5 m/s
    )
    
    from examples.example_radar import create_sample_radar_data
    radar_path = create_sample_radar_data()
    
    radar_results = radar_detector.detect(radar_path)
    print(f"Found {len(radar_results)} objects\n")
    
    moving = radar_detector.get_moving_objects()
    stationary = radar_detector.get_stationary_objects()
    print(f"Moving: {len(moving)} | Stationary: {len(stationary)}\n")
    
    for i, det in enumerate(radar_results, 1):
        rng = det.attributes['range']
        vel = det.attributes['doppler_velocity']
        print(f"Object {i}:")
        print(f"  Class: {det.class_name}")
        print(f"  Range: {rng:.1f}m")
        print(f"  Velocity: {vel:.1f}m/s\n")
    
    # 4. SENSOR FUSION
    print("="*50)
    print("4. SENSOR FUSION")
    print("="*50)
    
    from src.fusion_engine import FusionEngine
    
    fusion = FusionEngine(
        camera_weight=0.4,
        lidar_weight=0.4,
        radar_weight=0.2
    )
    
    fusion_results = fusion.process(
        camera_data=image_path,
        lidar_data=lidar_path,
        radar_data=radar_path
    )
    
    print(f"Fused into {fusion_results['num_fused_objects']} objects\n")
    
    for det in fusion_results['fused_detections']:
        print(f"Object {det['id']}:")
        print(f"  Class: {det['class']}")
        print(f"  Confidence: {det['confidence']:.1%}")
        print(f"  Sensors: {', '.join(det['sensor_contributions'].keys())}\n")
    
    # 5. EXPORT RESULTS
    print("="*50)
    print("5. EXPORT RESULTS")
    print("="*50)
    
    import os
    os.makedirs("output", exist_ok=True)
    
    # Export to JSON
    with open("output/my_results.json", "w") as f:
        json.dump(fusion_results, f, indent=2)
    
    print("✓ Results saved to: output/my_results.json")

if __name__ == "__main__":
    main()
```

Run this tutorial:
```bash
python my_first_detection.py
```

---

## Practical Scenarios

### Scenario 1: Processing a Real Image

```python
from src.camera_detector import CameraDetector

def detect_cars_in_image(image_path):
    """Detect only cars in an image"""
    detector = CameraDetector(model_name="yolov8m")
    
    # Run detection
    all_detections = detector.detect(image_path)
    
    # Filter for cars only
    cars = detector.filter_by_class(all_detections, ["car", "truck", "bus"])
    
    # Filter by confidence
    confident_cars = detector.filter_by_confidence(cars, 0.7)
    
    # Visualize
    detector.visualize_results(confident_cars, save_path="cars_only.jpg")
    
    return confident_cars

# Usage
results = detect_cars_in_image("street_scene.jpg")
print(f"Detected {len(results)} vehicles")
```

### Scenario 2: Processing Multiple Frames

```python
from src.camera_detector import CameraDetector
import os

def batch_process_images(folder_path):
    """Process all images in a folder"""
    detector = CameraDetector()
    
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
    
    for image_file in images:
        image_path = os.path.join(folder_path, image_file)
        results = detector.detect(image_path)
        
        print(f"{image_file}: {len(results)} objects")
        
        # Save annotated image
        output_path = f"output/annotated_{image_file}"
        detector.visualize_results(results, save_path=output_path)

# Usage
batch_process_images("images_folder/")
```

### Scenario 3: Real-Time Stream Processing

```python
from src.fusion_engine import FusionEngine
from collections import deque

class RealTimeProcessor:
    def __init__(self):
        self.fusion = FusionEngine()
        self.detection_history = deque(maxlen=10)  # Keep last 10 frames
    
    def process_frame(self, camera_data, lidar_data, radar_data):
        """Process a single frame from all sensors"""
        results = self.fusion.process(
            camera_data=camera_data,
            lidar_data=lidar_data,
            radar_data=radar_data
        )
        
        # Store in history
        self.detection_history.append(results)
        
        # Get trends
        avg_objects = sum(
            r['num_fused_objects'] 
            for r in self.detection_history
        ) / len(self.detection_history)
        
        return results, avg_objects
    
    def get_trajectory(self, object_id, num_frames=5):
        """Get object trajectory over last N frames"""
        trajectory = []
        
        for frame_results in list(self.detection_history)[-num_frames:]:
            for det in frame_results['fused_detections']:
                if det['id'] == object_id:
                    trajectory.append(det['position_3d'])
        
        return trajectory

# Usage
processor = RealTimeProcessor()

# In main loop:
# results, avg = processor.process_frame(cam_data, lidar_data, radar_data)
# trajectory = processor.get_trajectory(object_id=1)
```

### Scenario 4: Filtering by Detection Type

```python
from src.radar_detector import RadarDetector

def analyze_traffic(radar_data_path):
    """Analyze traffic using radar"""
    detector = RadarDetector()
    results = detector.detect(radar_data_path)
    
    # Get moving vehicles
    moving = detector.get_moving_objects(results)
    approaching = [d for d in moving 
                  if d.attributes['doppler_velocity'] > 0]
    receding = [d for d in moving 
               if d.attributes['doppler_velocity'] < 0]
    
    print(f"Moving vehicles: {len(moving)}")
    print(f"Approaching: {len(approaching)}")
    print(f"Receding: {len(receding)}")
    
    # Find closest approaching vehicle
    if approaching:
        closest = min(approaching, 
                     key=lambda d: d.attributes['range'])
        print(f"\nClosest approaching:")
        print(f"  Distance: {closest.attributes['range']:.1f}m")
        print(f"  Speed: {abs(closest.attributes['doppler_velocity']):.1f}m/s")

# Usage
analyze_traffic("radar.json")
```

### Scenario 5: Custom Analysis Pipeline

```python
from src.camera_detector import CameraDetector
from src.lidar_detector import LiDARDetector
import json

def create_detailed_report(image_path, pointcloud_path):
    """Create detailed analysis report"""
    
    report = {
        "camera_analysis": {},
        "lidar_analysis": {},
        "comparison": {}
    }
    
    # Camera analysis
    cam_detector = CameraDetector()
    cam_results = cam_detector.detect(image_path)
    
    report["camera_analysis"] = {
        "total_detections": len(cam_results),
        "by_class": {},
        "confidence_stats": {}
    }
    
    for det in cam_results:
        cls = det.class_name
        if cls not in report["camera_analysis"]["by_class"]:
            report["camera_analysis"]["by_class"][cls] = 0
        report["camera_analysis"]["by_class"][cls] += 1
    
    # LiDAR analysis
    lidar_detector = LiDARDetector()
    lidar_results = lidar_detector.detect(pointcloud_path)
    
    report["lidar_analysis"] = {
        "total_detections": len(lidar_results),
        "distance_range": {},
        "by_class": {}
    }
    
    for det in lidar_results:
        dist = det.attributes.get('distance', 0)
        cls = det.class_name
        
        if cls not in report["lidar_analysis"]["by_class"]:
            report["lidar_analysis"]["by_class"][cls] = 0
        report["lidar_analysis"]["by_class"][cls] += 1
    
    # Comparison
    report["comparison"] = {
        "total_camera": len(cam_results),
        "total_lidar": len(lidar_results),
        "agreement_rate": len(cam_results) / max(len(lidar_results), 1)
    }
    
    # Save report
    with open("report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    return report

# Usage
report = create_detailed_report("image.jpg", "pointcloud.pcd")
print(json.dumps(report, indent=2))
```

---

## Debugging & Troubleshooting

### Debug Mode: Verbose Output

```python
from src.camera_detector import CameraDetector

# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

detector = CameraDetector()
results = detector.detect("image.jpg")

# Print all detection attributes
for det in results:
    print(f"\n{det.class_name}:")
    print(det.to_dict())
```

### Inspecting Detections

```python
def debug_detections(detections):
    """Print detailed detection info"""
    for i, det in enumerate(detections):
        print(f"\n[{i}] {det.class_name}")
        print(f"  Confidence: {det.confidence}")
        print(f"  All attributes:")
        for key, value in det.attributes.items():
            print(f"    - {key}: {value}")

# Usage
debug_detections(results)
```

### Validation Checks

```python
def validate_detections(detections):
    """Validate detection data"""
    issues = []
    
    for i, det in enumerate(detections):
        # Check confidence range
        if not (0 <= det.confidence <= 1):
            issues.append(f"Detection {i}: Invalid confidence {det.confidence}")
        
        # Check bbox for camera
        if 'bbox_2d' in det.attributes:
            bbox = det.attributes['bbox_2d']
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                issues.append(f"Detection {i}: Invalid bbox {bbox}")
        
        # Check bbox for 3D
        if 'bbox_3d' in det.attributes:
            bbox = det.attributes['bbox_3d']
            if 'center' not in bbox:
                issues.append(f"Detection {i}: Missing 3D center")
    
    return issues

# Usage
issues = validate_detections(results)
if issues:
    print("Validation issues:")
    for issue in issues:
        print(f"  - {issue}")
```

---

## Performance Analysis

### Timing Profiler

```python
import time
from src.fusion_engine import FusionEngine

def profile_detectors():
    """Profile detection speed"""
    fusion = FusionEngine()
    
    times = {
        "camera": [],
        "lidar": [],
        "radar": [],
        "fusion": []
    }
    
    num_runs = 5
    
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}...")
        
        # Time camera
        start = time.time()
        fusion.camera_detector.detect("examples/sample_data/sample_camera_image.jpg")
        times["camera"].append(time.time() - start)
        
        # Time LiDAR
        start = time.time()
        fusion.lidar_detector.detect("examples/sample_data/sample_lidar_data.json")
        times["lidar"].append(time.time() - start)
        
        # Time Radar
        start = time.time()
        fusion.radar_detector.detect("examples/sample_data/sample_radar_data.json")
        times["radar"].append(time.time() - start)
    
    # Print results
    print("\nPerformance Results:")
    print("=" * 50)
    for sensor, timings in times.items():
        avg = sum(timings) / len(timings)
        print(f"{sensor.upper():10s} avg: {avg*1000:6.2f}ms (fps: {1/avg:5.1f})")

# Usage
profile_detectors()
```

---

## Best Practices Summary

1. **Always validate input data** before processing
2. **Use confidence filtering** to reduce false positives
3. **Start with smaller models** for testing (yolov8n)
4. **Profile your code** to find bottlenecks
5. **Export results** for analysis and debugging
6. **Use version control** for your detection pipelines
7. **Document sensor calibration** for accurate fusion
8. **Test with diverse data** for robustness

---

**Last Updated**: February 2026
**Version**: 1.0.0
