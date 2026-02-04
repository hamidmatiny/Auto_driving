# API Documentation

## Overview

Complete API reference for the Multi-Modal Object Detection system.

## Core Classes

### Detection

Represents a single object detection result.

```python
class Detection:
    def __init__(self, class_name: str, confidence: float, **kwargs)
```

**Attributes:**
- `class_name` (str): Object class label
- `confidence` (float): Detection confidence score (0-1)
- `attributes` (dict): Additional detection-specific attributes

**Methods:**

```python
def to_dict() -> Dict[str, Any]:
    """Convert detection to dictionary representation"""
```

**Example:**
```python
detection = Detection(
    class_name="car",
    confidence=0.95,
    bbox_2d=[100, 200, 300, 400],
    distance=25.5
)
print(detection)  # Detection(class=car, confidence=0.95)
print(detection.to_dict())
```

---

### BaseDetector

Abstract base class for all detector types.

```python
class BaseDetector(ABC):
    def __init__(self, config: Dict[str, Any] = None)
```

**Abstract Methods:**

```python
@abstractmethod
def detect(self, data_input: Any) -> List[Detection]:
    """Perform object detection on input data"""

@abstractmethod
def visualize_results(self, detections: List[Detection], 
                     data_input: Any = None):
    """Visualize detection results"""
```

**Utility Methods:**

```python
def get_detections(self) -> List[Detection]:
    """Get last detection results"""

def filter_by_confidence(self, detections: List[Detection], 
                         threshold: float) -> List[Detection]:
    """Filter detections by confidence threshold"""

def filter_by_class(self, detections: List[Detection], 
                   classes: List[str]) -> List[Detection]:
    """Filter detections by class names"""

def print_detections(self, detections: List[Detection] = None):
    """Print detection results in human-readable format"""
```

---

## Camera Detector

### CameraDetector

2D object detection using YOLOv8.

```python
class CameraDetector(BaseDetector):
    def __init__(self, model_name: str = "yolov8n", 
                 conf_threshold: float = 0.5)
```

**Parameters:**
- `model_name` (str): YOLOv8 variant ('n', 's', 'm', 'l', 'x')
- `conf_threshold` (float): Minimum confidence threshold

**Methods:**

#### detect()

```python
def detect(self, image_path: str) -> List[Detection]:
```

Detect objects in an image.

**Parameters:**
- `image_path` (str): Path to image file (JPG, PNG, etc.)

**Returns:**
- List of Detection objects with 2D bounding boxes

**Example:**
```python
detector = CameraDetector(model_name="yolov8m", conf_threshold=0.6)
detections = detector.detect("path/to/image.jpg")
for det in detections:
    print(f"{det.class_name}: {det.confidence:.2%}")
```

#### visualize_results()

```python
def visualize_results(self, detections: List[Detection] = None, 
                     save_path: str = None):
```

Visualize detections on image with bounding boxes.

**Parameters:**
- `detections` (List[Detection], optional): Detections to visualize
- `save_path` (str, optional): Path to save annotated image

**Example:**
```python
detector.visualize_results(
    detections=detections,
    save_path="annotated.jpg"
)
```

#### export_detections()

```python
def export_detections(self, format: str = "json") -> Dict[str, Any]:
```

Export detections in specified format.

**Parameters:**
- `format` (str): Export format ('json', 'xml', 'yolo')

**Returns:**
- Dictionary with export data

**Example:**
```python
export_data = detector.export_detections(format="json")
# Contains: model name, confidence threshold, detections list
```

---

## LiDAR Detector

### LiDARDetector

3D object detection from point clouds.

```python
class LiDARDetector(BaseDetector):
    def __init__(self, voxel_size: float = 0.05, 
                 min_points: int = 5)
```

**Parameters:**
- `voxel_size` (float): Voxel size for grid-based clustering (meters)
- `min_points` (int): Minimum points required to form a cluster

**Methods:**

#### detect()

```python
def detect(self, pointcloud_path: str) -> List[Detection]:
```

Detect objects in point cloud.

**Parameters:**
- `pointcloud_path` (str): Path to point cloud file (.pcd, .ply, .json, .npy)

**Returns:**
- List of Detection objects with 3D bounding boxes

**Example:**
```python
detector = LiDARDetector(voxel_size=0.1, min_points=10)
detections = detector.detect("pointcloud.pcd")

for det in detections:
    center = det.attributes["bbox_3d"]["center"]
    print(f"{det.class_name} at {center}")
```

#### visualize_results()

```python
def visualize_results(self, detections: List[Detection] = None, 
                     save_path: str = None):
```

Visualize 3D detections in Open3D viewer.

**Example:**
```python
detector.visualize_results(detections=detections)
```

#### export_detections()

```python
def export_detections(self, format: str = "json") -> Dict[str, Any]:
```

Export detections in specified format.

---

## Radar Detector

### RadarDetector

Velocity and range estimation from radar data.

```python
class RadarDetector(BaseDetector):
    def __init__(self, range_threshold: float = 100.0, 
                 min_doppler: float = 0.5)
```

**Parameters:**
- `range_threshold` (float): Maximum detection range (meters)
- `min_doppler` (float): Minimum Doppler velocity threshold (m/s)

**Methods:**

#### detect()

```python
def detect(self, radar_data_path: str) -> List[Detection]:
```

Detect objects from radar data.

**Parameters:**
- `radar_data_path` (str): Path to radar data file (JSON format)

**Returns:**
- List of Detection objects with range, azimuth, and velocity

**Example:**
```python
detector = RadarDetector(range_threshold=150.0, min_doppler=1.0)
detections = detector.detect("radar_data.json")

for det in detections:
    rng = det.attributes["range"]
    vel = det.attributes["doppler_velocity"]
    print(f"{det.class_name}: {rng}m away, {vel}m/s velocity")
```

#### get_moving_objects()

```python
def get_moving_objects(self, 
                      detections: List[Detection] = None) -> List[Detection]:
```

Get only moving objects (non-zero Doppler velocity).

**Example:**
```python
moving = detector.get_moving_objects()
print(f"Moving objects: {len(moving)}")
```

#### get_stationary_objects()

```python
def get_stationary_objects(self, 
                          detections: List[Detection] = None) -> List[Detection]:
```

Get only stationary objects.

#### generate_range_doppler_map()

```python
def generate_range_doppler_map(self, num_range_bins: int = 256, 
                              num_doppler_bins: int = 128) -> np.ndarray:
```

Generate 2D range-Doppler map.

**Returns:**
- 2D numpy array of shape (range_bins, doppler_bins)

**Example:**
```python
rdm = detector.generate_range_doppler_map()
print(f"Map shape: {rdm.shape}")
print(f"Peak SNR: {rdm.max():.2f} dB")
```

---

## Sensor Fusion Engine

### FusionEngine

Multi-modal sensor fusion combining all three modalities.

```python
class FusionEngine:
    def __init__(self, camera_weight: float = 0.4, 
                 lidar_weight: float = 0.4,
                 radar_weight: float = 0.2)
```

**Parameters:**
- `camera_weight` (float): Weight for camera confidence
- `lidar_weight` (float): Weight for LiDAR confidence
- `radar_weight` (float): Weight for radar confidence

**Methods:**

#### process()

```python
def process(self, camera_data: str = None, 
            lidar_data: str = None,
            radar_data: str = None) -> Dict[str, Any]:
```

Process multi-modal sensor data and fuse detections.

**Parameters:**
- `camera_data` (str, optional): Path to camera image
- `lidar_data` (str, optional): Path to LiDAR point cloud
- `radar_data` (str, optional): Path to radar data

**Returns:**
- Dictionary with fused detections and metadata

**Example:**
```python
fusion = FusionEngine(
    camera_weight=0.4,
    lidar_weight=0.4,
    radar_weight=0.2
)

results = fusion.process(
    camera_data="image.jpg",
    lidar_data="pointcloud.pcd",
    radar_data="radar.json"
)

print(f"Total objects: {results['num_fused_objects']}")
```

#### visualize_fusion_results()

```python
def visualize_fusion_results(self, results: Dict[str, Any] = None, 
                            save_path: str = None):
```

Visualize fusion results and statistics.

**Example:**
```python
fusion.visualize_fusion_results(results)
```

#### export_results()

```python
def export_results(self, format: str = "json", 
                  output_path: str = None) -> str:
```

Export fusion results.

**Parameters:**
- `format` (str): Export format ('json', 'xml', 'yolo')
- `output_path` (str, optional): Path to save results

**Returns:**
- Exported data as string

**Example:**
```python
exported = fusion.export_results(
    format="json",
    output_path="fused_results.json"
)
```

---

## Data Structures

### Detection Output Format

#### Camera Detection

```python
{
    "class": "car",
    "confidence": 0.95,
    "bbox_2d": [x1, y1, x2, y2],
    "bbox_center": [x_center, y_center],
    "bbox_width": width,
    "bbox_height": height,
    "class_id": 0
}
```

#### LiDAR Detection

```python
{
    "class": "car",
    "confidence": 0.92,
    "bbox_3d": {
        "center": [x, y, z],
        "size": [length, width, height],
        "min": [x_min, y_min, z_min],
        "max": [x_max, y_max, z_max]
    },
    "distance": 25.5,
    "num_points": 500,
    "density": 150.0,
    "cluster_id": 1
}
```

#### Radar Detection

```python
{
    "class": "vehicle",
    "confidence": 0.88,
    "range": 25.5,
    "azimuth": -15.0,
    "doppler_velocity": 12.3,
    "rcs": -5.0,
    "snr": 18.5,
    "position_cartesian": [x, y, z],
    "position_polar": [range, azimuth],
    "object_id": 1
}
```

### Fused Detection Format

```python
{
    "id": 1,
    "class": "car",
    "position_3d": [x, y, z],
    "size_3d": [length, width, height],
    "confidence": 0.92,
    "sensor_contributions": {
        "camera": {
            "confidence": 0.95,
            "bbox_2d": [x1, y1, x2, y2]
        },
        "lidar": {
            "confidence": 0.90,
            "distance": 25.5
        },
        "radar": {
            "confidence": 0.88,
            "velocity": [vx, vy, vz],
            "range": 25.5
        }
    }
}
```

---

## Usage Examples

### Example 1: Basic Camera Detection

```python
from src.camera_detector import CameraDetector

# Initialize
detector = CameraDetector(model_name="yolov8m")

# Detect
results = detector.detect("image.jpg")

# Filter
cars = detector.filter_by_class(results, ["car", "truck"])
high_conf = detector.filter_by_confidence(cars, 0.7)

# Visualize
detector.visualize_results(high_conf, save_path="output.jpg")

# Export
data = detector.export_detections(format="json")
```

### Example 2: LiDAR 3D Detection Pipeline

```python
from src.lidar_detector import LiDARDetector
import json

# Initialize
detector = LiDARDetector(voxel_size=0.05)

# Detect
results = detector.detect("pointcloud.pcd")

# Process
vehicles = detector.filter_by_class(results, ["car", "truck"])

# Export
export_data = detector.export_detections(format="json")
with open("output.json", "w") as f:
    json.dump(export_data, f, indent=2)
```

### Example 3: Complete Multi-Modal Pipeline

```python
from src.fusion_engine import FusionEngine

# Initialize
fusion = FusionEngine(
    camera_weight=0.35,
    lidar_weight=0.45,
    radar_weight=0.20
)

# Process
results = fusion.process(
    camera_data="image.jpg",
    lidar_data="pointcloud.pcd",
    radar_data="radar.json"
)

# Visualize
fusion.visualize_fusion_results(results)

# Export
fusion.export_results(
    format="json",
    output_path="fused_detections.json"
)

# Analyze
print(f"Objects detected: {results['num_fused_objects']}")
for det in results['fused_detections']:
    print(f"  {det['class']}: confidence {det['confidence']:.2%}")
```

---

## Error Handling

### Common Errors

#### FileNotFoundError

```python
try:
    results = detector.detect("nonexistent.jpg")
except FileNotFoundError as e:
    print(f"Error: {e}")
    # Handle file not found
```

#### ModelNotFoundError

```python
try:
    detector = CameraDetector(model_name="invalid_model")
except Exception as e:
    print(f"Model error: {e}")
    # Fall back to default model
```

#### InvalidDataFormat

```python
try:
    results = detector.detect("invalid_format.txt")
except Exception as e:
    print(f"Data format error: {e}")
    # Validate and preprocess data
```

---

## Performance Optimization

### Tips for Faster Processing

1. **Use smaller models**
   ```python
   detector = CameraDetector(model_name="yolov8n")  # Nano
   ```

2. **Enable GPU**
   ```yaml
   models:
     camera:
       device: "cuda"
   ```

3. **Batch processing**
   ```python
   images = ["img1.jpg", "img2.jpg", "img3.jpg"]
   for img in images:
       results = detector.detect(img)
   ```

4. **Filter early**
   ```python
   results = detector.detect("image.jpg")
   high_conf = detector.filter_by_confidence(results, 0.7)
   ```

---

## Best Practices

1. **Always validate input data**
2. **Filter by confidence threshold**
3. **Handle exceptions gracefully**
4. **Export results for debugging**
5. **Use appropriate sensor weights for fusion**
6. **Coordinate system consistency**
7. **Regular model updates**

---

**Last Updated**: February 2026
**Version**: 1.0.0
